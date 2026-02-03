
# benchmark_offline.py
# Adapted from mem0bench.py to benchmark OfflineAPI
# Run: python benchmark_offline.py --data_path ... --out_dir ... --workers 4 --recall_ks 3 5 10

import os
import json
import time
import uuid
import argparse
import traceback
import sys
from typing import Any, Dict, List, Tuple, Set, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

# Ensure we can import from local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from offline_api import OfflineAPI
from config import APIConfig, StoreConfig, PromptConfig
from openai import OpenAI

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--vllm_model_url", type=str, default="http://localhost:8002/v1")
    p.add_argument("--vllm_model_name", type=str, default="Qwen/Qwen3-4B-Instruct-2507-FP8")
    p.add_argument("--vllm_embed_url", type=str, default="http://localhost:8001/v1")
    p.add_argument("--vllm_embed_name", type=str, default="Qwen/Qwen3-Embedding-0.6B")
    p.add_argument("--api_key", type=str, default="EMPTY")
    
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--out_dir", type=str, default="./outputs_offline")
    
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--recall_ks", type=int, nargs="+", default=list(range(1, 21)))
    
    p.add_argument("--max_items", type=int, default=-1)
    
    # Resume behavior
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--no-resume", dest="resume", action="store_false")
    
    # OfflineAPI specific tuning
    p.add_argument("--candidate_n", type=int, default=30)
    p.add_argument("--gate_keep_max", type=int, default=20)
    p.add_argument("--use_gate", type=bool, default=True)
    p.add_argument("--use_rerank", type=bool, default=True)

    return p.parse_args()

def ensure_out_dir(path: str):
    os.makedirs(path, exist_ok=True)

def atomic_write_json(path: str, obj: Dict[str, Any]):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)

def append_jsonl(path: str, obj: Dict[str, Any]):
    line = json.dumps(obj, ensure_ascii=False)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())

def load_done_question_ids(out_dir: str) -> Set[str]:
    done: Set[str] = set()
    # 1) scan item_*.json
    if os.path.exists(out_dir):
        for name in os.listdir(out_dir):
            if not (name.startswith("item_") and name.endswith(".json")):
                continue
            fp = os.path.join(out_dir, name)
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    obj = json.load(f)
                qid = obj.get("question_id")
                if qid is not None:
                    done.add(str(qid))
            except Exception:
                pass

    # 2) scan progress.jsonl
    prog = os.path.join(out_dir, "progress.jsonl")
    if os.path.exists(prog):
        try:
            with open(prog, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        if obj.get("status") == "OK" and obj.get("question_id") is not None:
                            done.add(str(obj["question_id"]))
                    except Exception:
                        continue
        except Exception:
            pass
    return done

def offline_test_work(
    data: Dict[str, Any],
    api_cfg: APIConfig,
    base_store_cfg: StoreConfig,
    prompt_cfg: PromptConfig,
    user_id: str,
    recall_ks: List[int],
    args,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:

    # We need a unique storage per user effectively, which is handled by user_id.
    # But we want to ensure we don't mix data if we rerun. 
    # StoreConfig has USERS_DIR. offline_api uses that.
    
    # Initialize OfflineAPI
    # Note: We create a fresh instance per item to emulate fresh starts, 
    # but the storage is persistent on disk.
    store_cfg = StoreConfig(
        STORAGE_DIR=base_store_cfg.STORAGE_DIR,
        EMBEDDING_DIMS=base_store_cfg.EMBEDDING_DIMS
    )
    # We must ensure subdirectory definitions match the class logic if we manually instantiated,
    # but passing it to OfflineAPI is enough as it defaults USERS_DIR based on STORAGE_DIR if not set?
    # Actually StoreConfig sets USERS_DIR in __init__ (as field default with os.path.join which executes at module level? No, dataclass default)
    # Dataclass with default value uses the value at definition time.
    # The definitions in config.py are:
    # USERS_DIR: str = os.path.join(STORAGE_DIR, "users")
    # This is problematic because STORAGE_DIR is a field too.
    # In config.py:
    # USERS_DIR: str = os.path.join(STORAGE_DIR, "users") 
    # references the class variable or global?
    # Actually in python dataclass, one field cannot reference another default easily unless using __post_init__.
    # But config.py seems to rely on os.getenv at module level for defaults.
    # We should just rely on OfflineAPI internals to handle paths or ensure we pass valid config.
    # Let's verify config.py again...
    # It says: `STORAGE_DIR: str = os.getenv(...)`; `USERS_DIR: str = os.path.join(STORAGE_DIR, "users")`
    # This `os.path.join(STORAGE_DIR, ...)` uses the `STORAGE_DIR` *variable* defined in the module scope or previous line? 
    # In the `config.py` provided:
    # `STORAGE_DIR: str = os.getenv("STORAGE_DIR", "./storage")`
    # `USERS_DIR: str = os.path.join(STORAGE_DIR, "users")`
    # This works if they are just variable assignments in the class body.
    # So if we instantiate `StoreConfig(STORAGE_DIR="/new/path")`, `USERS_DIR` will NOT automatically update unless we update it too.
    # So we must update USERS_DIR manually.
    
    store_cfg.USERS_DIR = os.path.join(store_cfg.STORAGE_DIR, "users")
    
    offline = OfflineAPI(api_cfg=api_cfg, store_cfg=store_cfg, prompt_cfg=prompt_cfg)
    
    write_log: List[Dict[str, Any]] = []
    question_log: List[Dict[str, Any]] = []

    answer_session_ids = set(data["answer_session_ids"])
    haystack_session_ids = data["haystack_session_ids"]
    haystack_sessions = data["haystack_sessions"]
    haystack_dates = data["haystack_dates"]

    # 1. Add History
    conv_id_base = f"conv_{uuid.uuid4().hex[:6]}"
    
    for i in range(len(haystack_session_ids)):
        session_id = haystack_session_ids[i]
        dt = haystack_dates[i]
        raw_msgs = haystack_sessions[i]
        
        # Adapt messages: OfflineAPI skips system messages and expects user/assistant pairs.
        # We prepend date to the first user message to preserve context.
        # And we assume the session is [User, Assistant, User, Assistant...]
        
        adapted_msgs = []
        if raw_msgs and raw_msgs[0]['role'] == 'user':
            # Prepend date
            first_content = f"[{dt}] {raw_msgs[0]['content']}"
            adapted_msgs.append({"role": "user", "content": first_content})
            adapted_msgs.extend(raw_msgs[1:])
        else:
            # If starts with system or assistant, we might lose context if we don't handle it.
            # Assuming standard user-init conversations for now.
            adapted_msgs = list(raw_msgs)
            if adapted_msgs and adapted_msgs[0]['role'] == 'user':
                 adapted_msgs[0]['content'] = f"[{dt}] {adapted_msgs[0]['content']}"

        # We use a unique conv_id per session to mimic "sessions" or keep same conv_id?
        # mem0bench uses "haystack_session_ids" which suggests distinct sessions.
        # But OfflineAPI treats conv_id as a thread.
        # If we use different conv_ids, they are all stored.
        # Retrieval searches across all conversations of the user.
        # So using session_id as conv_id is appropriate.
        current_conv_id = f"{conv_id_base}_{session_id}"

        t0 = time.time()
        # added_ids = offline.add_history(user_id=user_id, conv_id=current_conv_id, history_messages=adapted_msgs)
        # However, OfflineAPI.add_history expects exact pairs. 
        # We iterate and add what we can.
        added_ids = offline.add_history(user_id, current_conv_id, adapted_msgs)
        t1 = time.time()
        
        write_log.append({
            "time_cost": t1 - t0,
            "mem_ids": added_ids,
            "new_mem_cnt": len(added_ids),
            "session_id": session_id,
            "datetime": dt,
            "is_answer": session_id in answer_session_ids,
        })
    
    # 2. Question / Retrieval
    question = data["question"]
    question_date = data["question_date"]
    answer = data["answer"]
    
    # We want to perform retrieval + generation for each K.
    # OfflineAPI.chat does this. But it also updates history.
    # We want to AVOID updating history during this loop so each K sees the same state.
    # We will manually invoke the pipeline components from offline.mem
    
    # Prepare prompt inputs
    # Recent turns from the "current" conversation? 
    # In this benchmark, the question starts a NEW conversation (or turn).
    # so recent_msgs is empty.
    
    # We'll use a dummy conv_id for the question to avoid excluding valuable history 
    # (since exclude_conv_id excludes the current one).
    query_conv_id = "bench_query_conv"

    for k in recall_ks:
        t0 = time.time()
        
        # A. Retrieval (mirrors OfflineAPI.chat steps 1-2)
        # 1. load recent (empty for new query conv)
        recent_msgs, recent_turn_ids = [], []
        
        # 2. retrieve bundle
        bundle, _debug = offline.mem.retrieve_memory_bundle(
            user_id=user_id,
            query=question,
            k_replay=k,
            candidate_n=args.candidate_n,
            use_gate=args.use_gate,
            gate_keep_max=args.gate_keep_max,
            use_rerank=args.use_rerank,
            exclude_conv_id=query_conv_id,
            exclude_turn_ids=set(recent_turn_ids),
        )
        replay = bundle["replay"]
        summary_only = bundle["summary_only"]
        
        t1 = time.time() # retrieval done
        
        # B. Generation (mirrors OfflineAPI.chat steps 3-4)
        # 3. build prompt
        # We need to manually inject datetime into the system prompt or user query 
        # because OfflineAPI doesn't have a standardized way to set "current time" 
        # other than what's in the prompt.
        
        # OfflineAPI._build_prompt uses self.prompt_cfg.CHAT_SYSTEM.
        # We'll prepend the date to the user query for the LLM to know context.
        query_with_date = f"[{question_date}] {question}"
        
        final_messages = offline._build_prompt(
            user_query=query_with_date,
            recent_msgs=recent_msgs,
            replay=replay,
            summary_only=summary_only,
        )
        
        # 4. llm chat
        response_text = offline.mem.llm_chat(
            final_messages,
            temperature=0.7, # Match mem0bench
            max_tokens=8192, # Match mem0bench
        )
        
        t2 = time.time() # generation done

        # We construct a result dict similar to mem0bench
        # mem0bench: recall_mems is the raw list from search.
        # We'll store our bundle info.
        
        question_log.append({
            "question_id": data.get("question_id"),
            "question": question,
            "question_type": data.get("question_type"),
            "answer": answer,
            "response": response_text,
            "recall_k": int(k),
            "recall_mems": {
                "replay": [x["memory_id"] for x in replay],
                "summary": [x["memory_id"] for x in summary_only]
            },
            "recall_time_cost": t1 - t0,
            "response_time_cost": t2 - t1,
        })
        
        # We DO NOT call update_conversation here, preserving state for next K.

    return write_log, question_log

def run_one_item(
    idx: int,
    item: Dict[str, Any],
    api_cfg: APIConfig,
    base_store_cfg: StoreConfig,
    prompt_cfg: PromptConfig,
    args,
) -> Dict[str, Any]:
    run_id = uuid.uuid4().hex[:10]
    user_id = f"user_{idx}_{run_id}"
    
    # We can use a subdirectory for storage if we want to isolate runs on disk,
    # but distinct user_ids is enough for isolation within the same storage dir.
    # mem0bench creates ./tmp/{collection_name}.
    # We will use base_store_cfg as is, and user_id handles isolation.
    
    t0 = time.time()
    write_log, question_log = offline_test_work(
        data=item,
        api_cfg=api_cfg,
        base_store_cfg=base_store_cfg,
        prompt_cfg=prompt_cfg,
        user_id=user_id,
        recall_ks=args.recall_ks,
        args=args
    )
    t1 = time.time()

    write_time = sum(x.get("time_cost", 0.0) for x in write_log)
    new_mem_total = sum(int(x.get("new_mem_cnt", 0)) for x in write_log)
    recall_time = sum(x.get("recall_time_cost", 0.0) for x in question_log)
    resp_time = sum(x.get("response_time_cost", 0.0) for x in question_log)

    return {
        "idx": idx,
        "question_id": item.get("question_id"),
        "question_type": item.get("question_type"),
        "recall_ks": args.recall_ks,
        "user_id": user_id,
        "elapsed_total": t1 - t0,
        "write_time_total": write_time,
        "new_mem_total": new_mem_total,
        "recall_time_total": recall_time,
        "response_time_total": resp_time,
        "write_log": write_log,
        "question_log": question_log,
        "item_json": "" # Filled by caller
    }

def main():
    args = parse_args()
    ensure_out_dir(args.out_dir)

    # Configs
    api_cfg = APIConfig(
        API_KEY=args.api_key,
        LLM_BASE_URL=args.vllm_model_url,
        EMB_BASE_URL=args.vllm_embed_url,
        LLM_MODEL=args.vllm_model_name,
        EMB_MODEL=args.vllm_embed_name
    )
    
    # Use a subdir for this run execution to avoid pollution if resumed?
    # No, resume logic relies on outputs in out_dir.
    # Storage should be persistent or transient?
    # mem0bench uses ./tmp/{collection} which is transient effectively.
    # We'll use os.path.join(args.out_dir, "storage") to keep it contained.
    storage_dir = os.path.join(args.out_dir, "storage")
    store_cfg = StoreConfig(STORAGE_DIR=storage_dir) # USERS_DIR will need update in loop
    
    prompt_cfg = PromptConfig()

    os.environ["OPENAI_API_KEY"] = args.api_key

    # Load data
    with open(args.data_path, "r", encoding="utf-8") as f:
        ds = json.load(f)

    if args.max_items is not None and args.max_items > 0:
        ds = ds[: args.max_items]

    # Resume
    done_qids: Set[str] = set()
    if args.resume:
        done_qids = load_done_question_ids(args.out_dir)
        if done_qids:
            print(f"[RESUME] Found {len(done_qids)} done question_id(s) in {args.out_dir}")

    progress_path = os.path.join(args.out_dir, "progress.jsonl")
    errors_path = os.path.join(args.out_dir, "errors.jsonl")

    submitted = 0
    skipped = 0
    
    t0 = time.time()
    results_meta: List[Dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = []
        for idx, item in enumerate(ds):
            qid = item.get("question_id")
            qid_s = str(qid) if qid is not None else None
            
            if qid_s is not None and qid_s in done_qids:
                skipped += 1
                append_jsonl(progress_path, {
                    "ts": time.time(),
                    "status": "SKIP",
                    "idx": idx,
                    "question_id": qid_s
                })
                continue
            
            futs.append(ex.submit(run_one_item, idx, item, api_cfg, store_cfg, prompt_cfg, args))
            submitted += 1

        print(f"[SUBMIT] submitted={submitted}, skipped={skipped}, total={len(ds)}")

        for fut in as_completed(futs):
            finished_ts = time.time()
            try:
                r = fut.result()
                
                # Atomic save
                out_path = os.path.join(args.out_dir, f"item_{r['idx']:05d}.json")
                r["item_json"] = os.path.basename(out_path)
                atomic_write_json(out_path, r)
                
                append_jsonl(progress_path, {
                    "ts": finished_ts,
                    "status": "OK",
                    "idx": r.get("idx"),
                    "question_id": str(r.get("question_id")),
                    "elapsed_total": r.get("elapsed_total"),
                    "item_json": r.get("item_json")
                })
                
                results_meta.append({
                    "idx": r.get("idx"),
                    "question_id": r.get("question_id"),
                    "question_type": r.get("question_type"),
                    "elapsed_total": r.get("elapsed_total"),
                    "item_json": r.get("item_json"),
                })
                
                print(f"[OK] idx={r['idx']} qid={r.get('question_id')} elapsed={r['elapsed_total']:.2f}s")
                
            except Exception as e:
                tb = traceback.format_exc()
                print(f"[ERR] {e}")
                print(tb)
                append_jsonl(errors_path, {
                    "ts": finished_ts, 
                    "status": "ERR", 
                    "error": str(e), 
                    "traceback": tb
                })
                
    t1 = time.time()
    
    # Final summary
    summary = {
        "data_path": args.data_path,
        "out_dir": args.out_dir,
        "total": len(ds),
        "submitted": submitted,
        "skipped": skipped,
        "elapsed": t1 - t0,
        "progress_jsonl": os.path.basename(progress_path)
    }
    atomic_write_json(os.path.join(args.out_dir, "summary.json"), summary)
    
    # CSV
    csv_path = os.path.join(args.out_dir, "summary.csv")
    with open(csv_path + ".tmp", "w", encoding="utf-8") as f:
        f.write("idx,question_id,question_type,elapsed,json\n")
        for r in sorted(results_meta, key=lambda x: x.get("idx", 0)):
            f.write(f"{r['idx']},{r['question_id']},{r['question_type']},{r['elapsed_total']:.4f},{r['item_json']}\n")
    os.replace(csv_path + ".tmp", csv_path)
    
    print("\n==== DONE ====")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
