# memory.py
import os
import re
import json
import time
import uuid
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import faiss
from openai import OpenAI

from config import APIConfig, StoreConfig, PromptConfig


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _now() -> float:
    return time.time()


def _sanitize_id(s: str, default: str = "default", max_len: int = 128) -> str:
    s = (s or "").strip()
    if not s:
        return default
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", s)[:max_len]


class UserMemory:
    """
    Isolated memory store for ONE user:
      - FAISS index (cosine via IP on normalized vectors)
      - meta.jsonl aligned with FAISS order
      - state.json stores embedding_dims
      - conversations/<conv_id>.jsonl stores full sequential conversation turns
    """

    def __init__(self, user_dir: str, embedding_dims_hint: int = 0):
        self.user_dir = user_dir
        _ensure_dir(self.user_dir)

        self.index_path = os.path.join(self.user_dir, "faiss.index")
        self.meta_path = os.path.join(self.user_dir, "meta.jsonl")
        self.state_path = os.path.join(self.user_dir, "state.json")

        self.conv_dir = os.path.join(self.user_dir, "conversations")
        _ensure_dir(self.conv_dir)

        self.embedding_dims: int = 0
        self.index = None  # type: ignore
        self.meta: List[Dict[str, Any]] = []

        self._load_state(embedding_dims_hint)
        self._load_meta()
        self._load_index()

        if self.index is not None and self.index.ntotal != len(self.meta):
            # Avoid crashing; user can rebuild by deleting index/meta if needed.
            print(f"[WARN] FAISS ntotal={self.index.ntotal} != meta={len(self.meta)} for {self.user_dir}")

    # -------------------- state --------------------
    def _load_state(self, hint: int):
        if os.path.exists(self.state_path):
            try:
                with open(self.state_path, "r", encoding="utf-8") as f:
                    st = json.load(f)
                self.embedding_dims = int(st.get("embedding_dims", 0))
            except Exception:
                self.embedding_dims = 0

        if self.embedding_dims == 0 and hint > 0:
            self.embedding_dims = int(hint)
            self._save_state()

    def _save_state(self):
        with open(self.state_path, "w", encoding="utf-8") as f:
            json.dump({"embedding_dims": self.embedding_dims}, f, ensure_ascii=False, indent=2)

    # -------------------- meta --------------------
    def _load_meta(self):
        self.meta = []
        if os.path.exists(self.meta_path):
            with open(self.meta_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        self.meta.append(json.loads(line))
                    except Exception:
                        continue

    def append_meta(self, item: Dict[str, Any]):
        with open(self.meta_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
        self.meta.append(item)

    # -------------------- index --------------------
    def _load_index(self):
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
            if self.embedding_dims == 0:
                self.embedding_dims = int(self.index.d)
                self._save_state()
        else:
            if self.embedding_dims > 0:
                self.index = faiss.IndexFlatIP(self.embedding_dims)
            else:
                self.index = None  # init when first vector arrives

    def _save_index(self):
        if self.index is not None:
            faiss.write_index(self.index, self.index_path)

    def ensure_index(self, dims: int):
        if self.embedding_dims == 0:
            self.embedding_dims = int(dims)
            self._save_state()
        if self.index is None:
            self.index = faiss.IndexFlatIP(self.embedding_dims)

    def add_vector(self, vec: np.ndarray):
        vec = vec.astype(np.float32).reshape(1, -1)
        self.ensure_index(vec.shape[1])
        faiss.normalize_L2(vec)
        self.index.add(vec)
        self._save_index()

    def search(self, q: np.ndarray, top_n: int) -> List[Dict[str, Any]]:
        if self.index is None or self.index.ntotal == 0:
            return []

        q = q.astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(q)

        scores, idxs = self.index.search(q, top_n)

        out: List[Dict[str, Any]] = []
        for score, idx in zip(scores[0].tolist(), idxs[0].tolist()):
            if 0 <= idx < len(self.meta):
                item = dict(self.meta[idx])
                item["score"] = float(score)
                out.append(item)
        return out

    # -------------------- conversation log --------------------
    def conv_log_path(self, conv_id: str) -> str:
        safe = _sanitize_id(conv_id, default="default", max_len=128)
        return os.path.join(self.conv_dir, f"{safe}.jsonl")

    def _get_next_turn_id(self, path: str) -> int:
        if not os.path.exists(path):
            return 1

        # read last non-empty line using tail chunk
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(size - 8192, 0))
            tail = f.read().decode("utf-8", errors="ignore").splitlines()

        for line in reversed(tail):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                return int(obj.get("turn_id", 0)) + 1
            except Exception:
                continue
        return 1

    def append_turn(self, conv_id: str, user_msg: str, assistant_msg: str) -> Dict[str, Any]:
        path = self.conv_log_path(conv_id)
        turn_id = self._get_next_turn_id(path)
        rec = {
            "turn_id": turn_id,
            "ts": _now(),
            "user": user_msg,
            "assistant": assistant_msg,
        }
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec

    def load_recent_turns(self, conv_id: str, recent_turns: int) -> Tuple[List[Dict[str, str]], List[int]]:
        """
        Returns:
          - recent_msgs: multi-turn messages (user/assistant/user/assistant...)
          - recent_turn_ids: [turn_id, ...] for dedup filtering
        """
        if recent_turns <= 0:
            return [], []

        path = self.conv_log_path(conv_id)
        if not os.path.exists(path):
            return [], []

        # tail read (increase if your turns are extremely large)
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(size - 262144, 0))  # 256KB
            lines = f.read().decode("utf-8", errors="ignore").splitlines()

        turns: List[Dict[str, Any]] = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                turns.append(json.loads(line))
            except Exception:
                continue

        turns = turns[-recent_turns:]

        recent_msgs: List[Dict[str, str]] = []
        recent_ids: List[int] = []
        for t in turns:
            tid = int(t["turn_id"])
            recent_ids.append(tid)
            recent_msgs.append({"role": "user", "content": t.get("user", "")})
            recent_msgs.append({"role": "assistant", "content": t.get("assistant", "")})
        return recent_msgs, recent_ids


class MemoryStore:
    """
    Manager:
      - uses OpenAI package to call vLLM endpoints
      - per-user isolated storage (UserMemory)
      - retrieval: vector -> (exclude recent) -> gate -> rerank -> replay + summary_only
    """

    def __init__(self, api: APIConfig, store: StoreConfig, prompts: PromptConfig):
        self.api = api
        self.store = store
        self.prompts = prompts

        _ensure_dir(self.store.STORAGE_DIR)
        _ensure_dir(self.store.USERS_DIR)

        self.llm_client = OpenAI(base_url=self.api.LLM_BASE_URL, api_key=self.api.API_KEY)
        self.emb_client = OpenAI(base_url=self.api.EMB_BASE_URL, api_key=self.api.API_KEY)

        self._cache: Dict[str, UserMemory] = {}

    def _get_user_mem(self, user_id: str) -> UserMemory:
        uid = _sanitize_id(user_id, default="anonymous", max_len=128)
        if uid in self._cache:
            return self._cache[uid]
        user_dir = os.path.join(self.store.USERS_DIR, uid)
        um = UserMemory(user_dir=user_dir, embedding_dims_hint=self.store.EMBEDDING_DIMS)
        self._cache[uid] = um
        return um

    # ---------- OpenAI calls ----------
    def llm_chat(self, messages: List[Dict[str, str]], temperature: float = 0.2, max_tokens: int = 512) -> str:
        r = self.llm_client.chat.completions.create(
            model=self.api.LLM_MODEL,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return r.choices[0].message.content or ""

    def embed(self, text: str) -> np.ndarray:
        r = self.emb_client.embeddings.create(model=self.api.EMB_MODEL, input=text)
        vec = np.array(r.data[0].embedding, dtype=np.float32)
        return vec

    # ---------- sequential conversation ----------
    def load_recent(self, user_id: str, conv_id: str, recent_turns: int) -> Tuple[List[Dict[str, str]], List[int]]:
        um = self._get_user_mem(user_id)
        return um.load_recent_turns(conv_id, recent_turns)

    def append_conversation_turn(self, user_id: str, conv_id: str, user_msg: str, assistant_msg: str) -> Dict[str, Any]:
        um = self._get_user_mem(user_id)
        return um.append_turn(conv_id, user_msg, assistant_msg)

    # ---------- memory write ----------
    def summarize_turn(self, user_msg: str, assistant_msg: str) -> str:
        msgs = [
            {"role": "system", "content": self.prompts.SUMMARY_SYSTEM},
            {"role": "user", "content": f"USER:\n{user_msg}\n\nASSISTANT:\n{assistant_msg}\n\nMEMORY SUMMARY:"},
        ]
        return self.llm_chat(msgs, temperature=0.0, max_tokens=200).strip()

    def update_conversation(self, user_id: str, conv_id: str, user_msg: str, assistant_msg: str) -> Dict[str, Any]:
        um = self._get_user_mem(user_id)

        # 1) append authoritative sequential log
        rec = um.append_turn(conv_id, user_msg, assistant_msg)
        turn_id = int(rec["turn_id"])

        # 2) summarize + embed
        summary = self.summarize_turn(user_msg, assistant_msg)
        vec = self.embed(summary)

        # 3) add to user's FAISS + meta (aligned order)
        um.add_vector(vec)

        item = {
            "memory_id": str(uuid.uuid4()),
            "user_id": user_id,
            "conv_id": conv_id,
            "turn_id": turn_id,
            "ts": rec["ts"],
            "summary": summary,
            "raw_user": user_msg,
            "raw_assistant": assistant_msg,
        }
        um.append_meta(item)
        return item

    # ---------- retrieval ----------
    def retrieve_candidates(
        self,
        user_id: str,
        query: str,
        top_n: int = 40,
        exclude_conv_id: Optional[str] = None,
        exclude_turn_ids: Optional[set] = None,
    ) -> List[Dict[str, Any]]:
        um = self._get_user_mem(user_id)
        q = self.embed(query)
        um.ensure_index(q.shape[0])
        cands = um.search(q, top_n)

        # strict dedup: exclude recent window turns from the same conversation
        if exclude_conv_id and exclude_turn_ids:
            out = []
            for c in cands:
                if c.get("conv_id") == exclude_conv_id and int(c.get("turn_id", -1)) in exclude_turn_ids:
                    continue
                out.append(c)
            return out
        return cands

    def gate_with_llm(self, query: str, candidates: List[Dict[str, Any]], keep_max: int = 30) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        if not candidates:
            return [], {"gate_raw": None, "keep_ids": [], "drop_ids": []}

        formatted = "\n".join([f"- [{c['memory_id']}] {c['summary']}" for c in candidates])
        msgs = [
            {"role": "system", "content": self.prompts.GATE_SYSTEM},
            {"role": "user", "content": f"QUERY:\n{query}\n\nCANDIDATES:\n{formatted}\n\nJSON:"},
        ]
        raw = self.llm_chat(msgs, temperature=0.0, max_tokens=450)

        keep_ids: List[str] = []
        drop_ids: List[str] = []
        reasons: Dict[str, str] = {}

        try:
            obj = json.loads(raw)
            keep_ids = (obj.get("keep_ids") or [])[:keep_max]
            drop_ids = obj.get("drop_ids") or []
            reasons = obj.get("reasons") or {}
        except Exception:
            keep_ids = [c["memory_id"] for c in candidates[: min(keep_max, len(candidates))]]
            drop_ids = [c["memory_id"] for c in candidates[min(keep_max, len(candidates)) :]]

        id2c = {c["memory_id"]: c for c in candidates}
        kept = [id2c[mid] for mid in keep_ids if mid in id2c]

        return kept, {"gate_raw": raw, "keep_ids": keep_ids, "drop_ids": drop_ids, "reasons": reasons}

    def rerank_with_llm(self, query: str, candidates: List[Dict[str, Any]], k: int = 5) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        if not candidates:
            return [], {"rerank_raw": None, "top_ids": []}

        formatted = "\n".join([f"- [{c['memory_id']}] {c['summary']}" for c in candidates])
        msgs = [
            {"role": "system", "content": self.prompts.RERANK_SYSTEM},
            {"role": "user", "content": f"QUERY:\n{query}\n\nCANDIDATES:\n{formatted}\n\nJSON:"},
        ]
        raw = self.llm_chat(msgs, temperature=0.0, max_tokens=320)

        try:
            obj = json.loads(raw)
            top_ids = (obj.get("top_ids") or [])[:k]
        except Exception:
            top_ids = [c["memory_id"] for c in candidates[:k]]

        id2c = {c["memory_id"]: c for c in candidates}
        selected = [id2c[mid] for mid in top_ids if mid in id2c]
        return selected, {"rerank_raw": raw, "top_ids": top_ids}

    def retrieve_memory_bundle(
        self,
        user_id: str,
        query: str,
        k_replay: int = 5,
        candidate_n: int = 60,
        use_gate: bool = True,
        gate_keep_max: int = 30,
        use_rerank: bool = True,
        exclude_conv_id: Optional[str] = None,
        exclude_turn_ids: Optional[set] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Returns:
          bundle = {
            "replay": [items with raw_user/raw_assistant],
            "summary_only": [items with summary],
          }
        All items in bundle are guaranteed NOT to overlap with excluded recent turns
        (for the same conversation).
        """
        cands = self.retrieve_candidates(
            user_id=user_id,
            query=query,
            top_n=candidate_n,
            exclude_conv_id=exclude_conv_id,
            exclude_turn_ids=exclude_turn_ids,
        )
        debug: Dict[str, Any] = {"vector_candidates": cands}

        gated = cands
        if use_gate:
            gated, gdbg = self.gate_with_llm(query, gated, keep_max=gate_keep_max)
            debug.update(gdbg)

        if not gated:
            return {"replay": [], "summary_only": []}, debug

        if use_rerank:
            replay, rdbg = self.rerank_with_llm(query, gated, k=k_replay)
            debug.update(rdbg)
        else:
            replay = gated[:k_replay]

        replay_ids = {x["memory_id"] for x in replay}
        summary_only = [x for x in gated if x["memory_id"] not in replay_ids]

        return {"replay": replay, "summary_only": summary_only}, debug
