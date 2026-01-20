# chatbot_api.py
import time
import uuid
import json
import traceback
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from config import APIConfig, StoreConfig, PromptConfig
from memory import MemoryStore

app = FastAPI()

api_cfg = APIConfig()
store_cfg = StoreConfig()
prompt_cfg = PromptConfig()

mem = MemoryStore(api=api_cfg, store=store_cfg, prompts=prompt_cfg)


@app.exception_handler(Exception)
async def all_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"error": str(exc), "traceback": traceback.format_exc()},
    )


# ---------- OpenAI-like request models ----------
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionsRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.3
    max_tokens: Optional[int] = 768
    top_p: Optional[float] = 1.0
    stream: Optional[bool] = False
    user: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


def _extract_last_user_text(messages: List[ChatMessage]) -> str:
    for m in reversed(messages):
        if m.role == "user":
            return m.content
    return messages[-1].content if messages else ""


def _memories_to_multiturn(replay: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    Replay memories as synthetic multi-turn conversation (original user/assistant).
    """
    msgs: List[Dict[str, str]] = []
    if not replay:
        return msgs

    msgs.append({
        "role": "system",
        "content": "### Retrieved Relevant Past Conversations (Replay)\nBelow are past user/assistant exchanges that may help.",
    })
    for m in replay:
        msgs.append({"role": "user", "content": m.get("raw_user", "")})
        msgs.append({"role": "assistant", "content": m.get("raw_assistant", "")})
    return msgs


def _summaries_to_system_block(summary_only: List[Dict[str, Any]]) -> Optional[str]:
    if not summary_only:
        return None
    lines = []
    for x in summary_only:
        ts = time.strftime("%Y-%m-%d", time.localtime(float(x.get("ts", 0.0))))
        lines.append(f"- ({ts}) {x.get('summary','')}")
    return "### Additional Relevant Memory (Summaries)\n" + "\n".join(lines)


def _build_prompt(
    user_query: str,
    recent_msgs: List[Dict[str, str]],
    replay: List[Dict[str, Any]],
    summary_only: List[Dict[str, Any]],
) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = [{"role": "system", "content": prompt_cfg.CHAT_SYSTEM}]

    sys_block = _summaries_to_system_block(summary_only)
    if sys_block:
        out.append({"role": "system", "content": sys_block})

    out.extend(_memories_to_multiturn(replay))
    out.extend(recent_msgs)
    out.append({"role": "user", "content": user_query})
    return out


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/health/embed")
def health_embed():
    v = mem.embed("hello")
    return {"ok": True, "dim": int(v.shape[0])}


@app.get("/health/llm")
def health_llm():
    out = mem.llm_chat([{"role": "user", "content": "Say OK"}], temperature=0.0, max_tokens=8)
    return {"ok": True, "sample": out}


@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionsRequest):
    created = int(time.time())
    response_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"

    user_id = req.user or "anonymous"
    md = req.metadata or {}

    conv_id = md.get("conversation_id") or md.get("conv_id") or user_id

    # knobs
    recent_turns = int(md.get("recent_turns", 6))
    candidate_n = int(md.get("candidate_n", 60))
    retrieve_k = int(md.get("retrieve_k", 5))

    use_gate = bool(md.get("use_gate", True))
    gate_keep_max = int(md.get("gate_keep_max", 30))
    use_rerank = bool(md.get("use_rerank", True))

    user_query = _extract_last_user_text(req.messages)

    # recent window from sequential log + dedup ids
    recent_msgs, recent_turn_ids = mem.load_recent(user_id, conv_id, recent_turns)
    recent_turn_id_set = set(recent_turn_ids)

    # retrieve memory bundle excluding anything already in recent window
    bundle, _debug = mem.retrieve_memory_bundle(
        user_id=user_id,
        query=user_query,
        k_replay=retrieve_k,
        candidate_n=candidate_n,
        use_gate=use_gate,
        gate_keep_max=gate_keep_max,
        use_rerank=use_rerank,
        exclude_conv_id=conv_id,
        exclude_turn_ids=recent_turn_id_set,
    )
    replay = bundle["replay"]
    summary_only = bundle["summary_only"]

    final_messages = _build_prompt(
        user_query=user_query,
        recent_msgs=recent_msgs,
        replay=replay,
        summary_only=summary_only,
    )

    assistant_reply = mem.llm_chat(
        final_messages,
        temperature=req.temperature or 0.3,
        max_tokens=req.max_tokens or 768,
    )

    # write: append sequential log + store summary embedding
    mem.update_conversation(
        user_id=user_id,
        conv_id=conv_id,
        user_msg=user_query,
        assistant_msg=assistant_reply,
    )

    return {
        "id": response_id,
        "object": "chat.completion",
        "created": created,
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": assistant_reply},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None},
        # frontend display payload
        "retrieved": {
            "replay": [
                {
                    "memory_id": x.get("memory_id"),
                    "ts": x.get("ts"),
                    "score": x.get("score"),
                    "raw_user": x.get("raw_user"),
                    "raw_assistant": x.get("raw_assistant"),
                    "conv_id": x.get("conv_id"),
                    "turn_id": x.get("turn_id"),
                }
                for x in replay
            ],
            "summary_only": [
                {
                    "memory_id": x.get("memory_id"),
                    "ts": x.get("ts"),
                    "score": x.get("score"),
                    "summary": x.get("summary"),
                    "conv_id": x.get("conv_id"),
                    "turn_id": x.get("turn_id"),
                }
                for x in summary_only
            ],
        },
        # Optional: you can return debug when needed
        # "debug": _debug,
    }
