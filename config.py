# config.py
import os
from dataclasses import dataclass


# =========================================================
# Backend: API settings (OpenAI package -> your vLLM endpoints)
# =========================================================
@dataclass
class APIConfig:
    """
    Use OpenAI python package but point base_url to your local vLLM servers.

    Environment variables:
      - OPENAI_API_KEY: can be dummy for vLLM, but required by OpenAI client
      - LLM_BASE_URL: e.g. http://localhost:8000/v1
      - EMB_BASE_URL: e.g. http://localhost:8001/v1
      - LLM_MODEL: model name your vLLM exposes
      - EMB_MODEL: embedding model name your embedding server exposes
    """
    API_KEY: str = os.getenv("OPENAI_API_KEY", "EMPTY")

    LLM_BASE_URL: str = os.getenv("LLM_BASE_URL", "http://localhost:8000/v1")
    EMB_BASE_URL: str = os.getenv("EMB_BASE_URL", "http://localhost:8002/v1")

    LLM_MODEL: str = os.getenv("LLM_MODEL", "Qwen/Qwen3-4B-Instruct-2507-FP8")
    EMB_MODEL: str = os.getenv("EMB_MODEL", "Qwen/Qwen3-Embedding-0.6B")


# =========================================================
# Backend: storage settings (per-user isolated memory)
# =========================================================
@dataclass
class StoreConfig:
    """
    Storage layout:
      STORAGE_DIR/
        users/
          <user_id_sanitized>/
            faiss.index
            meta.jsonl
            state.json
            conversations/
              <conv_id>.jsonl

    Environment variables:
      - STORAGE_DIR: default ./storage
      - EMBEDDING_DIMS: 0 means infer from first embedding vector
    """
    STORAGE_DIR: str = os.getenv("STORAGE_DIR", "./storage")
    USERS_DIR: str = os.path.join(STORAGE_DIR, "users")
    EMBEDDING_DIMS: int = int(os.getenv("EMBEDDING_DIMS", "0"))


# =========================================================
# Backend: prompt templates
# =========================================================
@dataclass
class PromptConfig:
    """
    You can tune these prompts freely.
    Keep outputs JSON-only for gate/rerank to make parsing robust.
    """

    # Main assistant system prompt
    CHAT_SYSTEM: str = (
        "You are a helpful assistant.\n"
        "You may receive retrieved memory in two forms:\n"
        "1) Replay conversations: full multi-turn user/assistant exchanges.\n"
        "2) Summary-only memory: short bullet summaries in system messages.\n"
        "Use them only if relevant and do not hallucinate facts."
    )

    # Summarize each (user, assistant) turn into a memory snippet
    SUMMARY_SYSTEM: str = (
        "You are an assistant that writes compact, factual memory summaries.\n"
        "Write 1-3 sentences capturing durable info, decisions, preferences, and key facts.\n"
        "Avoid ephemeral details unless they matter.\n"
        "Return only the summary text."
    )

    # Gate: keep or drop candidates (NOT ranking)
    GATE_SYSTEM: str = (
        "You are a strict relevance filter.\n"
        "Given a QUERY and a list of CANDIDATES (each with an id and a summary),\n"
        "decide which candidates are relevant enough to keep.\n"
        "\n"
        "Output JSON ONLY with fields:\n"
        "  keep_ids: list of kept candidate ids (ordered by your confidence, most relevant first)\n"
        "  drop_ids: list of dropped candidate ids\n"
        "  reasons: a dict mapping id -> short reason (optional)\n"
        "\n"
        "Rules:\n"
        "- Keep if it directly helps answer the query or provides necessary context.\n"
        "- Drop if irrelevant, too generic, or about a different topic.\n"
        "- Prefer precision over recall."
    )

    # Rerank: pick top-K among gated candidates for replay
    RERANK_SYSTEM: str = (
        "You are a reranker.\n"
        "Given QUERY and CANDIDATES (id + summary), select the best ones to replay as full conversations.\n"
        "\n"
        "Output JSON ONLY with field:\n"
        "  top_ids: list of candidate ids in descending usefulness.\n"
        "\n"
        "Ranking criteria:\n"
        "- Most directly answers or supports the query.\n"
        "- Contains concrete decisions, preferences, constraints, or prior reasoning.\n"
        "- Avoid duplicates and overly generic memories."
    )


# =========================================================
# Frontend: runtime config (backend URL NOT editable in GUI)
# =========================================================
@dataclass
class FrontendRuntimeConfig:
    """
    Frontend reads BACKEND_URL at startup:
      export BACKEND_URL="http://localhost:9000/v1/chat/completions"
    """
    BACKEND_URL: str = os.getenv("BACKEND_URL", "http://localhost:9000/v1/chat/completions")


# =========================================================
# Frontend: default parameter values
# =========================================================
@dataclass
class UIParamDefaults:
    recent_turns: int = 6
    candidate_n: int = 60
    retrieve_k: int = 5
    use_gate: bool = True
    gate_keep_max: int = 30
    use_rerank: bool = True
    temperature: float = 0.3
    max_tokens: int = 768


# =========================================================
# Frontend: i18n UI text (extendable)
# =========================================================
UI_TEXT = {
    "EN": {
        "app_title": "💬 Persistent Memory Chatbot",
        "sidebar": {
            "language": "Language",

            "identity_title": "Identity",
            "user_id": "User ID",
            "user_id_help": (
                "Used to isolate memory across different users. "
                "Keep it stable to reuse memory across sessions."
            ),
            "conv_id": "Conversation ID",
            "conv_id_help": (
                "Identifies a chat thread (recent window is per conversation). "
                "Click “New chat” to start a fresh thread."
            ),
            "new_chat": "New chat",
            "clear_ui": "Clear UI only",
            "clear_ui_help": "Clears the frontend display only; backend memory is not deleted.",

            "context_title": "Context & Memory",
            "recent_turns": "Recent window (turns)",
            "recent_turns_help": (
                "Last N user+assistant turns in THIS conversation, injected as normal multi-turn context."
            ),
            "candidate_n": "Vector candidates (N)",
            "candidate_n_help": (
                "How many memory summaries to retrieve from FAISS before filtering. "
                "Higher improves recall but increases gate/rerank cost."
            ),
            "retrieve_k": "Replay top-K",
            "retrieve_k_help": (
                "How many filtered memories are replayed as full multi-turn (original user+assistant)."
            ),
            "use_gate": "Enable gate",
            "use_gate_help": (
                "Binary filter: keep/drop candidate memories before reranking (NOT ranking)."
            ),
            "gate_keep_max": "Gate keep max",
            "gate_keep_max_help": (
                "Maximum candidates kept after gate. Non-replayed kept items go to summary-only system memory."
            ),
            "use_rerank": "Enable rerank",
            "use_rerank_help": (
                "Use an LLM reranker to select top-K for replay. If off, use first K gated candidates."
            ),

            "gen_title": "Generation",
            "temperature": "Temperature",
            "temperature_help": "Controls randomness. Lower is more deterministic.",
            "max_tokens": "Max tokens",
            "max_tokens_help": "Maximum output tokens for assistant reply.",
        },
        "main": {
            "input_placeholder": "Message",
            "thinking": "Thinking...",
            "history_used": "History used this turn",
            "replay_section": "Replay conversations (multi-turn)",
            "summary_section": "Additional memory (summaries only)",
            "none": "None.",
            "backend_error_tip": (
                "Tip: check backend logs, or ensure BACKEND_URL is correct. "
                "If the backend returns JSON traceback, it will be shown above."
            ),
        },
    },

    "ZH": {
        "app_title": "💬 可持久化记忆对话机器人",
        "sidebar": {
            "language": "语言",

            "identity_title": "身份设置",
            "user_id": "用户 ID",
            "user_id_help": (
                "用于隔离不同用户的记忆索引（不同用户不会共享记忆）。"
                "如果希望跨会话复用记忆，请保持该 ID 稳定。"
            ),
            "conv_id": "对话 ID",
            "conv_id_help": (
                "用于标识一个对话线程（recent window 按对话区分）。"
                "点击“新对话”会创建新的对话线程。"
            ),
            "new_chat": "新对话",
            "clear_ui": "仅清空界面",
            "clear_ui_help": "只清空前端显示，不会删除后端已写入的记忆。",

            "context_title": "上下文与记忆",
            "recent_turns": "最近窗口（轮数）",
            "recent_turns_help": (
                "当前对话线程内最近 N 轮 user+assistant，"
                "以多轮对话形式直接注入模型上下文。"
            ),
            "candidate_n": "向量候选数（N）",
            "candidate_n_help": (
                "从 FAISS 召回 N 条候选摘要，再进行 Gate/重排。"
                "N 越大召回率更高，但 Gate/重排成本也更高。"
            ),
            "retrieve_k": "回放 Top-K",
            "retrieve_k_help": (
                "从过滤后的候选中选出 Top-K，以“原始多轮对话（user+assistant）”形式回放给模型。"
            ),
            "use_gate": "启用过滤门（Gate）",
            "use_gate_help": (
                "二分类过滤：判断候选记忆是否与当前问题相关（保留/丢弃），不是排序。"
            ),
            "gate_keep_max": "Gate 保留上限",
            "gate_keep_max_help": (
                "Gate 后最多保留多少条候选。"
                "其中未被回放的部分会作为“摘要（system）记忆”注入。"
            ),
            "use_rerank": "启用重排（Rerank）",
            "use_rerank_help": (
                "使用 LLM 对 Gate 后候选重排，选择 Top-K 回放。关闭时直接取前 K 条。"
            ),

            "gen_title": "生成参数",
            "temperature": "温度（Temperature）",
            "temperature_help": "随机性控制，越低越稳定。",
            "max_tokens": "最大生成 Tokens",
            "max_tokens_help": "限制助手回复的最大 token 数。",
        },
        "main": {
            "input_placeholder": "输入消息…",
            "thinking": "思考中…",
            "history_used": "本轮使用的历史上下文",
            "replay_section": "回放对话（多轮形式）",
            "summary_section": "补充记忆（仅摘要）",
            "none": "无。",
            "backend_error_tip": (
                "提示：请检查后端日志，或确认 BACKEND_URL 配置正确。"
                "如果后端返回 JSON traceback，会在上方显示。"
            ),
        },
    },
}
