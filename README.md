# OpenPersistentMemory

OpenPersistentMemory is a lightweight **persistent-memory chatbot** stack that works with **vLLM-deployed OpenAI-compatible APIs** (LLM + Embedding).  
It adds long-term memory with **FAISS**, stores full conversations **sequentially**, and injects relevant history back into the model with a **two-level retrieval strategy**:

1) **Recent window (multi-turn):** the latest turns from the current conversation thread  
2) **Persistent memory (retrieved):**
   - **Replay conversations (multi-turn):** selected past user/assistant exchanges inserted as chat turns  
   - **Summary-only memory (system):** additional relevant summaries injected into the system prompt

It also includes a **GPT-like Streamlit frontend** with **Chinese/English UI** and a clear display of which memories were used each turn.

---

## Features

- ✅ **OpenAI package compatible**: backend uses `openai` Python client, pointed to your local vLLM endpoints via `base_url`
- ✅ **Two types of retrieval**
  - **Recent context**: last *N turns* in the current conversation
  - **Persistent memory**: FAISS retrieval + **Gate (filter)** + optional **Rerank**
- ✅ **Conversation Gate (filter, not ranker)**: drops irrelevant candidates before reranking
- ✅ **Strict de-duplication**: anything already in the recent window is excluded from replay/summary memory
- ✅ **Per-user isolation**: different `user_id` has different FAISS index and storage folder (no shared memory)
- ✅ **Sequential conversation storage**: every conversation is stored as append-only JSONL, easy to reload and rebuild recent window
- ✅ **Streamlit frontend**: GPT-like chat UI, configurable memory knobs, live display of retrieved history per turn
- ✅ **i18n-ready UI**: all frontend text & descriptions stored in `config.py` (`UI_TEXT` dict), easy to add more languages

---

## Repository Structure

```

OpenPersistentMemory/
chatbot_api.py          # FastAPI backend: OpenAI-like /v1/chat/completions
memory.py               # Memory core: per-user FAISS + sequential logs + gate/rerank
frontend_streamlit.py   # Streamlit frontend (GPT-like UI)
config.py               # All configs: endpoints/models/prompts/storage + i18n UI texts
storage/                # Created at runtime

```

### Storage Layout (created automatically)

```bash
storage/
users/
<user_id_sanitized>/
faiss.index
meta.jsonl
state.json
conversations/
<conv_id>.jsonl
```

- `conversations/<conv_id>.jsonl` stores the **full conversation sequentially**, one record per turn:

```json
  {"turn_id": 12, "ts": 1700000000.0, "user": "...", "assistant": "..."}
```

* `faiss.index` + `meta.jsonl` store **persistent memory** (summaries + raw messages) for retrieval.

---

## Requirements

Python 3.9+ recommended.

Install dependencies:

```bash
pip install -r requirements.txt
```

> On some platforms you may prefer `conda install -c pytorch faiss-cpu`.

---

## 1) Prepare your vLLM endpoints

You need two OpenAI-compatible endpoints:

* **LLM chat endpoint**: `.../v1/chat/completions`
* **Embedding endpoint**: `.../v1/embeddings`

Typical setup:

* LLM server: `http://localhost:8000/v1`
* Embedding server: `http://localhost:8001/v1`

---

## 2) Configure environment variables

The system reads configuration from environment variables (also see `config.py`).

```bash
# Required by OpenAI client (can be dummy for local vLLM)
export OPENAI_API_KEY="EMPTY"

# Your vLLM endpoints
export LLM_BASE_URL="http://localhost:8000/v1"
export EMB_BASE_URL="http://localhost:8001/v1"

# Model names exposed by your servers
export LLM_MODEL="Qwen/Qwen3-4B-Instruct"
export EMB_MODEL="Qwen/Qwen3-Embedding-8B"

# Storage
export STORAGE_DIR="./storage"
export EMBEDDING_DIMS=0   # 0 = infer from first embedding

# Frontend -> backend
export BACKEND_URL="http://localhost:9000/v1/chat/completions"
```

---

## 3) Start the backend (FastAPI)

```bash
uvicorn chatbot_api:app --host 0.0.0.0 --port 9000
```

### Health checks

```bash
curl http://localhost:9000/health
curl http://localhost:9000/health/embed
curl http://localhost:9000/health/llm
```

If `health/embed` fails → embedding endpoint config is wrong
If `health/llm` fails → LLM endpoint config is wrong

---

## 4) Start the frontend (Streamlit)

```bash
streamlit run frontend_streamlit.py
```

The frontend reads `BACKEND_URL` from the environment; it is **not editable** in the UI.

---

## How memory works

For each user message:

1. **Recent window**: load last `recent_turns` turns from `conversations/<conv_id>.jsonl`
2. **Retrieve candidates**: FAISS retrieves `candidate_n` memory summaries
3. **De-dup**: remove memories that correspond to turns already in the recent window
4. **Gate (filter)**: LLM decides keep/drop candidates (binary filter, not ranking)
5. **Rerank (optional)**: LLM selects top `retrieve_k` memories to replay
6. Build prompt to LLM:

   * System prompt
   * **Summary-only memory** (system block)
   * **Replay conversations** (multi-turn)
   * **Recent window** (multi-turn)
   * Current user message
7. After answering, the system:

   * Appends the turn to the sequential conversation log (JSONL)
   * Summarizes the turn and stores it (summary embedding + meta)

---

## Frontend knobs (what they mean)

These are controlled in the Streamlit sidebar, with Chinese/English descriptions sourced from `config.py`.

* **Recent window (turns)**: how many latest turns of this conversation are included directly
* **Vector candidates (N)**: FAISS top-N candidates to consider before filtering
* **Replay top-K**: how many past turns are replayed as full multi-turn context
* **Gate keep max**: max number of candidates kept after gate
* **Enable gate**: on/off for the relevance filter (binary keep/drop)
* **Enable rerank**: on/off for LLM reranking (select best K to replay)
* **Temperature / Max tokens**: generation controls

---

## Multi-user & multi-conversation

* Different **`user_id`** → different memory index and storage folder → no shared memories
* Different **`conv_id`** under the same user → different conversation thread for recent window

In the frontend:

* `User ID` controls which user memory index you use
* `Conversation ID` controls the thread (recent window) and log file
* “New chat” generates a new `conv_id`

---

## Troubleshooting

### 1) `500 Internal Server Error`

Check backend console logs. The backend also returns JSON traceback in error responses.

### 2) Embedding dim mismatch

If you switch embedding models, old FAISS indices may become incompatible.

Fix by deleting the user’s index files:

```bash
rm -rf storage/users/<user_id_sanitized>/
```

(Or just delete `faiss.index` + `meta.jsonl` + `state.json` for that user.)

### 3) Streamlit session_state errors

This repo uses widget-bound keys (`ui_user_id`, `ui_conv_id`) to avoid Streamlit key mutation errors.

---

## Customization

### Prompts

Edit `PromptConfig` in `config.py`:

* `CHAT_SYSTEM`
* `SUMMARY_SYSTEM`
* `GATE_SYSTEM`
* `RERANK_SYSTEM`

### UI language

All UI strings live in `UI_TEXT` in `config.py`. Add a new language by adding a new top-level key (e.g. `"JP"`) with the same structure.

---

## License

