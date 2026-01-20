# frontend_streamlit.py
import time
import uuid
from typing import Dict, Any, List

import requests
import streamlit as st

from config import FrontendRuntimeConfig, UIParamDefaults, UI_TEXT

RUNTIME = FrontendRuntimeConfig()
DEFAULTS = UIParamDefaults()


def _ts_str(ts: float) -> str:
    try:
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(float(ts)))
    except Exception:
        return "N/A"


def call_backend(
    messages: List[Dict[str, str]],
    user_id: str,
    conv_id: str,
    params: Dict[str, Any],
    timeout_sec: int = 180,
) -> Dict[str, Any]:
    payload = {
        "model": "chatbot",
        "messages": messages,
        "user": user_id,
        "metadata": {
            "conversation_id": conv_id,
            "recent_turns": int(params["recent_turns"]),
            "candidate_n": int(params["candidate_n"]),
            "retrieve_k": int(params["retrieve_k"]),
            "use_gate": bool(params["use_gate"]),
            "gate_keep_max": int(params["gate_keep_max"]),
            "use_rerank": bool(params["use_rerank"]),
        },
        "temperature": float(params["temperature"]),
        "max_tokens": int(params["max_tokens"]),
    }

    r = requests.post(RUNTIME.BACKEND_URL, json=payload, timeout=timeout_sec)
    if r.status_code != 200:
        raise RuntimeError(f"Backend error {r.status_code}:\n{r.text}")
    return r.json()


def init_state():
    if "lang" not in st.session_state:
        st.session_state.lang = "ZH"  # default Chinese

    # widget-bound keys
    if "ui_user_id" not in st.session_state:
        st.session_state.ui_user_id = "user-" + uuid.uuid4().hex[:8]
    if "ui_conv_id" not in st.session_state:
        st.session_state.ui_conv_id = "conv-" + uuid.uuid4().hex[:8]

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "system", "content": "You are a helpful assistant."}]

    if "used_history" not in st.session_state:
        # per assistant turn: {"replay":[...], "summary_only":[...]}
        st.session_state.used_history = []


def new_chat():
    st.session_state.ui_conv_id = "conv-" + uuid.uuid4().hex[:8]
    st.session_state.messages = [{"role": "system", "content": "You are a helpful assistant."}]
    st.session_state.used_history = []


def clear_ui_only():
    st.session_state.messages = [{"role": "system", "content": "You are a helpful assistant."}]
    st.session_state.used_history = []


def render_used_history_block(t: Dict[str, Any], used: Dict[str, Any]):
    replay = used.get("replay", []) or []
    summary_only = used.get("summary_only", []) or []

    st.markdown(f"#### {t['main']['replay_section']}")
    if not replay:
        st.caption(t["main"]["none"])
    else:
        for m in replay:
            with st.chat_message("user"):
                st.markdown(m.get("raw_user", ""))
            with st.chat_message("assistant"):
                st.markdown(m.get("raw_assistant", ""))
            st.markdown("---")

    st.markdown(f"#### {t['main']['summary_section']}")
    if not summary_only:
        st.caption(t["main"]["none"])
    else:
        for m in summary_only:
            ts = _ts_str(m.get("ts", 0.0))
            score = m.get("score", None)
            header = f"• {ts}"
            if score is not None:
                try:
                    header += f"  (score={float(score):.4f})"
                except Exception:
                    pass
            st.markdown(header)
            st.markdown(m.get("summary", ""))
            st.markdown("---")


# ---------------- UI ----------------
st.set_page_config(page_title="Memory Chatbot", page_icon="💬", layout="wide")
init_state()

t = UI_TEXT.get(st.session_state.lang, UI_TEXT["ZH"])

st.markdown(
    """
    <style>
      .block-container { padding-top: 1.0rem; max-width: 1200px; }
      .stCaption { font-size: 0.88rem; opacity: 0.85; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title(t["app_title"])

with st.sidebar:
    st.markdown(f"## {t['sidebar']['language']}")
    st.session_state.lang = st.selectbox(
        t["sidebar"]["language"],
        options=list(UI_TEXT.keys()),
        index=list(UI_TEXT.keys()).index(st.session_state.lang) if st.session_state.lang in UI_TEXT else 0,
        label_visibility="collapsed",
        key="ui_lang_select",
    )
    t = UI_TEXT.get(st.session_state.lang, UI_TEXT["ZH"])

    st.divider()
    st.markdown(f"## {t['sidebar']['identity_title']}")

    st.caption(t["sidebar"]["user_id_help"])
    st.text_input(t["sidebar"]["user_id"], key="ui_user_id")

    st.caption(t["sidebar"]["conv_id_help"])
    st.text_input(t["sidebar"]["conv_id"], key="ui_conv_id")

    cols = st.columns(2)
    with cols[0]:
        if st.button(t["sidebar"]["new_chat"], use_container_width=True, key="btn_new_chat"):
            new_chat()
            st.rerun()
    with cols[1]:
        if st.button(
            t["sidebar"]["clear_ui"],
            use_container_width=True,
            help=t["sidebar"]["clear_ui_help"],
            key="btn_clear_ui",
        ):
            clear_ui_only()
            st.rerun()

    st.divider()
    st.markdown(f"## {t['sidebar']['context_title']}")

    st.caption(t["sidebar"]["recent_turns_help"])
    recent_turns = st.slider(t["sidebar"]["recent_turns"], 0, 30, DEFAULTS.recent_turns, 1, key="ui_recent_turns")

    st.caption(t["sidebar"]["candidate_n_help"])
    candidate_n = st.slider(t["sidebar"]["candidate_n"], 5, 200, DEFAULTS.candidate_n, 5, key="ui_candidate_n")

    st.caption(t["sidebar"]["retrieve_k_help"])
    retrieve_k = st.slider(t["sidebar"]["retrieve_k"], 0, 20, DEFAULTS.retrieve_k, 1, key="ui_retrieve_k")

    st.caption(t["sidebar"]["use_gate_help"])
    use_gate = st.toggle(t["sidebar"]["use_gate"], value=DEFAULTS.use_gate, key="ui_use_gate")

    st.caption(t["sidebar"]["gate_keep_max_help"])
    gate_keep_max = st.slider(t["sidebar"]["gate_keep_max"], 0, 80, DEFAULTS.gate_keep_max, 1, key="ui_gate_keep_max")

    st.caption(t["sidebar"]["use_rerank_help"])
    use_rerank = st.toggle(t["sidebar"]["use_rerank"], value=DEFAULTS.use_rerank, key="ui_use_rerank")

    st.divider()
    st.markdown(f"## {t['sidebar']['gen_title']}")

    st.caption(t["sidebar"]["temperature_help"])
    temperature = st.slider(t["sidebar"]["temperature"], 0.0, 1.5, float(DEFAULTS.temperature), 0.05, key="ui_temp")

    st.caption(t["sidebar"]["max_tokens_help"])
    max_tokens = st.slider(t["sidebar"]["max_tokens"], 64, 4096, DEFAULTS.max_tokens, 64, key="ui_max_tokens")

params = dict(
    recent_turns=recent_turns,
    candidate_n=candidate_n,
    retrieve_k=retrieve_k,
    use_gate=use_gate,
    gate_keep_max=gate_keep_max,
    use_rerank=use_rerank,
    temperature=temperature,
    max_tokens=max_tokens,
)

user_id = st.session_state.ui_user_id
conv_id = st.session_state.ui_conv_id

# render chat + used history
assistant_turn_seen = 0
for msg in st.session_state.messages:
    if msg["role"] == "system":
        continue
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

    if msg["role"] == "assistant":
        if assistant_turn_seen < len(st.session_state.used_history):
            used = st.session_state.used_history[assistant_turn_seen]
            with st.expander(t["main"]["history_used"], expanded=False):
                render_used_history_block(t, used)
        assistant_turn_seen += 1

# input
user_text = st.chat_input(t["main"]["input_placeholder"])
if user_text:
    st.session_state.messages.append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.markdown(user_text)

    with st.chat_message("assistant"):
        try:
            with st.spinner(t["main"]["thinking"]):
                resp = call_backend(
                    messages=st.session_state.messages,
                    user_id=user_id,
                    conv_id=conv_id,
                    params=params,
                    timeout_sec=180,
                )

            assistant_reply = resp["choices"][0]["message"]["content"]
            st.markdown(assistant_reply)

            retrieved = resp.get("retrieved", {}) or {}
            used = {
                "replay": retrieved.get("replay", []) or [],
                "summary_only": retrieved.get("summary_only", []) or [],
            }

            st.session_state.messages.append({"role": "assistant", "content": assistant_reply})
            st.session_state.used_history.append(used)

            with st.expander(t["main"]["history_used"], expanded=True):
                render_used_history_block(t, used)

        except Exception as e:
            st.error(str(e))
            st.caption(t["main"]["backend_error_tip"])
