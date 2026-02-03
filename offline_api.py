import time
import uuid
from typing import List, Dict, Any, Optional

from config import APIConfig, StoreConfig, PromptConfig
from memory import MemoryStore

class OfflineAPI:
    def __init__(self, api_cfg: Optional[APIConfig] = None, store_cfg: Optional[StoreConfig] = None, prompt_cfg: Optional[PromptConfig] = None):
        self.api_cfg = api_cfg or APIConfig()
        self.store_cfg = store_cfg or StoreConfig()
        self.prompt_cfg = prompt_cfg or PromptConfig()
        self.mem = MemoryStore(api=self.api_cfg, store=self.store_cfg, prompts=self.prompt_cfg)

    def _memories_to_multiturn(self, replay: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        msgs = []
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

    def _summaries_to_system_block(self, summary_only: List[Dict[str, Any]]) -> Optional[str]:
        if not summary_only:
            return None
        lines = []
        for x in summary_only:
            ts = time.strftime("%Y-%m-%d", time.localtime(float(x.get("ts", 0.0))))
            lines.append(f"- ({ts}) {x.get('summary', '')}")
        return "### Additional Relevant Memory (Summaries)\n" + "\n".join(lines)

    def _build_prompt(
        self,
        user_query: str,
        recent_msgs: List[Dict[str, str]],
        replay: List[Dict[str, Any]],
        summary_only: List[Dict[str, Any]],
    ) -> List[Dict[str, str]]:
        out = [{"role": "system", "content": self.prompt_cfg.CHAT_SYSTEM}]

        sys_block = self._summaries_to_system_block(summary_only)
        if sys_block:
            out.append({"role": "system", "content": sys_block})

        out.extend(self._memories_to_multiturn(replay))
        out.extend(recent_msgs)
        out.append({"role": "user", "content": user_query})
        return out

    def add_history(self, user_id: str, conv_id: str, history_messages: List[Dict[str, str]]) -> List[str]:
        """
        Add history conversations to memory.
        Expects a list of message dicts like [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
        Processes them in sequential pairs.
        Returns a list of memory_ids for the added conversation turns.
        """
        added_ids = []
        i = 0
        while i < len(history_messages) - 1:
            msg1 = history_messages[i]
            msg2 = history_messages[i+1]
            
            if msg1['role'] == 'user' and msg2['role'] == 'assistant':
                res = self.mem.update_conversation(
                    user_id=user_id,
                    conv_id=conv_id,
                    user_msg=msg1['content'],
                    assistant_msg=msg2['content']
                )
                added_ids.append(res.get("memory_id"))
                i += 2
            else:
                # If the current is user but next isn't assistant, we just skip the user message
                # If current isn't user, we look for next user.
                if msg1['role'] != 'user':
                    i += 1
                elif msg2['role'] != 'assistant':
                    i += 1
        return added_ids
    
    def chat(
        self,
        user_id: str,
        conv_id: str,
        query: str,
        recent_turns: int = 6,
        candidate_n: int = 60,
        retrieve_k: int = 5,
        use_gate: bool = True,
        gate_keep_max: int = 30,
        use_rerank: bool = True,
        temperature: float = 0.3,
        max_tokens: int = 768,
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Chat with memory, returning the same structure as the chatbot_api.
        """
        created = int(time.time())
        response_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        
        # 1. load recent
        recent_msgs, recent_turn_ids = self.mem.load_recent(user_id, conv_id, recent_turns)
        recent_turn_id_set = set(recent_turn_ids)

        # 2. retrieve memory bundle
        bundle, _debug = self.mem.retrieve_memory_bundle(
            user_id=user_id,
            query=query,
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

        # 3. build prompt
        final_messages = self._build_prompt(
            user_query=query,
            recent_msgs=recent_msgs,
            replay=replay,
            summary_only=summary_only,
        )

        # 4. llm chat
        assistant_reply = self.mem.llm_chat(
            final_messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # 5. update conversation
        self.mem.update_conversation(
            user_id=user_id,
            conv_id=conv_id,
            user_msg=query,
            assistant_msg=assistant_reply,
        )

        return {
            "id": response_id,
            "object": "chat.completion",
            "created": created,
            "model": model or self.api_cfg.LLM_MODEL,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": assistant_reply},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None},
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
        }
