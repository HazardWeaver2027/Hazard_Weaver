"""Context window budgeting for headline / vLLM runs (P0 stability)."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence


def _chars_per_token() -> float:
    raw = os.environ.get("HWA_TOKEN_CHARS_PER_TOKEN")
    if raw is None or str(raw).strip() == "":
        return 3.0
    return max(1.5, float(raw))


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return int(default)
    return int(raw)


def _bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class ContextBudget:
    max_model_len: int = 8192
    max_input_tokens: int = 6000
    max_output_tokens: int = 768
    max_obs_chars: int = 3000
    trim_messages: bool = True
    context_safety_tokens: int = 4096
    token_chars_per_token: float = 3.0

    @property
    def reserve_output_tokens(self) -> int:
        return max(256, int(self.max_output_tokens))

    def as_dict(self) -> Dict[str, Any]:
        return {
            "max_model_len": self.max_model_len,
            "max_input_tokens": self.max_input_tokens,
            "max_output_tokens": self.max_output_tokens,
            "max_obs_chars": self.max_obs_chars,
            "trim_messages": self.trim_messages,
            "context_safety_tokens": self.context_safety_tokens,
            "token_chars_per_token": self.token_chars_per_token,
        }


def context_budget_from_env() -> ContextBudget:
    max_model_len = _int_env("HWA_MAX_MODEL_LEN", 8192)
    max_output = _int_env("HWA_MAX_OUTPUT_TOKENS", 768)
    default_input = max(1024, max_model_len - max_output - 256)
    return ContextBudget(
        max_model_len=max_model_len,
        max_input_tokens=_int_env("HWA_MAX_INPUT_TOKENS", default_input),
        max_output_tokens=max_output,
        max_obs_chars=_int_env("HWA_MAX_OBS_CHARS", 3000),
        trim_messages=_bool_env("HWA_TRIM_MESSAGES", True),
    )


def context_budget_from_arm_cfg(arm_cfg: Mapping[str, Any]) -> ContextBudget:
    raw = dict(arm_cfg.get("context_budget") or {})
    max_model_len = int(raw.get("max_model_len") or arm_cfg.get("max_model_len") or 8192)
    max_output = int(raw.get("max_output_tokens") or 768)
    default_input = max(1024, max_model_len - max_output - 256)
    default_safety = 6144 if max_model_len >= 32768 else 2048
    dense_default = 2.0 if max_model_len >= 32768 else 3.0
    return ContextBudget(
        max_model_len=max_model_len,
        max_input_tokens=int(raw.get("max_input_tokens") or default_input),
        max_output_tokens=max_output,
        max_obs_chars=int(raw.get("max_obs_chars") or 3000),
        trim_messages=bool(raw.get("trim_messages", True)),
        context_safety_tokens=int(raw.get("context_safety_tokens") or default_safety),
        token_chars_per_token=float(raw.get("token_chars_per_token") or dense_default),
    )


def apply_context_budget_env(budget: ContextBudget) -> None:
    os.environ["HWA_MAX_MODEL_LEN"] = str(budget.max_model_len)
    os.environ["HWA_MAX_INPUT_TOKENS"] = str(budget.max_input_tokens)
    os.environ["HWA_MAX_OUTPUT_TOKENS"] = str(budget.max_output_tokens)
    os.environ["HWA_MAX_OBS_CHARS"] = str(budget.max_obs_chars)
    os.environ["HWA_TRIM_MESSAGES"] = "1" if budget.trim_messages else "0"
    os.environ["HWA_CONTEXT_SAFETY_TOKENS"] = str(budget.context_safety_tokens)
    os.environ["HWA_TOKEN_CHARS_PER_TOKEN"] = str(budget.token_chars_per_token)


def effective_message_token_budget(
    budget: ContextBudget,
    *,
    tools: Optional[Sequence[Mapping[str, Any]]] = None,
) -> int:
    """Message token budget before max_output_tokens; reserves tools + safety slack."""
    safety = _int_env("HWA_CONTEXT_SAFETY_TOKENS", int(budget.context_safety_tokens))
    tools_tok = estimate_tools_tokens(tools)
    hard_cap = (
        int(budget.max_model_len)
        - int(budget.reserve_output_tokens)
        - int(safety)
        - int(tools_tok)
    )
    return max(512, min(int(budget.max_input_tokens), hard_cap))


def agent_limits_for_headline(*, arm_id: Optional[str] = None):
    from hazardweaver.hwa.agent_runtime.loop import AgentLimits

    if arm_id:
        from hazardweaver.hwa.llm.iclr_model_arms import get_arm

        budget = context_budget_from_arm_cfg(get_arm(arm_id))
        apply_context_budget_env(budget)
    else:
        budget = context_budget_from_env()
    return AgentLimits(
        max_steps=_int_env("HWA_AGENT_MAX_STEPS", 15),
        max_wall_s=float(os.environ.get("HWA_AGENT_MAX_WALL_S", "300")),
        max_obs_chars=budget.max_obs_chars,
        temperature=0.2,
        max_tokens=budget.max_output_tokens,
    )


def estimate_tokens(text: str) -> int:
    """Conservative chars→tokens heuristic (tool JSON tends to tokenize dense)."""
    if not text:
        return 0
    cpt = _chars_per_token()
    return max(1, int((len(text) + cpt - 1) // cpt))


def estimate_tools_tokens(tools: Optional[Sequence[Mapping[str, Any]]]) -> int:
    if not tools:
        return 0
    return estimate_tokens(json.dumps(list(tools), ensure_ascii=False))


def estimate_message_tokens(messages: Sequence[Mapping[str, Any]]) -> int:
    total = 0
    for msg in messages:
        total += 4  # role/overhead
        content = msg.get("content")
        if content is not None:
            total += estimate_tokens(str(content))
        tool_calls = msg.get("tool_calls")
        if tool_calls:
            total += estimate_tokens(json.dumps(tool_calls, ensure_ascii=False))
        for key in ("name", "tool_call_id"):
            if msg.get(key):
                total += estimate_tokens(str(msg[key]))
    return total


def trim_messages_for_budget(
    messages: Sequence[Mapping[str, Any]],
    *,
    max_input_tokens: int,
    reserve_output_tokens: int = 768,
) -> List[Dict[str, Any]]:
    """Keep system + first user task; drop oldest middle turns until within budget."""
    if not messages:
        return []
    budget = max(512, int(max_input_tokens) - int(reserve_output_tokens))
    serialized: List[Dict[str, Any]] = [dict(m) for m in messages]

    pinned: List[Dict[str, Any]] = []
    for msg in reversed(serialized):
        if str(msg.get("role")) != "tool":
            continue
        name = str(msg.get("name") or "")
        if name == "controller_commit_route":
            pinned.insert(0, dict(msg))
            break

    if estimate_message_tokens(serialized) <= budget:
        return serialized

    if len(serialized) <= 2:
        return _truncate_message_contents(serialized, budget)

    head = [dict(serialized[0])]
    if len(serialized) > 1 and str(serialized[1].get("role")) == "user":
        head.append(dict(serialized[1]))
        tail_start = 2
    else:
        tail_start = 1

    tail = [dict(m) for m in serialized[tail_start:]]
    if pinned:
        tail = [
            m
            for m in tail
            if not (
                str(m.get("role")) == "tool"
                and str(m.get("name") or "") == "controller_commit_route"
            )
        ]
    while tail and estimate_message_tokens(head + pinned + tail) > budget:
        if pinned and tail and tail[0] in pinned:
            break
        tail.pop(0)

    merged = head + pinned + tail
    if estimate_message_tokens(merged) > budget:
        merged = _truncate_message_contents(merged, budget)

    if tail and head and estimate_message_tokens(merged) <= budget:
        if pinned:
            return merged
        with_note = head + [{"role": "system", "content": "[Earlier tool turns truncated for context budget.]"}] + tail
        if estimate_message_tokens(with_note) <= budget:
            return with_note
    return merged


def _truncate_message_contents(
    messages: List[Dict[str, Any]],
    budget_tokens: int,
) -> List[Dict[str, Any]]:
    """Last resort: truncate largest tool/user contents."""
    out = [dict(m) for m in messages]
    while estimate_message_tokens(out) > budget_tokens:
        idx = max(range(len(out)), key=lambda i: estimate_tokens(str(out[i].get("content") or "")))
        content = str(out[idx].get("content") or "")
        if len(content) <= 200:
            break
        out[idx]["content"] = content[: max(120, len(content) // 2)] + "\n…[truncated for context budget]"
    return out


def maybe_trim_messages(
    messages: Sequence[Mapping[str, Any]],
    *,
    budget: Optional[ContextBudget] = None,
    tools: Optional[Sequence[Mapping[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    b = budget or context_budget_from_env()
    if not b.trim_messages:
        return [dict(m) for m in messages]
    return trim_messages_for_budget(
        messages,
        max_input_tokens=effective_message_token_budget(b, tools=tools),
        reserve_output_tokens=0,
    )


def _payload_token_estimate(
    messages: Sequence[Mapping[str, Any]],
    tools: Optional[Sequence[Mapping[str, Any]]],
    *,
    chars_per_token: float,
) -> int:
    prev = os.environ.get("HWA_TOKEN_CHARS_PER_TOKEN")
    os.environ["HWA_TOKEN_CHARS_PER_TOKEN"] = str(chars_per_token)
    try:
        return estimate_message_tokens(messages) + estimate_tools_tokens(tools)
    finally:
        if prev is None:
            os.environ.pop("HWA_TOKEN_CHARS_PER_TOKEN", None)
        else:
            os.environ["HWA_TOKEN_CHARS_PER_TOKEN"] = prev


def _vllm_context_cap(budget: ContextBudget, tools: Optional[Sequence[Mapping[str, Any]]]) -> int:
    return max(
        512,
        int(budget.max_model_len)
        - int(budget.reserve_output_tokens)
        - int(budget.context_safety_tokens)
        - estimate_tools_tokens(tools),
    )


def fit_chat_payload_to_budget(
    messages: Sequence[Mapping[str, Any]],
    tools: Optional[Sequence[Mapping[str, Any]]],
    *,
    budget: Optional[ContextBudget] = None,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Trim chat messages so messages + OpenAI tools fit under the model context cap."""
    b = budget or context_budget_from_env()
    tools_out = [dict(t) for t in (tools or [])]
    if not b.trim_messages:
        return [dict(m) for m in messages], tools_out
    trimmed = maybe_trim_messages(messages, budget=b, tools=tools_out)
    cap = _vllm_context_cap(b, tools_out)
    dense_cpt = max(1.5, float(b.token_chars_per_token))
    while _payload_token_estimate(trimmed, tools_out, chars_per_token=dense_cpt) > cap:
        prev_len = len(trimmed)
        trimmed = trim_messages_for_budget(
            trimmed,
            max_input_tokens=max(512, int(cap * 0.85)),
            reserve_output_tokens=0,
        )
        if len(trimmed) >= prev_len and prev_len <= 2:
            trimmed = _truncate_message_contents(trimmed, max(256, int(cap * 0.75)))
            break
    return trimmed, tools_out
