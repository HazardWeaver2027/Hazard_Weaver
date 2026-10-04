"""Normalize OpenAI-style tool_call ids for vLLM backends with strict parsers (e.g. Mistral)."""

from __future__ import annotations

import hashlib
import re
import uuid
from typing import Any, Dict, List, Mapping, MutableMapping, Sequence, Union

_TOOL_CALL_ID_RE = re.compile(r"^[A-Za-z0-9]{9}$")


def is_valid_tool_call_id(value: str) -> bool:
    return bool(_TOOL_CALL_ID_RE.match(str(value or "").strip()))


def normalize_tool_call_id(raw: str | None, *, fallback_index: int = 0) -> str:
    """Map any id to a stable 9-char alphanumeric string (Mistral chat template requirement)."""
    s = str(raw or "").strip()
    if is_valid_tool_call_id(s):
        return s
    seed = s or f"fallback{fallback_index}"
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()[:9]


def new_tool_call_id() -> str:
    return hashlib.sha256(uuid.uuid4().bytes).hexdigest()[:9]


def sanitize_vllm_chat_messages(
    messages: Sequence[Union[Mapping[str, Any], MutableMapping[str, Any]]],
) -> List[Dict[str, Any]]:
    """Return a copy of chat messages with tool_call ids normalized for strict vLLM parsers."""
    out: List[Dict[str, Any]] = []
    fallback = 0
    for raw in messages:
        msg = dict(raw)
        role = str(msg.get("role") or "user")
        if role == "tool":
            msg["tool_call_id"] = normalize_tool_call_id(
                str(msg.get("tool_call_id") or ""),
                fallback_index=fallback,
            )
            fallback += 1
            out.append(msg)
            continue
        tool_calls = msg.get("tool_calls")
        if role == "assistant" and isinstance(tool_calls, list):
            norm_calls: List[Dict[str, Any]] = []
            for i, tc in enumerate(tool_calls):
                if not isinstance(tc, Mapping):
                    continue
                row = dict(tc)
                row["id"] = normalize_tool_call_id(
                    str(row.get("id") or ""),
                    fallback_index=fallback + i,
                )
                norm_calls.append(row)
            msg["tool_calls"] = norm_calls
            fallback += len(norm_calls)
        out.append(msg)
    return out
