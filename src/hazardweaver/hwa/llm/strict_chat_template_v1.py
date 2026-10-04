"""Sanitize OpenAI chat messages for vLLM backends with strict role alternation (Mistral)."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Mapping


def needs_strict_role_alternation(*, model_id: str, profile: str = "") -> bool:
    blob = f"{model_id} {profile} {os.environ.get('HW_LLM_PROFILE', '')}".lower()
    return "mixtral" in blob or "mistral" in blob


def sanitize_strict_chat_template_messages(messages: List[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Collapse system blocks; map tool/error to user; merge consecutive roles."""
    system_chunks: List[str] = []
    body: List[Dict[str, Any]] = []
    for raw in messages:
        role = str(raw.get("role") or "user")
        if role == "system":
            text = str(raw.get("content") or "").strip()
            if text:
                system_chunks.append(text)
            continue
        if role == "tool":
            name = str(raw.get("name") or "tool")
            content = str(raw.get("content") or "")
            body.append({"role": "user", "content": f"[tool_result {name}]\n{content}"})
            continue
        if role in {"parse_error", "error"}:
            text = str(raw.get("error") or raw.get("content") or raw)
            body.append({"role": "user", "content": f"[parse_error]\n{text}"})
            continue
        if role not in {"user", "assistant"}:
            body.append({"role": "user", "content": json.dumps(dict(raw), default=str)})
            continue
        row: Dict[str, Any] = {"role": role, "content": raw.get("content")}
        if row["content"] is None:
            row["content"] = ""
        if raw.get("tool_calls"):
            row["tool_calls"] = list(raw.get("tool_calls") or [])
        body.append(row)

    out: List[Dict[str, Any]] = []
    if system_chunks:
        out.append({"role": "system", "content": "\n\n".join(system_chunks)})

    for row in body:
        role = str(row["role"])
        content = str(row.get("content") or "")
        if out and out[-1]["role"] == role:
            prev = out[-1]
            prev["content"] = f"{prev.get('content', '')}\n{content}".strip()
            if role == "assistant" and row.get("tool_calls") and not prev.get("tool_calls"):
                prev["tool_calls"] = row["tool_calls"]
            continue
        out.append(row)

    if len(out) >= 2 and out[0]["role"] == "system" and out[1]["role"] != "user":
        out.insert(1, {"role": "user", "content": "(continue)"})
    if not out:
        out.append({"role": "user", "content": ""})
    elif out[0]["role"] != "system" and out[0]["role"] != "user":
        out.insert(0, {"role": "user", "content": ""})
    return out
