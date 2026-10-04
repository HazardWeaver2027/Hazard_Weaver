"""Thin OpenAI-compatible chat client for vLLM (stdlib urllib — no openai dep)."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union


def _agent_llm_deadline_monotonic() -> Optional[float]:
    raw = os.environ.get("HW_LLM_DEADLINE_MONOTONIC")
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _retry_budget(timeout_s: float, max_retries: int) -> Tuple[float, int]:
    """Cap per-attempt timeout and retry count to remaining agent wall budget."""
    deadline = _agent_llm_deadline_monotonic()
    if deadline is None:
        return timeout_s, max_retries
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise RuntimeError("LLM request aborted: agent wall budget exhausted")
    timeout_s = min(timeout_s, max(0.1, remaining))
    capped_retries = min(max_retries, max(1, int(remaining // max(timeout_s, 0.1)) + 1))
    return timeout_s, capped_retries


def _sleep_before_retry(seconds: float) -> None:
    deadline = _agent_llm_deadline_monotonic()
    wait_s = seconds
    if deadline is not None:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError("LLM request aborted: agent wall budget exhausted during retry")
        wait_s = min(wait_s, remaining)
    if wait_s > 0:
        time.sleep(wait_s)


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str

    def to_dict(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass
class VLLMClientConfig:
    """Connection settings. Defaults favor L4 iteration model."""

    base_url: str = field(
        default_factory=lambda: os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000/v1").rstrip("/")
    )
    model_id: str = field(
        default_factory=lambda: os.environ.get(
            "VLLM_MODEL_ID", "Qwen/Qwen2.5-Coder-7B-Instruct"
        )
    )
    api_key: str = field(
        default_factory=lambda: os.environ.get("VLLM_API_KEY", "EMPTY")
    )
    timeout_s: float = field(
        default_factory=lambda: float(os.environ.get("VLLM_TIMEOUT_S", "120"))
    )
    max_retries: int = field(
        default_factory=lambda: int(os.environ.get("VLLM_MAX_RETRIES", "6"))
    )
    retry_backoff_s: float = field(
        default_factory=lambda: float(os.environ.get("VLLM_RETRY_BACKOFF_S", "2.0"))
    )
    profile: str = field(
        default_factory=lambda: os.environ.get("HW_LLM_PROFILE", "l4_fast")
    )


class VLLMChatClient:
    """OpenAI-compatible `/v1/chat/completions` client aimed at local vLLM."""

    def __init__(self, config: VLLMClientConfig | None = None):
        self.config = config or VLLMClientConfig()

    @property
    def provider(self) -> str:
        from hazardweaver.hwa.llm.provider_normalize import normalize_llm_provider

        transport = (
            "minimax"
            if os.environ.get("HW_CLOSED_API_PROVIDER") == "minimax"
            else "vllm"
        )
        return normalize_llm_provider(
            transport,
            profile=self.config.profile,
            model_id=self.config.model_id,
        )

    @property
    def transport(self) -> str:
        """Underlying chat transport (vLLM local or closed OpenAI-compat API)."""
        if os.environ.get("HW_CLOSED_API_PROVIDER") == "minimax":
            return "closed_api_minimax"
        return "vllm"

    @property
    def model_id(self) -> str:
        return self.config.model_id

    def list_models(self) -> Dict[str, Any]:
        return self._request("GET", "/models")

    def chat(
        self,
        messages: Sequence[Union[ChatMessage, Mapping[str, Any]]],
        *,
        temperature: float = 0.2,
        max_tokens: int = 512,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        from hazardweaver.hwa.llm.strict_chat_template_v1 import (
            needs_strict_role_alternation,
            sanitize_strict_chat_template_messages,
        )
        from hazardweaver.hwa.llm.tool_call_ids_v1 import sanitize_vllm_chat_messages

        msgs = sanitize_vllm_chat_messages([_coerce_message(m) for m in messages])
        if needs_strict_role_alternation(
            model_id=self.config.model_id, profile=self.config.profile
        ):
            msgs = sanitize_strict_chat_template_messages(msgs)
        payload: Dict[str, Any] = {
            "model": self.config.model_id,
            "messages": msgs,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if extra:
            payload.update(dict(extra))
        return self._request("POST", "/chat/completions", payload)

    def complete_text(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> str:
        msgs: List[ChatMessage] = []
        if system:
            msgs.append(ChatMessage(role="system", content=system))
        msgs.append(ChatMessage(role="user", content=prompt))
        resp = self.chat(msgs, temperature=temperature, max_tokens=max_tokens)
        try:
            return str(resp["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"unexpected chat response: {resp!r}") from exc

    def _request(
        self,
        method: str,
        path: str,
        payload: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        url = f"{self.config.base_url}{path if path.startswith('/') else '/' + path}"
        data = None
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        last_exc: Optional[BaseException] = None
        timeout_s, attempts = _retry_budget(
            float(self.config.timeout_s), max(1, int(self.config.max_retries))
        )
        for attempt in range(attempts):
            try:
                timeout_s, attempts = _retry_budget(timeout_s, attempts)
                with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                    body = resp.read().decode("utf-8")
                return json.loads(body)
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                last_exc = RuntimeError(f"vLLM HTTP {exc.code} at {url}: {detail}")
                if not _retryable_http_error(exc.code, detail) or attempt >= attempts - 1:
                    raise last_exc from exc
            except urllib.error.URLError as exc:
                last_exc = RuntimeError(
                    f"vLLM unreachable at {url}. Start serve on a compute node "
                    f"(see docs/engineering/wildfire_dual_system_mvp/LLM_RUNTIME.md). "
                    f"Underlying: {exc}"
                )
                if attempt >= attempts - 1:
                    raise last_exc from exc
            if attempt < attempts - 1:
                _sleep_before_retry(min(self.config.retry_backoff_s * (2**attempt), 30.0))
        if last_exc is not None:
            raise last_exc
        raise RuntimeError(f"vLLM request failed at {url}")


def _messages_for_ollama(
    messages: Sequence[Union[ChatMessage, Mapping[str, Any]]],
) -> List[Dict[str, Any]]:
    """Map OpenAI-style chat history to Ollama /api/chat messages (native tool roles)."""
    out: List[Dict[str, Any]] = []
    for raw in messages:
        m = _coerce_message(raw)
        role = str(m.get("role") or "user")
        if role == "tool":
            row: Dict[str, Any] = {
                "role": "tool",
                "content": str(m.get("content") or ""),
            }
            if m.get("name"):
                row["tool_name"] = str(m["name"])
            if m.get("tool_call_id"):
                row["tool_call_id"] = str(m["tool_call_id"])
            out.append(row)
            continue
        if role == "assistant" and m.get("tool_calls"):
            out.append(
                {
                    "role": "assistant",
                    "content": str(m.get("content") or ""),
                    "tool_calls": _ollama_tool_calls_for_request(m.get("tool_calls") or []),
                }
            )
            continue
        if role not in {"system", "user", "assistant"}:
            role = "user"
        out.append({"role": role, "content": str(m.get("content") or "")})
    return out


def _ollama_tool_calls_for_request(tool_calls: Any) -> List[Dict[str, Any]]:
    """Normalize stored OpenAI tool_calls for Ollama request payload."""
    if not isinstance(tool_calls, list):
        return []
    out: List[Dict[str, Any]] = []
    for i, tc in enumerate(tool_calls):
        if not isinstance(tc, Mapping):
            continue
        fn = tc.get("function") if isinstance(tc.get("function"), Mapping) else {}
        name = str(fn.get("name") or "")
        args = fn.get("arguments")
        if isinstance(args, str):
            try:
                args_obj = json.loads(args) if args.strip() else {}
            except json.JSONDecodeError:
                args_obj = {}
        elif isinstance(args, Mapping):
            args_obj = dict(args)
        else:
            args_obj = {}
        out.append(
            {
                "id": str(tc.get("id") or f"call_{i}"),
                "type": "function",
                "function": {"name": name, "arguments": args_obj},
            }
        )
    return out


def _openai_tool_calls_from_ollama(tool_calls: Any) -> List[Dict[str, Any]]:
    """Normalize Ollama response tool_calls to OpenAI chat.completion shape."""
    if not isinstance(tool_calls, list):
        return []
    out: List[Dict[str, Any]] = []
    for i, tc in enumerate(tool_calls):
        if not isinstance(tc, Mapping):
            continue
        if isinstance(tc.get("function"), Mapping):
            fn = dict(tc["function"])
            args = fn.get("arguments")
            if isinstance(args, Mapping):
                fn["arguments"] = json.dumps(args, ensure_ascii=False)
            elif args is None:
                fn["arguments"] = "{}"
            else:
                fn["arguments"] = str(args)
            out.append(
                {
                    "id": str(tc.get("id") or f"call_{i}"),
                    "type": str(tc.get("type") or "function"),
                    "function": fn,
                }
            )
            continue
        out.append(dict(tc))
    return out


def _ollama_tools_for_request(tools: Any) -> List[Dict[str, Any]]:
    """Pass OpenAI-shaped tool specs through to Ollama (native tools API)."""
    if not isinstance(tools, list):
        return []
    return [dict(t) for t in tools if isinstance(t, Mapping)]


def _ollama_tool_choice_for_request(tool_choice: Any) -> Any:
    if tool_choice is None:
        return None
    if isinstance(tool_choice, str):
        return tool_choice
    if isinstance(tool_choice, Mapping):
        return dict(tool_choice)
    return tool_choice


class OllamaCloudChatClient:
    """Ollama Cloud native /api/chat client with OpenAI-shaped ``chat()`` responses."""

    def __init__(self, config: VLLMClientConfig | None = None):
        self.config = config or VLLMClientConfig()

    @property
    def provider(self) -> str:
        from hazardweaver.hwa.llm.provider_normalize import normalize_llm_provider

        tag = os.environ.get("HW_CLOSED_API_PROVIDER_TAG", "").strip()
        raw = tag or os.environ.get("HW_CLOSED_API_PROVIDER", "minimax")
        return normalize_llm_provider(
            raw,
            profile=self.config.profile,
            model_id=self.config.model_id,
        )

    @property
    def transport(self) -> str:
        return "closed_api_ollama"

    @property
    def model_id(self) -> str:
        return self.config.model_id

    def list_models(self) -> Dict[str, Any]:
        return {"object": "list", "data": [{"id": self.config.model_id}]}

    def chat(
        self,
        messages: Sequence[Union[ChatMessage, Mapping[str, Any]]],
        *,
        temperature: float = 0.2,
        max_tokens: int = 512,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        host = self.config.base_url.rstrip("/")
        url = f"{host}/api/chat"
        extra_map = dict(extra or {})
        tools = extra_map.get("tools")
        ollama_messages = _messages_for_ollama(messages)
        wire_model = self.config.model_id
        try:
            from hazardweaver.baselines.baselines.system.ds_agent.ollama_cloud_client import (
                normalize_ollama_cloud_model,
            )

            wire_model = normalize_ollama_cloud_model(wire_model)
        except ImportError:
            pass
        payload: Dict[str, Any] = {
            "model": wire_model,
            "messages": ollama_messages,
            "stream": False,
            "options": {
                "temperature": float(temperature),
                "num_predict": int(max_tokens),
            },
        }
        if tools:
            payload["tools"] = _ollama_tools_for_request(tools)
            tool_choice = _ollama_tool_choice_for_request(extra_map.get("tool_choice"))
            if tool_choice is not None:
                payload["tool_choice"] = tool_choice
        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        last_exc: Optional[BaseException] = None
        timeout_s, attempts = _retry_budget(
            float(self.config.timeout_s), max(1, int(self.config.max_retries))
        )
        for attempt in range(attempts):
            try:
                timeout_s, attempts = _retry_budget(timeout_s, attempts)
                with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                message = body.get("message")
                content = ""
                tool_calls = None
                if isinstance(message, Mapping):
                    content = str(message.get("content") or "")
                    tool_calls = message.get("tool_calls")
                else:
                    content = str(body.get("response") or "")
                out_message: Dict[str, Any] = {"role": "assistant", "content": content}
                if isinstance(tool_calls, list) and tool_calls:
                    out_message["tool_calls"] = _openai_tool_calls_from_ollama(tool_calls)
                return {
                    "id": "ollama-chat",
                    "object": "chat.completion",
                    "model": self.config.model_id,
                    "choices": [
                        {
                            "index": 0,
                            "message": out_message,
                            "finish_reason": "stop",
                        }
                    ],
                }
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                last_exc = RuntimeError(f"Ollama HTTP {exc.code} at {url}: {detail[:500]}")
                if not _retryable_http_error(exc.code, detail) or attempt >= attempts - 1:
                    raise last_exc from exc
            except urllib.error.URLError as exc:
                last_exc = RuntimeError(f"Ollama unreachable at {url}: {exc}")
                if attempt >= attempts - 1:
                    raise last_exc from exc
            if attempt < attempts - 1:
                if isinstance(last_exc, RuntimeError) and "HTTP 429" in str(last_exc):
                    backoff = min(30.0 * (2**attempt), 120.0)
                else:
                    backoff = min(self.config.retry_backoff_s * (2**attempt), 30.0)
                _sleep_before_retry(backoff)
        if last_exc is not None:
            raise last_exc
        raise RuntimeError(f"Ollama request failed at {url}")

    def complete_text(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> str:
        msgs: List[ChatMessage] = []
        if system:
            msgs.append(ChatMessage(role="system", content=system))
        msgs.append(ChatMessage(role="user", content=prompt))
        resp = self.chat(msgs, temperature=temperature, max_tokens=max_tokens)
        return str(resp["choices"][0]["message"]["content"])


def _retryable_http_error(code: int, detail: str) -> bool:
    if code in {408, 409, 429, 500, 502, 503, 504}:
        return True
    lowered = detail.lower()
    if code == 400 and (
        "maximum context length" in lowered
        or "context length" in lowered
        or "input_tokens" in lowered
    ):
        return False
    return code >= 500


def _coerce_message(m: Union[ChatMessage, Mapping[str, Any]]) -> Dict[str, Any]:
    """Normalize chat messages; preserve OpenAI tool-calling fields when present."""
    if isinstance(m, ChatMessage):
        return m.to_dict()
    out: Dict[str, Any] = {"role": str(m["role"])}
    if "content" in m and m["content"] is not None:
        out["content"] = m["content"]
    elif "tool_calls" not in m:
        out["content"] = ""
    for key in ("tool_calls", "tool_call_id", "name", "refusal"):
        if key in m and m[key] is not None:
            out[key] = m[key]
    return out


def client_from_env(*, profile: Optional[str] = None) -> Union[VLLMChatClient, OllamaCloudChatClient]:
    """Build client from env + optional registry profile.

    Precedence for ``model_id``:
      1. Explicit ``VLLM_MODEL_ID`` (W1 multi-arm compare must not be clobbered)
      2. Registry ``hf_id`` for a *known* profile role
      3. ``VLLMClientConfig`` default (7B)

    Unknown ``HW_LLM_PROFILE`` values are treated as labels only when
    ``VLLM_MODEL_ID`` is set (avoids KeyError on invented roles like
    ``vllm_fp8``). Without an explicit model id, unknown roles still raise.
    """
    if os.environ.get("HW_CLOSED_API_TRANSPORT") == "ollama_cloud":
        cfg = VLLMClientConfig()
        return OllamaCloudChatClient(cfg)

    cfg = VLLMClientConfig()
    explicit_model = os.environ.get("VLLM_MODEL_ID")
    if not profile:
        return VLLMChatClient(cfg)

    from hazardweaver.hwa.llm.registry import default_registry

    try:
        meta = default_registry().get(profile)
    except KeyError:
        if not explicit_model:
            raise
        return VLLMChatClient(
            VLLMClientConfig(
                base_url=cfg.base_url,
                model_id=explicit_model,
                api_key=cfg.api_key,
                timeout_s=cfg.timeout_s,
                profile=profile,
            )
        )

    return VLLMChatClient(
        VLLMClientConfig(
            base_url=cfg.base_url,
            model_id=explicit_model or str(meta["hf_id"]),
            api_key=cfg.api_key,
            timeout_s=cfg.timeout_s,
            profile=profile,
        )
    )
