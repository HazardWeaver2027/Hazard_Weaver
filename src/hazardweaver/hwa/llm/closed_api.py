"""Load closedAPI credentials for HWA paper runs (MiniMax, future Azure/Bedrock)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_KEY_ENV = PROJECT_ROOT / "closedAPI" / "api_key.env"

# OpenAI-compatible endpoints. Keys come from api_key.env (never commit).
PROVIDER_SPECS: Dict[str, Dict[str, Any]] = {
    "minimax": {
        # International keys → api.minimax.io ; mainland keys → api.minimaxi.com
        "base_url": "https://api.minimax.io/v1",
        "base_url_cn": "https://api.minimaxi.com/v1",
        "default_model": "MiniMax-M2.7",
        "profile": "closed_api_minimax",
        "provider_tag": "minimax_m27",
        # ollama_minimax27: same key as DS-Agent Ollama Cloud (api_key.env.example).
        "key_aliases": ("minimax", "minimax_api", "ollama_minimax27"),
        "api_key_env": "MINIMAX_API_KEY",
        "ollama_key_aliases": ("ollama_minimax27",),
    },
    "deepseek_v4_flash": {
        # Ollama Cloud native /api/chat — unified 143 uses deepseek-v4.1-flash:cloud (PI 2026-09).
        "base_url": "https://ollama.com",
        "default_model": "deepseek-v4.1-flash:cloud",
        "profile": "closed_api_ollama_deepseek",
        "provider_tag": "deepseek_v4_flash",
        "key_aliases": (
            "deepseek_v4_flash",
            "ollama_deepseekv4flash",
            "ollama_deepseekv41_1flash",
            "ollama_deepseekv41_2flash",
        ),
        "api_key_env": "OLLAMA_API_KEY",
        "ollama_key_aliases": (
            "ollama_deepseekv4flash",
            "ollama_deepseekv41_1flash",
            "ollama_deepseekv41_2flash",
        ),
    },
    "gemini": {
        # OpenAI-compatible Gemini endpoint (Google AI Studio / Vertex OpenAPI compat).
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "default_model": "gemini-2.0-flash-lite",
        "profile": "closed_api_gemini",
        "provider_tag": "gemini_flash_lite",
        "key_aliases": ("gemini", "google", "gemini_api", "google_api"),
        "api_key_env": "GEMINI_API_KEY",
    },
}


def parse_api_key_env(text: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        out[key.strip().lower()] = value.strip()
    return out


def load_api_key_env(path: Path | None = None) -> Dict[str, str]:
    env_path = Path(path or os.environ.get("HW_CLOSED_API_KEY_ENV", DEFAULT_KEY_ENV))
    if not env_path.is_file():
        raise FileNotFoundError(
            f"closed API key file missing: {env_path} "
            f"(expected lines like 'minimax: <key>')"
        )
    return parse_api_key_env(env_path.read_text(encoding="utf-8"))


def _minimax_base_url(spec: Mapping[str, Any]) -> str:
    explicit = os.environ.get("HW_MINIMAX_BASE_URL", "").strip()
    if explicit:
        return explicit.rstrip("/")
    region = os.environ.get("HW_MINIMAX_REGION", "global").strip().lower()
    if region in {"cn", "china", "mainland"}:
        return str(spec.get("base_url_cn") or spec["base_url"]).rstrip("/")
    return str(spec["base_url"]).rstrip("/")


def _provider_base_url(spec: Mapping[str, Any]) -> str:
    if str(spec.get("provider_tag", "")).startswith("minimax"):
        return _minimax_base_url(spec)
    explicit = os.environ.get("HW_CLOSED_API_BASE_URL", "").strip()
    if explicit:
        return explicit.rstrip("/")
    return str(spec.get("base_url") or "").rstrip("/")


def _resolve_key(
    keys: Mapping[str, str],
    aliases: tuple[str, ...],
    *,
    preferred_alias: str | None = None,
) -> tuple[str, str]:
    if preferred_alias:
        val = keys.get(preferred_alias.lower(), "").strip()
        if val:
            return val, preferred_alias.lower()
    for alias in aliases:
        val = keys.get(alias.lower(), "").strip()
        if val:
            return val, alias.lower()
    raise KeyError(f"no API key for aliases {aliases} in closedAPI/api_key.env")


def apply_closed_api_provider(
    provider: str,
    *,
    key_env: Path | None = None,
    model_id: str | None = None,
) -> Dict[str, str]:
    """Set process env for HWA OpenAI-compat client. Returns applied vars (no secrets)."""
    spec = PROVIDER_SPECS.get(provider)
    if spec is None:
        raise KeyError(f"unsupported closed API provider: {provider}")

    keys = load_api_key_env(key_env)
    preferred = (
        os.environ.get("HW_CLOSED_API_KEY_ALIAS", "").strip()
        or os.environ.get("HW_CLOSED_API_FORCE_KEY_ALIAS", "").strip()
        or None
    )
    api_key, key_alias = _resolve_key(keys, tuple(spec["key_aliases"]), preferred_alias=preferred)
    ollama_aliases = tuple(spec.get("ollama_key_aliases") or ())
    if key_alias in ollama_aliases:
        # Ollama Cloud uses native POST /api/chat (not OpenAI /v1/chat/completions).
        base_url = os.environ.get("OLLAMA_CLOUD_HOST", "https://ollama.com").strip().rstrip("/")
        model = (
            model_id
            or os.environ.get("OLLAMA_MODEL")
            or os.environ.get("HW_CLOSED_API_MODEL")
            or str(spec.get("default_model") or "minimax-m2.7")
        ).strip()
        # Registry may use MiniMax-M2.7; Ollama Cloud model id is minimax-m2.7.
        if model.lower().replace("_", "-") in {"minimax-m2.7", "minimax-m27"}:
            model = "minimax-m2.7"
        from hazardweaver.baselines.baselines.system.ds_agent.ollama_cloud_client import (
            normalize_ollama_cloud_model,
        )

        model = normalize_ollama_cloud_model(model)
    else:
        model = (model_id or os.environ.get("HW_CLOSED_API_MODEL") or spec["default_model"]).strip()
        base_url = _provider_base_url(spec)
    timeout_s = os.environ.get("HW_CLOSED_API_TIMEOUT_S", "180")

    transport = "ollama_cloud" if key_alias in ollama_aliases else "openai_compat"
    applied = {
        "VLLM_BASE_URL": base_url,
        "VLLM_MODEL_ID": model,
        "VLLM_API_KEY": api_key,
        spec["api_key_env"]: api_key,
        "HW_LLM_PROFILE": str(spec["profile"]),
        "HW_CLOSED_API_PROVIDER": provider,
        "HW_CLOSED_API_PROVIDER_TAG": str(spec["provider_tag"]),
        "HW_CLOSED_API_TRANSPORT": transport,
        "VLLM_TIMEOUT_S": timeout_s,
    }
    for key, val in applied.items():
        os.environ[key] = val
    return {
        "provider": provider,
        "provider_tag": spec["provider_tag"],
        "base_url": base_url,
        "model_id": model,
        "profile": spec["profile"],
        "transport": transport,
        "key_alias": key_alias,
        "key_env": str(key_env or DEFAULT_KEY_ENV),
    }
