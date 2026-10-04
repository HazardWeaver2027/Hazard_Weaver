"""P0-5 slot-gated scripted user for Clarify-Resolution.

Claim Guard: scripted_user ≠ STRENGTH; Decision mode does not require this.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


def _norm(s: str) -> str:
    return str(s).strip().lower().replace("-", "_")


def _resolve_alias(asked: str, required: Sequence[str]) -> Optional[str]:
    """Map a near-miss asked slot onto exactly one required slot when unambiguous.

    Allows ``time_window`` → ``event_time_window`` (unique suffix / containment),
    never fuzzy multi-hit guessing.
    """
    a = _norm(asked)
    req = [str(r) for r in required]
    exact = [r for r in req if _norm(r) == a]
    if len(exact) == 1:
        return exact[0]
    if exact:
        return None
    suffix = [r for r in req if _norm(r).endswith("_" + a) or _norm(r).endswith(a)]
    if len(suffix) == 1:
        return suffix[0]
    contain = [r for r in req if a and a in _norm(r)]
    if len(contain) == 1:
        return contain[0]
    return None


def respond(
    *,
    asked_slots: Sequence[str],
    scripted_user: Mapping[str, Any],
) -> Dict[str, Any]:
    """Return answered / refused / irrelevant based on asked slots vs required."""
    required = [str(s) for s in (scripted_user.get("required_slots") or [])]
    answers_cfg = dict(scripted_user.get("answers") or {})
    refuse_policy = str(scripted_user.get("refuse_policy") or "irrelevant")
    asked = [str(s) for s in asked_slots]

    if not asked:
        return {
            "status": "refused",
            "ok": True,
            "note": "no slots requested",
            "answers": {},
        }

    # Exact hits first
    hits = [s for s in asked if s in required or s in answers_cfg]
    matched_required: List[str] = []
    alias_map: Dict[str, str] = {}
    if not hits:
        for s in asked:
            mapped = _resolve_alias(s, required)
            if mapped:
                hits.append(s)
                matched_required.append(mapped)
                alias_map[s] = mapped

    if hits:
        out_answers = {}
        for s in hits:
            key = alias_map.get(s, s)
            # Prefer answer keyed by required/canonical slot
            out_answers[key] = answers_cfg.get(key) or answers_cfg.get(s) or f"resolved_{key}"
            if key != s:
                out_answers[s] = out_answers[key]
        return {
            "status": "answered",
            "ok": True,
            "answers": out_answers,
            "matched_slots": hits,
            "matched_required_slots": matched_required or hits,
            "alias_map": alias_map or None,
        }

    note = (
        "user reply does not resolve the missing information; "
        f"re-ask ask_user with slots exactly equal to {required}, "
        "then call submit_clarification with that slot_id"
    )
    if refuse_policy == "refuse":
        return {
            "status": "refused",
            "ok": True,
            "note": note,
            "answers": {},
            "asked_slots": asked,
            "required_slots": required,
        }
    return {
        "status": "irrelevant",
        "ok": True,
        "note": note,
        "answers": {"distractor": "not sure about that"},
        "asked_slots": asked,
        "required_slots": required,
    }
