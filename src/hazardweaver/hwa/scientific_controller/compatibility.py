"""Schema/interface, units, CRS, and time compatibility checks."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from hazardweaver.hwa.scientific_controller.reason_codes import ACapVerdict


def _support_dict(support: Any) -> Dict[str, Any]:
    if support is None:
        return {}
    if hasattr(support, "model_dump"):
        return support.model_dump(mode="json")
    if isinstance(support, Mapping):
        return dict(support)
    return {}


def check_support_compatibility(
    required: Any,
    available: Any,
) -> Tuple[bool, List[str]]:
    """Wrap hcg.compatibility.supports_compatible when available."""
    try:
        from hazardweaver.hcg.compatibility import supports_compatible
        from hazardweaver.hcg.models import SupportSpec

        req = SupportSpec.model_validate(required) if not isinstance(required, SupportSpec) else required
        avail = SupportSpec.model_validate(available) if not isinstance(available, SupportSpec) else available
        ok, issues = supports_compatible(req, avail)
        return ok, list(issues)
    except Exception:
        req = _support_dict(required)
        avail = _support_dict(available)
        issues: List[str] = []
        for key in ("crs", "units", "temporal_scale", "schema_id"):
            rv, av = req.get(key), avail.get(key)
            if rv is not None and av is not None and str(rv) != str(av):
                issues.append(f"{key}_mismatch:{rv}!={av}")
        return len(issues) == 0, issues


def compatibility_verdict_from_issues(issues: Sequence[str]) -> str:
    joined = " ".join(issues).lower()
    if "schema" in joined:
        return ACapVerdict.INCOMPATIBLE_SCHEMA.value
    if "units" in joined:
        return ACapVerdict.INCOMPATIBLE_UNITS.value
    if "crs" in joined:
        return ACapVerdict.INCOMPATIBLE_CRS.value
    if "temporal" in joined or "time" in joined:
        return ACapVerdict.INCOMPATIBLE_TIME.value
    return ACapVerdict.INCOMPATIBLE_SCHEMA.value


def evaluate_interface_compatibility(
    *,
    input_contract: Optional[Mapping[str, Any]] = None,
    output_contract: Optional[Mapping[str, Any]] = None,
    available_support: Optional[Mapping[str, Any]] = None,
) -> Tuple[str, List[str]]:
    """Return (verdict, issue_codes) for schema/interface checks."""
    issues: List[str] = []
    ic = dict(input_contract or {})
    oc = dict(output_contract or {})
    avail = dict(available_support or {})

    required_schema = ic.get("schema_id") or oc.get("schema_id")
    if required_schema and avail.get("schema_id") and str(required_schema) != str(avail["schema_id"]):
        issues.append(f"schema_id_mismatch:{required_schema}!={avail['schema_id']}")

    if ic.get("units") and avail.get("units") and str(ic["units"]) != str(avail["units"]):
        issues.append(f"units_mismatch:{ic['units']}!={avail['units']}")
    if ic.get("crs") and avail.get("crs") and str(ic["crs"]) != str(avail["crs"]):
        issues.append(f"crs_mismatch:{ic['crs']}!={avail['crs']}")
    if ic.get("temporal_scale") and avail.get("temporal_scale"):
        if str(ic["temporal_scale"]) != str(avail["temporal_scale"]):
            issues.append(f"temporal_scale_mismatch:{ic['temporal_scale']}!={avail['temporal_scale']}")

    if issues:
        return compatibility_verdict_from_issues(issues), issues
    return ACapVerdict.REACHABLE.value, []
