"""Phase C scientific tier paths, manifests, and eligibility checks."""

from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = ["check_scientific_eligibility"]

if TYPE_CHECKING:
    from hazardweaver.hcg.carp.scientific.tier import check_scientific_eligibility


def __getattr__(name: str):
    if name == "check_scientific_eligibility":
        from hazardweaver.hcg.carp.scientific.tier import check_scientific_eligibility

        return check_scientific_eligibility
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
