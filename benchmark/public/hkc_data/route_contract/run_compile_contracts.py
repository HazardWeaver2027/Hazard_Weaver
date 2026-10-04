"""CLI: compile Route Cards → Scientific Route Contracts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT / "docs/final_four/HKC") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "docs/final_four/HKC"))

from route_contract.converter import compile_contract_from_route_card


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route-cards", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--evidence-manifest", type=Path, default=None)
    args = parser.parse_args(argv)

    evidence_by_item: Dict[str, str] = {}
    if args.evidence_manifest and args.evidence_manifest.is_file():
        for row in _read_jsonl(args.evidence_manifest):
            iid = str(row.get("item_id") or "")
            if iid:
                evidence_by_item[iid] = str(row.get("evidence_text") or "")

    cards = _read_jsonl(args.route_cards)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with args.out.open("w", encoding="utf-8") as fh:
        for card in cards:
            if card.get("error"):
                continue
            iid = str(card.get("item_id") or "")
            ev = evidence_by_item.get(iid) or ""
            contract = compile_contract_from_route_card(card, ev)
            row = contract.to_dict()
            row["item_id"] = iid
            row["paper_id"] = str(card.get("paper_id") or "")
            row["route_card_id"] = str(card.get("route_card_id") or "")
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    print(json.dumps({"n_contracts": n, "out": str(args.out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
