#!/usr/bin/env python3
"""Run a single benchmark instance with a configured model (optional, compute-heavy)."""

from __future__ import annotations

import argparse
import json
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="Optional agent re-run (requires inference stack).")
    parser.add_argument("--instance-id", default="")
    parser.add_argument("--model-config", default="configs/models/llama70b.yaml")
    args = parser.parse_args()
    print(
        json.dumps(
            {
                "error": "full_agent_rerun_not_packaged",
                "message": "Use released_results to audit paper numbers. Wire your vLLM/API env and model configs to run agents.",
                "instance_id": args.instance_id or None,
                "model_config": args.model_config,
            },
            indent=2,
        ),
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
