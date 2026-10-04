PYTHON ?= python3
ROOT := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

.PHONY: setup verify smoke evaluate-released reproduce-paper test

setup:
	$(PYTHON) -m pip install -e "$(ROOT).[dev]"

verify:
	$(PYTHON) "$(ROOT)scripts/verify_release.py"

smoke:
	$(PYTHON) "$(ROOT)scripts/run_smoke.py"

evaluate-released:
	$(PYTHON) "$(ROOT)scripts/evaluate_runs.py"

reproduce-paper:
	$(PYTHON) "$(ROOT)scripts/reproduce_paper.py"

test:
	cd "$(ROOT)" && $(PYTHON) -m pytest tests -q
