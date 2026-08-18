#!/usr/bin/env bash
set -euo pipefail

python -m spg_lcda.cli --config configs/experiment/smoke.yaml --dry-run
python -m pytest -q

