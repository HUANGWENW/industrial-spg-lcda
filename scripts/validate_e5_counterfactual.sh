#!/usr/bin/env bash
set -euo pipefail

WEIGHTS="${1:?usage: bash scripts/validate_e5_counterfactual.sh <best.pt> [extra args]}"
shift
python -m spg_lcda.validate_e5_counterfactual --weights "$WEIGHTS" "$@"
