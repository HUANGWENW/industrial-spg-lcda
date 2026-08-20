#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:?usage: bash scripts/train_e5.sh <e5_g.yaml|e5_s.yaml> [extra args]}"
shift
python -m spg_lcda.train_e5 --config "$CONFIG_PATH" "$@"
