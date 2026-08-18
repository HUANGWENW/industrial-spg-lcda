#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:?usage: bash scripts/train.sh <config.yaml>}"
python -m spg_lcda.cli --config "$CONFIG_PATH"

