#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${SPG_ENV_NAME:-spg-lcda}"
TORCH_INDEX_URL="${SPG_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"

if ! command -v conda >/dev/null 2>&1; then
  echo "Conda is required. Install Miniconda or Mambaforge first." >&2
  exit 1
fi

conda env create -n "$ENV_NAME" -f environment.yml || \
  conda env update -n "$ENV_NAME" -f environment.yml --prune

eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

python -m pip install --upgrade pip==24.3.1
python -m pip install \
  torch==2.5.1 torchvision==0.20.1 \
  --index-url "$TORCH_INDEX_URL"
python -m pip install -r requirements/base.txt -r requirements/dev.txt
python -m pip install -e . --no-deps

python tools/check_environment.py
bash scripts/smoke_test.sh

