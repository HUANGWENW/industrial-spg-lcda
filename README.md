# industrial-spg-lcda

Minimal reproducible implementation for parameter-grounded, localization-centric domain adaptation in single-class industrial 2D Sim2Real detection.

## MVP experiment path

1. Synthetic-only detector baseline.
2. Simulator-parameter-grounded prompt alignment (`L_shift`).
3. Industrial hard-negative prompt bank (`L_HN`).
4. Paired-render localization consistency (`L_loc_cons`).
5. Zero-shot evaluation, followed by optional 1/5-shot calibration.

## Quick start

```bash
conda env create -f environment.yml
conda activate spg-lcda
python -m pip install -e .
bash scripts/smoke_test.sh
```

Copy `.env.example` to `.env` and set server-local data/output paths. Dataset files, model weights, secrets, and experiment outputs are intentionally excluded from Git.

## Reproducibility

Every formal run should record its Git commit, configuration, seed, dataset version, GPU, package environment, command, checkpoint, and final metrics.

