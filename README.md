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

For a Linux GPU server, follow `docs/server_setup.md` or run
`bash scripts/setup_server.sh`. The default reproducible stack is Python 3.11,
PyTorch 2.5.1, torchvision 0.20.1, CUDA 12.1 wheels, YOLOv8, and OpenCLIP.

## Repository layout

```text
configs/       Data paths and experiment configurations
src/           Dataset, model, prompt, loss, and evaluation modules
scripts/       Environment setup and experiment entry points
tools/         Dataset/environment validation utilities
tests/         Fast unit and smoke tests
docs/          Experiment registry and server deployment notes
```

Copy `.env.example` to `.env` and set server-local data/output paths. Dataset files, model weights, secrets, and experiment outputs are intentionally excluded from Git.

## Reproducibility

Every formal run should record its Git commit, configuration, seed, dataset version, GPU, package environment, command, checkpoint, and final metrics.
