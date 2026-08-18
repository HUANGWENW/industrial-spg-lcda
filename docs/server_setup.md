# Server environment

## Assumed MVP platform

- Linux x86_64
- NVIDIA GPU with a driver compatible with CUDA 12.1 wheels
- Conda or Mamba
- Python 3.11
- Git
- At least 16 GB system RAM; 8 GB GPU memory is the practical minimum for
  YOLOv8n at 640 px with a reduced batch size

## Installation

```bash
git clone <private-repository-url>
cd industrial-spg-lcda
cp .env.example .env
# Edit the three SPG_* paths before continuing.
bash scripts/setup_server.sh
```

The setup script uses the project-local Tsinghua Conda configuration at
`configs/conda/condarc.tuna.yaml`. It therefore ignores stale channels in the
server's user-level `.condarc`, including the removed `anaconda/pkgs/free`.

To repair the server-wide Conda configuration as well, first inspect its
sources and remove the obsolete entry:

```bash
conda config --show-sources
conda config --remove channels \
  https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free || true
conda clean --index-cache -y
```

If `pkgs/free` appears under `default_channels` rather than `channels`, edit
the `.condarc` file reported by `conda config --show-sources` and remove only
that line. Do not add `pkgs/free` back; the channel has been retired.

If the server driver requires another supported PyTorch wheel, override it:

```bash
SPG_TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124 \
  bash scripts/setup_server.sh
```

Do not copy datasets, checkpoints, secrets, or run outputs into Git. Keep them
under the server paths configured in `.env`.

## Required checks before training

```bash
nvidia-smi
conda activate spg-lcda
python tools/check_environment.py
bash scripts/smoke_test.sh
```

Record `git rev-parse HEAD`, the configuration file, seed, data version, GPU,
and `python -m pip freeze` for every formal run.
