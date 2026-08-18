import json
import platform
import subprocess

import torch


def nvidia_smi() -> str:
    try:
        return subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            text=True,
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unavailable"


report = {
    "platform": platform.platform(),
    "python": platform.python_version(),
    "torch": torch.__version__,
    "cuda_runtime": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "gpu_count": torch.cuda.device_count(),
    "nvidia_smi": nvidia_smi(),
}

print(json.dumps(report, indent=2, ensure_ascii=False))
if not report["cuda_available"]:
    raise SystemExit("CUDA is unavailable; verify the NVIDIA driver and PyTorch wheel.")

