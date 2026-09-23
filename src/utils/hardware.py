"""
Hardware environment inspector and resource constraint validator.
WIUT Hackathon 2026 — Algorix Team

Hardware Constraints (Rule 3 & 5):
- 1x GPU (T4-class, 16GB VRAM)
- 8 CPU cores
- 32GB RAM
- Total weights size <= 5 GB (Rule 3)
- Execution time <= video_duration * 3 (Rule 4)
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Tuple
import psutil

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# Constants from Rule 3, 4, 5
TARGET_GPU_VRAM_GB = 16.0
TARGET_CPU_CORES = 8
TARGET_RAM_GB = 32.0
MAX_WEIGHTS_SIZE_BYTES = 5 * 1024 * 1024 * 1024  # 5 GB
MAX_TIME_FACTOR = 3.0  # video_duration * 3


def get_hardware_info() -> Dict[str, Any]:
    """Inspect current machine hardware specifications."""
    mem = psutil.virtual_memory()
    info: Dict[str, Any] = {
        "python_version": sys.version.split()[0],
        "cpu_count_logical": psutil.cpu_count(logical=True) or 1,
        "cpu_count_physical": psutil.cpu_count(logical=False) or 1,
        "ram_total_gb": round(mem.total / (1024 ** 3), 2),
        "ram_available_gb": round(mem.available / (1024 ** 3), 2),
        "cuda_available": False,
        "gpu_count": 0,
        "gpu_name": None,
        "gpu_vram_total_gb": 0.0,
        "torch_version": None,
    }

    if TORCH_AVAILABLE:
        info["torch_version"] = torch.__version__
        if torch.cuda.is_available():
            info["cuda_available"] = True
            info["gpu_count"] = torch.cuda.device_count()
            info["gpu_name"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            info["gpu_vram_total_gb"] = round(props.total_memory / (1024 ** 3), 2)

    return info


def get_optimal_device() -> str:
    """Return 'cuda' if GPU is available, else 'cpu'."""
    if TORCH_AVAILABLE and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def validate_weights_size(weights_dir: Path) -> Tuple[bool, int, str]:
    """
    Validate that total size of weight files does not exceed 5 GB (Rule 3).
    Returns: (is_valid, total_bytes, message)
    """
    if not weights_dir.exists():
        return True, 0, f"Weights directory {weights_dir} does not exist yet."

    total_bytes = 0
    file_count = 0
    weight_extensions = {".pt", ".pth", ".onnx", ".bin", ".weights", ".safetensors"}

    for p in weights_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in weight_extensions:
            size = p.stat().st_size
            total_bytes += size
            file_count += 1

    total_gb = total_bytes / (1024 ** 3)
    is_valid = total_bytes <= MAX_WEIGHTS_SIZE_BYTES

    msg = (
        f"Weights: {file_count} files, {total_gb:.3f} GB / "
        f"5.000 GB limit ({'PASS' if is_valid else 'FAIL - EXCEEDED'})"
    )
    return is_valid, total_bytes, msg


def validate_execution_time(video_duration_sec: float, elapsed_sec: float) -> Tuple[bool, str]:
    """
    Validate that execution time is <= video_duration * 3 (Rule 4).
    """
    max_allowed = video_duration_sec * MAX_TIME_FACTOR
    is_valid = elapsed_sec <= max_allowed
    msg = (
        f"Time check: Elapsed {elapsed_sec:.2f}s vs Max Allowed {max_allowed:.2f}s "
        f"({elapsed_sec / max(video_duration_sec, 0.001):.2f}x video duration) -> "
        f"{'PASS' if is_valid else 'FAIL - EXCEEDED'}"
    )
    return is_valid, msg


if __name__ == "__main__":
    print("=== Algorix Hardware Environment Check ===")
    info = get_hardware_info()
    for k, v in info.items():
        print(f"  {k}: {v}")
    print(f"  Selected inference device: {get_optimal_device()}")
    print("==========================================")
