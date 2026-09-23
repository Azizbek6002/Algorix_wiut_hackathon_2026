"""
Unit tests for Modul 0 (Environment, Determinism, Hardware constraints).
"""

import random
from pathlib import Path
import numpy as np
import pytest

from src.utils.determinism import set_global_seed
from src.utils.hardware import (
    get_hardware_info,
    get_optimal_device,
    validate_weights_size,
    validate_execution_time,
)


def test_determinism_python_and_numpy():
    """Verify that set_global_seed ensures deterministic random outputs."""
    set_global_seed(42)
    py_r1 = [random.random() for _ in range(5)]
    np_r1 = np.random.rand(5).tolist()

    # Reset seed and re-sample
    set_global_seed(42)
    py_r2 = [random.random() for _ in range(5)]
    np_r2 = np.random.rand(5).tolist()

    assert py_r1 == py_r2, "Python random was not deterministic"
    assert np_r1 == np_r2, "NumPy random was not deterministic"


def test_hardware_inspection():
    """Verify system hardware information extraction."""
    info = get_hardware_info()
    assert "python_version" in info
    assert "cpu_count_logical" in info
    assert "ram_total_gb" in info
    assert info["cpu_count_logical"] >= 1
    assert info["ram_total_gb"] > 0
    assert get_optimal_device() in ("cuda", "cpu")


def test_validate_weights_size_pass(tmp_path: Path):
    """Test weight validation passes within 5GB limit."""
    weights_dir = tmp_path / "weights"
    weights_dir.mkdir()
    fake_weight = weights_dir / "detector.pt"
    fake_weight.write_bytes(b"0" * 1024)  # 1 KB

    is_valid, total_bytes, msg = validate_weights_size(weights_dir)
    assert is_valid is True
    assert total_bytes == 1024
    assert "PASS" in msg


def test_validate_execution_time():
    """Test execution time factor (video_duration * 3)."""
    video_duration = 60.0  # 60s video -> max allowed 180s

    # 100s execution -> PASS
    is_valid, msg = validate_execution_time(video_duration, 100.0)
    assert is_valid is True
    assert "PASS" in msg

    # 200s execution -> FAIL
    is_valid, msg = validate_execution_time(video_duration, 200.0)
    assert is_valid is False
    assert "FAIL" in msg
