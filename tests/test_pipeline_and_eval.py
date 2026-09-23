"""
End-to-End Pipeline & Comprehensive System Integration Tests.
WIUT Hackathon 2026 — Algorix Team
"""

import re
from pathlib import Path
import numpy as np
import pytest

from solution import CLASSES, detect_events, RiskEstimator
from evaluate import compute_temporal_iou, evaluate_part_a, validate_prediction_format
from eda.video_analyzer import create_synthetic_test_video


def test_no_forbidden_network_calls():
    """Verify Rule 1: No network/API calls in src/ or solution.py."""
    root_dir = Path(__file__).resolve().parent.parent
    forbidden_patterns = [
        r"requests\.",
        r"urllib\.request",
        r"openai",
        r"anthropic",
        r"generativelanguage",
    ]

    files_to_check = list((root_dir / "src").rglob("*.py")) + [root_dir / "solution.py"]

    for py_file in files_to_check:
        content = py_file.read_text(encoding="utf-8")
        for pat in forbidden_patterns:
            matches = re.findall(pat, content)
            assert not matches, f"Forbidden network call pattern '{pat}' found in {py_file.name}!"


def test_temporal_iou_calculation():
    """Verify 1D Temporal IoU calculation."""
    # Complete match
    assert compute_temporal_iou((10.0, 20.0), (10.0, 20.0)) == pytest.approx(1.0)
    # Disjoint
    assert compute_temporal_iou((0.0, 5.0), (10.0, 15.0)) == 0.0
    # Overlap: [10, 20] and [15, 25] -> inter=5, union=15 -> 1/3
    assert compute_temporal_iou((10.0, 20.0), (15.0, 25.0)) == pytest.approx(5.0 / 15.0)


def test_evaluate_part_a_metrics():
    """Verify Part A evaluation metric calculation."""
    gt = [
        [10.0, 20.0, "accident"],
        [30.0, 40.0, "wrong_way"],
    ]
    # Exact predictions
    preds_perfect = [
        [10.0, 20.0, "accident"],
        [30.0, 40.0, "wrong_way"],
    ]
    results = evaluate_part_a(preds_perfect, gt, iou_thresholds=(0.3, 0.5, 0.7))
    assert results["mean_f1"] == 1.0

    # Predictions with partial overlap
    preds_shifted = [
        [12.0, 22.0, "accident"],  # IoU with [10, 20] is 8/12 = 0.667 -> passes 0.3 and 0.5, fails 0.7
    ]
    res_shifted = evaluate_part_a(preds_shifted, gt, iou_thresholds=(0.3, 0.5, 0.7))
    assert res_shifted["f1@0.3"] > 0.0
    assert res_shifted["f1@0.5"] > 0.0
    assert res_shifted["f1@0.7"] == 0.0


def test_validate_prediction_format():
    """Verify prediction formatting validator."""
    valid_data = [[1.5, 3.0, "accident"], [10.0, 15.0, "red_light"]]
    valid, msg = validate_prediction_format(valid_data)
    assert valid is True

    # Bad class
    invalid_data = [[1.5, 3.0, "unknown_class"]]
    valid, msg = validate_prediction_format(invalid_data)
    assert valid is False
    assert "unknown event class" in msg


def test_end_to_end_synthetic_execution(tmp_path: Path):
    """Run full end-to-end execution on a synthetic video."""
    video_path = tmp_path / "e2e_traffic_test.mp4"
    create_synthetic_test_video(video_path, duration_sec=3, fps=30)

    # 1. detect_events
    events = detect_events(str(video_path))
    assert isinstance(events, list)

    # 2. RiskEstimator online streaming simulation
    estimator = RiskEstimator()
    dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    risk_timeline = []
    for t in [0.0, 0.5, 1.0, 1.5, 2.0]:
        score = estimator.update(dummy_frame, t)
        assert 0.0 <= score <= 1.0
        risk_timeline.append(score)

    assert len(risk_timeline) == 5
