"""
Unit tests for Modul 2 (Object Detection, Class Remapping, and Speed Profiling).
WIUT Hackathon 2026 — Algorix Team
"""

from pathlib import Path
import numpy as np
import pytest

from src.detection.detector import (
    Detection,
    ObjectDetector,
    remap_coco_class,
    COCO_TO_TARGET_MAP,
)
from src.utils.hardware import validate_weights_size


def test_detection_dataclass():
    """Verify Detection dataclass geometric calculations and properties."""
    det = Detection(
        bbox=(100.0, 200.0, 300.0, 500.0),  # w=200, h=300
        class_name="vehicle",
        confidence=0.89,
        class_id=2,
    )
    # Center: ((100+300)/2, (200+500)/2) = (200, 350)
    assert det.center == (200.0, 350.0)
    # Bottom center: ((100+300)/2, 500) = (200, 500)
    assert det.bottom_center == (200.0, 500.0)
    # Area: 200 * 300 = 60000
    assert det.area == 60000.0

    d = det.to_dict()
    assert d["class_name"] == "vehicle"
    assert d["confidence"] == 0.89
    assert d["bbox"] == [100.0, 200.0, 300.0, 500.0]


def test_coco_class_remapping():
    """Verify COCO class ID conversion to competition target classes."""
    assert remap_coco_class(0) == "pedestrian"  # person
    assert remap_coco_class(2) == "vehicle"     # car
    assert remap_coco_class(3) == "vehicle"     # motorcycle
    assert remap_coco_class(5) == "vehicle"     # bus
    assert remap_coco_class(7) == "vehicle"     # truck
    assert remap_coco_class(99) is None         # non-target class


def test_detector_initialization_and_stride():
    """Verify detector parameters, frame stride, and device configuration."""
    detector = ObjectDetector(
        conf_threshold=0.35,
        iou_threshold=0.50,
        frame_stride=3,
    )
    assert detector.conf_threshold == 0.35
    assert detector.iou_threshold == 0.50
    assert detector.frame_stride == 3
    assert detector.device in ("cpu", "cuda")


def test_detector_profile_speed():
    """Verify profiling latency and effective FPS with frame stride."""
    detector = ObjectDetector(frame_stride=2)
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    profile = detector.profile_inference_speed(
        sample_frame=dummy_frame,
        num_warmup=1,
        num_runs=5,
    )
    assert "avg_latency_ms" in profile
    assert "raw_fps" in profile
    assert "effective_fps_with_stride" in profile
    assert profile["frame_stride"] == 2
    assert profile["effective_fps_with_stride"] >= profile["raw_fps"]


def test_weights_dir_size_limit():
    """Verify weights directory is within 5GB limit."""
    weights_dir = Path(__file__).resolve().parent.parent / "weights"
    is_valid, total_bytes, msg = validate_weights_size(weights_dir)
    assert is_valid is True
    assert total_bytes <= 5 * 1024 * 1024 * 1024
