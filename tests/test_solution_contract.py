"""
Test contract compliance for solution.py according to Hackathon rules.
"""

import inspect
import numpy as np
import pytest
from solution import CLASSES, detect_events, RiskEstimator

EXPECTED_CLASSES = [
    "accident", "near_miss", "red_light", "wrong_way", "illegal_u_turn",
    "stopped_vehicle", "jaywalking", "failure_to_yield", "illegal_turn",
    "solid_line_crossing", "stop_line", "congestion", "road_obstacle", "fire_smoke",
]


def test_classes_contract():
    assert CLASSES == EXPECTED_CLASSES, "CLASSES list does not match exact specification"
    assert len(CLASSES) == 14, f"Expected 14 classes, found {len(CLASSES)}"


def test_detect_events_signature():
    assert callable(detect_events), "detect_events must be a callable function"
    sig = inspect.signature(detect_events)
    assert "video_path" in sig.parameters, "detect_events must accept 'video_path'"


def test_risk_estimator_contract():
    estimator = RiskEstimator()
    assert hasattr(estimator, "update"), "RiskEstimator must have an 'update' method"
    sig = inspect.signature(estimator.update)
    assert "frame" in sig.parameters, "update method must accept 'frame'"
    assert "timestamp" in sig.parameters, "update method must accept 'timestamp'"

    # Test dummy frame
    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    score = estimator.update(dummy_frame, 0.0)
    assert isinstance(score, float), "Risk score must be float"
    assert 0.0 <= score <= 1.0, "Risk score must be in [0.0, 1.0]"
