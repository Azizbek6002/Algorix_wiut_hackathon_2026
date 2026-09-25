"""Format/validation tests for the official solution interface.

Checks the exact contract run_submission.py / evaluate.py assumes:
  * detect_events returns a list of [start, end, label],
  * labels are in CLASSES, start < end, start >= 0,
  * end <= duration (+0.5 tolerance),
  * no overlapping same-class segments,
  * RiskEstimator.reset/step return a bounded float and need no model/video,
  * detect_events on a missing file returns [] (no weights/GPU required).

Run:  python tests/test_solution_format.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np  # noqa: E402
from _validation import validate_events  # noqa: E402
from solution import CLASSES, RISK_HORIZON_SEC, RiskEstimator, detect_events  # noqa: E402


def test_class_id_surface():
    assert len(CLASSES) == 14
    assert len(set(CLASSES)) == len(CLASSES)
    # every official label exactly once, order is irrelevant
    assert set(CLASSES) == {
        "accident", "near_miss", "red_light", "wrong_way", "illegal_u_turn",
        "stopped_vehicle", "jaywalking", "failure_to_yield", "illegal_turn",
        "solid_line_crossing", "stop_line", "congestion", "road_obstacle",
        "fire_smoke",
    }


def test_valid_output_example():
    assert validate_events([], 30.0) == []
    ev = [[0.0, 5.0, "congestion"], [1.0, 2.0, "wrong_way"]]
    assert validate_events(ev, 30.0) == []
    # overlap in the same class is invalid
    assert validate_events([[0.0, 5.0, "congestion"], [4.0, 8.0, "congestion"]],
                           30.0)
    # cross-class overlaps are allowed
    assert validate_events([[0.0, 5.0, "congestion"], [4.0, 8.0, "wrong_way"]],
                           30.0) == []
    assert validate_events([[0.0, 5.0, "not_a_label"]], 30.0)
    assert validate_events([[5.0, 5.0, "congestion"]], 30.0)
    assert validate_events([[0.0, 31.0, "congestion"]], 30.0)


def test_detect_events_missing_video_returns_empty():
    # no model load, no GPU: the video cannot open -> []
    assert detect_events(os.path.join(os.path.dirname(__file__), "nope.mp4")) == []


def test_risk_estimator_interface():
    est = RiskEstimator()
    est.reset({"video_id": "t", "fps": 10.0, "width": 64, "height": 64,
               "n_frames": 10})
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    score = est.step(frame, 1.0)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0
    # strictly past-only: stepping again returns a bounded float too
    assert 0.0 <= est.step(frame, 2.0) <= 1.0
    assert RISK_HORIZON_SEC > 0


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"OK: {len(tests)} tests passed")


if __name__ == "__main__":
    main()