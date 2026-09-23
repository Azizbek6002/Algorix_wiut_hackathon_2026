"""
Unit tests for Modul 3 (Multi-Object Tracker / ByteTrack).
WIUT Hackathon 2026 — Algorix Team
"""

import pytest
from src.detection.detector import Detection
from src.tracking.tracker import ByteTracker, calculate_iou, Track


def test_calculate_iou():
    """Verify IoU calculation between boxes."""
    boxA = (0.0, 0.0, 10.0, 10.0)
    boxB = (0.0, 0.0, 10.0, 10.0)
    assert calculate_iou(boxA, boxB) == pytest.approx(1.0)

    # 50% overlap horizontally: inter=50, union=150 -> 1/3
    boxC = (5.0, 0.0, 15.0, 10.0)
    assert calculate_iou(boxA, boxC) == pytest.approx(50.0 / 150.0)

    # No overlap
    boxD = (20.0, 20.0, 30.0, 30.0)
    assert calculate_iou(boxA, boxD) == 0.0


def test_tracker_association_and_velocity():
    """Verify tracking association across frames and velocity computation."""
    tracker = ByteTracker()

    # Frame 1: Detection at (100, 100, 150, 150), timestamp 0.0
    det1 = Detection(bbox=(100.0, 100.0, 150.0, 150.0), class_name="vehicle", confidence=0.9, class_id=2)
    tracks_f1 = tracker.update([det1], timestamp=0.0)
    assert len(tracks_f1) == 1
    track_id = tracks_f1[0].track_id
    assert track_id == 1

    # Frame 2: Same object moved slightly to (110, 100, 160, 150) at timestamp 0.1s
    det2 = Detection(bbox=(110.0, 100.0, 160.0, 150.0), class_name="vehicle", confidence=0.88, class_id=2)
    tracks_f2 = tracker.update([det2], timestamp=0.1)
    assert len(tracks_f2) == 1
    assert tracks_f2[0].track_id == track_id  # Same track ID maintained!
    # Moved 10px in 0.1s -> ~100 px/s in x
    assert tracks_f2[0].speed > 10.0


def test_tracker_stopped_vehicle_duration():
    """Verify stopped_duration increases when vehicle does not move."""
    tracker = ByteTracker()
    box = (200.0, 200.0, 250.0, 250.0)

    # Stationary for 12 seconds
    for t in range(12):
        det = Detection(bbox=box, class_name="vehicle", confidence=0.95, class_id=2)
        tracks = tracker.update([det], timestamp=float(t))

    assert len(tracks) == 1
    assert tracks[0].stopped_duration >= 10.0  # Stopped > 10s condition met!


def test_tracker_low_confidence_matching():
    """Verify ByteTrack associates low-confidence detection with existing track."""
    tracker = ByteTracker(high_score_thresh=0.6, low_score_thresh=0.2)

    # Frame 1: High confidence detection creates track
    det_high = Detection(bbox=(300.0, 300.0, 350.0, 350.0), class_name="vehicle", confidence=0.85, class_id=2)
    t1 = tracker.update([det_high], timestamp=0.0)
    tid = t1[0].track_id

    # Frame 2: Heavy occlusion / low confidence (0.3), but same location
    det_low = Detection(bbox=(305.0, 302.0, 355.0, 352.0), class_name="vehicle", confidence=0.35, class_id=2)
    t2 = tracker.update([det_low], timestamp=0.05)
    assert len(t2) == 1
    assert t2[0].track_id == tid  # Matched in stage 2!
