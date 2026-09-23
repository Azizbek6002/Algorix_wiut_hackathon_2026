"""
Unit tests for Modul 4 (Scene Geometry Integration & SceneState).
WIUT Hackathon 2026 — Algorix Team
"""

from src.geometry.camera_parser import load_scene_geometry
from src.geometry.scene_state import SceneState, TrackSceneContext
from src.tracking.tracker import Track


def test_track_scene_context_mapping():
    """Verify track mapping to lane and geometry properties."""
    scene = load_scene_geometry()

    # Track in lane_south_1 at (1100, 600)
    trk = Track(
        track_id=1,
        class_name="vehicle",
        bbox=(1050.0, 500.0, 1150.0, 700.0),
        confidence=0.9,
        first_timestamp=0.0,
        last_timestamp=1.0,
        velocity=(0.0, 50.0),  # moving in direction [0, 1]
    )
    trk.trajectory = [(0.0, (1100.0, 550.0)), (1.0, (1100.0, 600.0))]

    ctx = TrackSceneContext(trk, scene)
    assert ctx.is_on_road is True
    assert ctx.lane_id is not None
    assert ctx.is_wrong_way is False


def test_scene_state_congestion():
    """Verify scene congestion metric calculation."""
    state = SceneState()

    # Create 6 slow-moving / stopped vehicles
    slow_tracks = [
        Track(
            track_id=i,
            class_name="vehicle",
            bbox=(200.0 + i * 50, 400.0, 240.0 + i * 50, 450.0),
            confidence=0.9,
            first_timestamp=0.0,
            last_timestamp=1.0,
            velocity=(0.0, 5.0),  # speed 5 px/s < 15
        )
        for i in range(6)
    ]

    state.update(slow_tracks)
    assert state.vehicle_count == 6
    assert state.mean_vehicle_speed < 15.0
    assert state.is_congested() is True
