"""
Unit tests for Modul 1 (Dataset, EDA, Scene Geometry & Labeling Tool).
WIUT Hackathon 2026 — Algorix Team
"""

from pathlib import Path
import pytest
from src.geometry.camera_parser import (
    load_scene_geometry,
    point_in_polygon,
    segments_intersect,
    direction_cosine,
)
from labeling.label_tool import DatasetLabelManager, CLASSES
from eda.video_analyzer import VideoAnalyzer, classify_lighting, create_synthetic_test_video


def test_load_scene_geometry_json():
    """Verify loading scene geometry from default_scene.json."""
    scene = load_scene_geometry()
    assert scene.resolution == (1920, 1080)
    assert scene.fps == 30.0
    assert len(scene.lanes) == 3
    assert len(scene.stop_lines) >= 1
    assert len(scene.crossings) >= 1
    assert "carriageway" in scene.road_polygons


def test_load_scene_geometry_markdown():
    """Verify parsing markdown camera.md configuration."""
    md_path = Path(__file__).resolve().parent.parent / "camera_configs" / "camera.md"
    scene = load_scene_geometry(md_path)
    assert scene.resolution == (1920, 1080)
    assert len(scene.lanes) >= 1


def test_point_in_polygon_and_crosswalk():
    """Test ray-casting point-in-polygon logic."""
    poly = [[0, 0], [10, 0], [10, 10], [0, 10]]
    assert point_in_polygon((5, 5), poly) is True
    assert point_in_polygon((15, 15), poly) is False

    scene = load_scene_geometry()
    # Point inside crosswalk_south: y is between 720 and 800, x around 1000
    assert scene.is_in_crosswalk((1000, 750)) is True
    assert scene.is_in_crosswalk((100, 100)) is False


def test_wrong_way_direction():
    """Verify direction cosine similarity and wrong way detection."""
    # Same direction: [0, 1] vs [0, 1] -> cos = 1.0
    assert direction_cosine((0.0, 1.0), (0.0, 1.0)) == pytest.approx(1.0)
    # Opposite direction: [0, -1] vs [0, 1] -> cos = -1.0
    assert direction_cosine((0.0, -1.0), (0.0, 1.0)) == pytest.approx(-1.0)

    scene = load_scene_geometry()
    # In lane_south_1, expected is [0, 1]. Moving [0, -1] is wrong_way
    lane_point = (1200, 600)  # inside lane_south_1
    assert scene.is_wrong_way(lane_point, (0.0, -1.0)) is True
    assert scene.is_wrong_way(lane_point, (0.0, 1.0)) is False


def test_line_intersections():
    """Verify segment intersection check for stop lines and solid lines."""
    scene = load_scene_geometry()
    # stop_line_main is from [760, 680] to [1450, 680]
    # Moving from (1000, 600) to (1000, 750) crosses the stop line
    assert scene.crossed_stop_line((1000, 600), (1000, 750)) is True
    # Moving entirely below does not cross
    assert scene.crossed_stop_line((1000, 700), (1000, 750)) is False

    # solid_divider_center is from [760, 420] to [720, 1080]
    # Moving horizontally from (700, 700) to (800, 700) crosses it
    assert scene.crossed_solid_line((700, 700), (800, 700)) is True
    assert scene.crossed_solid_line((800, 700), (900, 700)) is False


def test_label_manager_crud_and_validation(tmp_path: Path):
    """Verify GT dataset creation, validation, and JSON/CSV export."""
    manager = DatasetLabelManager(labels_dir=tmp_path / "labels")

    # Valid event
    events = [
        {"start_time": 5.0, "end_time": 10.0, "event_type": "accident", "notes": "Two car collision"},
        {"start_time": 15.2, "end_time": 20.0, "event_type": "wrong_way"},
    ]
    gt_file = manager.create_gt("test_video.mp4", events)
    assert gt_file.exists()
    assert (tmp_path / "labels" / "test_video_gt.csv").exists()

    loaded = manager.load_gt("test_video.mp4")
    assert len(loaded) == 2
    assert loaded[0] == [5.0, 10.0, "accident"]
    assert loaded[1] == [15.2, 20.0, "wrong_way"]

    # Invalid class name should raise ValueError
    with pytest.raises(ValueError, match="Invalid event_type"):
        manager.create_gt("invalid.mp4", [{"start_time": 1.0, "end_time": 2.0, "event_type": "non_existent_class"}])

    # End time before start time should raise ValueError
    with pytest.raises(ValueError, match="end_time .* must be strictly greater"):
        manager.create_gt("invalid.mp4", [{"start_time": 10.0, "end_time": 5.0, "event_type": "accident"}])


def test_video_analyzer_synthetic_and_lighting(tmp_path: Path):
    """Create a synthetic test video and verify analyzer profiling."""
    assert classify_lighting(220, 30) == "overexposed"
    assert classify_lighting(30, 20) == "night"
    assert classify_lighting(75, 40) == "dusk_dawn"
    assert classify_lighting(130, 45) == "daylight"

    video_path = tmp_path / "synthetic_test.mp4"
    create_synthetic_test_video(video_path, duration_sec=2, fps=30)
    assert video_path.exists()

    analyzer = VideoAnalyzer(sample_interval_frames=10)
    meta = analyzer.analyze_video(str(video_path))
    assert meta["width"] == 1920
    assert meta["height"] == 1080
    assert meta["fps"] == 30.0
    assert meta["duration_sec"] >= 1.9
    assert "lighting" in meta
    assert "motion_density" in meta
