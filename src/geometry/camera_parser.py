"""
Camera Scene Geometry Parser & Spatial Reasoning Engine.
WIUT Hackathon 2026 — Algorix Team

Parses camera.md / default_scene.json to provide:
- Point-in-polygon queries (pure NumPy/Python ray-casting, no heavy dependencies)
- Line segment intersection (for stop-lines and solid boundary crossing)
- Direction alignment check (cosine similarity for wrong-way detection)
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import numpy as np


def point_in_polygon(point: Tuple[float, float], polygon: List[List[float]]) -> bool:
    """
    Ray-casting algorithm to determine if a point (x, y) is inside a polygon.
    Polygon is a list of [x, y] coordinates.
    """
    x, y = point
    inside = False
    n = len(polygon)
    if n < 3:
        return False

    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside


def segments_intersect(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    q1: Tuple[float, float],
    q2: Tuple[float, float],
) -> bool:
    """
    Check if line segment p1-p2 intersects with line segment q1-q2.
    """
    def ccw(a, b, c):
        return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])

    return (ccw(p1, q1, q2) != ccw(p2, q1, q2)) and (ccw(p1, p2, q1) != ccw(p1, p2, q2))


def direction_cosine(v1: Tuple[float, float], v2: Tuple[float, float]) -> float:
    """
    Calculate cosine similarity between two 2D vectors: dot(v1, v2) / (|v1| * |v2|).
    Returns 1.0 (same direction), 0.0 (perpendicular), -1.0 (opposite/wrong way).
    """
    norm1 = np.hypot(v1[0], v1[1])
    norm2 = np.hypot(v2[0], v2[1])
    if norm1 < 1e-6 or norm2 < 1e-6:
        return 0.0
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    return float(np.clip(dot / (norm1 * norm2), -1.0, 1.0))


class SceneGeometry:
    """Represents full calibrated scene geometry for the fixed CCTV camera."""

    def __init__(self, data: Dict[str, Any]):
        self.resolution = tuple(data.get("resolution", [1920, 1080]))
        self.fps = float(data.get("nominal_fps", 30.0))
        self.road_polygons = data.get("road_polygons", {})
        self.lanes = data.get("lanes", [])
        self.solid_lines = data.get("solid_lines", [])
        self.stop_lines = data.get("stop_lines", [])
        self.crossings = data.get("crossings", [])
        self.traffic_light = data.get("traffic_light", {})

    def is_on_road(self, point: Tuple[float, float]) -> bool:
        """Check if point is inside any road / carriageway polygon."""
        for poly in self.road_polygons.values():
            if point_in_polygon(point, poly):
                return True
        return False

    def is_in_crosswalk(self, point: Tuple[float, float]) -> bool:
        """Check if point is inside pedestrian zebra crossing."""
        for c in self.crossings:
            if point_in_polygon(point, c["polygon"]):
                return True
        return False

    def get_lane_for_point(self, point: Tuple[float, float]) -> Optional[Dict[str, Any]]:
        """Find the lane containing the point, or None."""
        for lane in self.lanes:
            if point_in_polygon(point, lane["polygon"]):
                return lane
        return None

    def is_wrong_way(self, point: Tuple[float, float], motion_vector: Tuple[float, float]) -> bool:
        """
        Check if moving in opposite direction of current lane's expected direction.
        cos < -0.4 indicates wrong way motion.
        """
        lane = self.get_lane_for_point(point)
        if not lane:
            return False
        exp_dir = lane.get("expected_direction", [0.0, 0.0])
        cos_sim = direction_cosine(motion_vector, tuple(exp_dir))
        return cos_sim < -0.4

    def crossed_stop_line(self, prev_point: Tuple[float, float], curr_point: Tuple[float, float]) -> bool:
        """Check if object trajectory crossed any stop-line."""
        for sl in self.stop_lines:
            seg = sl["segment"]
            q1, q2 = (seg[0][0], seg[0][1]), (seg[1][0], seg[1][1])
            if segments_intersect(prev_point, curr_point, q1, q2):
                return True
        return False

    def crossed_solid_line(self, prev_point: Tuple[float, float], curr_point: Tuple[float, float]) -> bool:
        """Check if object trajectory crossed any solid lane divider line."""
        for sol in self.solid_lines:
            seg = sol["segment"]
            q1, q2 = (seg[0][0], seg[0][1]), (seg[1][0], seg[1][1])
            if segments_intersect(prev_point, curr_point, q1, q2):
                return True
        return False


def load_scene_geometry(config_path: Optional[Path] = None) -> SceneGeometry:
    """
    Load scene geometry from JSON or Markdown.
    If no path provided, searches default locations in camera_configs/.
    """
    if config_path is None:
        base_dir = Path(__file__).resolve().parent.parent.parent / "camera_configs"
        json_path = base_dir / "default_scene.json"
        md_path = base_dir / "camera.md"
        if json_path.exists():
            config_path = json_path
        elif md_path.exists():
            config_path = md_path
        else:
            raise FileNotFoundError("No scene geometry config found in camera_configs/")

    config_path = Path(config_path)
    if config_path.suffix == ".json":
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return SceneGeometry(data)

    # Parse markdown file by extracting json blocks
    with open(config_path, "r", encoding="utf-8") as f:
        content = f.read()

    data: Dict[str, Any] = {
        "resolution": [1920, 1080],
        "nominal_fps": 30.0,
        "road_polygons": {},
        "lanes": [],
        "solid_lines": [],
        "stop_lines": [],
        "crossings": [],
        "traffic_light": {},
    }

    # Extract JSON code blocks
    code_blocks = re.findall(r"```json(.*?)```", content, re.DOTALL)
    for block in code_blocks:
        try:
            parsed = json.loads(block.strip())
            if isinstance(parsed, dict):
                if "carriageway" in parsed:
                    data["road_polygons"]["carriageway"] = parsed["carriageway"]
                elif "has_traffic_light" in parsed:
                    data["traffic_light"] = parsed
            elif isinstance(parsed, list) and len(parsed) > 0:
                first = parsed[0]
                if "expected_direction" in first:
                    data["lanes"] = parsed
                elif "linked_lanes" in first:
                    data["stop_lines"] = parsed
                elif "segment" in first and "id" in first and "solid" in first["id"]:
                    data["solid_lines"] = parsed
                elif "polygon" in first and "crosswalk" in first.get("id", ""):
                    data["crossings"] = parsed
        except Exception:
            continue

    return SceneGeometry(data)
