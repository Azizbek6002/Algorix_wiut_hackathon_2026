"""Deterministic unit tests for src/geometry.py + scene_config.json.

Run:  python tests/test_geometry.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.geometry import Geometry, point_in_polygon, segments_intersect


def _cfg():
    return {
        "provenance": {"reference_resolution": [3840, 2160]},
        "road_polygon": {
            "enabled": True,
            "points": [[0, 0], [100, 0], [100, 100], [0, 100]],
        },
        "lanes": [
            {"lane_id": "L1", "enabled": True,
             "polygon": [[10, 10], [40, 10], [40, 40], [10, 40]],
             "expected_direction": 90.0},
            {"lane_id": "L2", "enabled": False,   # disabled lanes must be ignored
             "polygon": [[50, 50], [70, 50], [70, 70], [50, 70]],
             "expected_direction": 180.0},
        ],
        "crosswalks": [{"enabled": True,
                        "polygon": [[20, 20], [30, 20], [30, 30], [20, 30]]}],
        "intersection_zones": [{"enabled": True,
                                "polygon": [[40, 40], [60, 40], [60, 60], [40, 60]]}],
        "u_turn_zones": [{"enabled": True,
                          "polygon": [[80, 80], [90, 80], [90, 90], [80, 90]]}],
        "stop_lines": [{"enabled": True, "line": [[0, 50], [100, 50]]}],
        "solid_lines": [{"enabled": True, "line": [[25, 0], [25, 100]]}],
        "traffic_light_rois": [{"enabled": True, "line": [[0, 0], [100, 100]]}],
        "exclusion_regions": [{"enabled": True,
                               "polygon": [[90, 0], [100, 0], [100, 10], [90, 10]]}],
    }


def test_point_in_polygon_cases():
    sq = [[0, 0], [10, 0], [10, 10], [0, 10]]
    assert point_in_polygon((5, 5), sq) is True
    assert point_in_polygon((15, 5), sq) is False
    assert point_in_polygon((0, 5), sq) is True    # edge, inclusive
    assert point_in_polygon((10, 10), sq) is True  # vertex, inclusive
    assert point_in_polygon((5, -0.5), sq) is False
    assert point_in_polygon((5, 5), []) is False


def test_segment_intersection_cases():
    # proper crossing
    assert segments_intersect((25, 0), (25, 100), (0, 50), (100, 50)) is True
    # parallel, no touch
    assert segments_intersect((25, 0), (25, 50), (50, 0), (50, 50)) is False
    # endpoint touch counts
    assert segments_intersect((0, 50), (10, 50), (10, 40), (10, 60)) is True
    # collinear overlap
    assert segments_intersect((0, 50), (30, 50), (20, 50), (40, 50)) is True
    # disjoint
    assert segments_intersect((0, 0), (10, 0), (20, 20), (30, 20)) is False


def test_lane_membership_and_direction():
    g = Geometry(_cfg())
    assert g.get_lane((20, 20)) == "L1"
    assert g.get_lane((5, 5)) is None          # on road but no lane
    assert g.get_lane((70, 70)) is None        # inside a DISABLED lane -> ignored
    assert g.get_lane_direction("L1") == 90.0
    assert g.get_lane_direction("L2") is None  # disabled -> unknown
    assert g.get_lane_direction("NOPE") is None


def test_zone_memberships():
    g = Geometry(_cfg())
    assert g.is_on_road((50, 50)) is True
    assert g.is_on_road((150, 50)) is False
    assert g.is_on_road((0, 50)) is True   # boundary inclusive
    assert g.is_in_crosswalk((25, 25)) is True
    assert g.is_in_crosswalk((5, 5)) is False
    assert g.is_in_intersection((50, 50)) is True
    assert g.is_in_intersection((25, 25)) is False
    assert g.is_in_u_turn_zone((85, 85)) is True
    assert g.is_in_u_turn_zone((50, 50)) is False
    assert g.is_in_exclusion((95, 5)) is True
    assert g.is_in_exclusion((92, 5)) is True
    assert g.is_in_exclusion((89, 5)) is False


def test_crossings():
    g = Geometry(_cfg())
    # stop line at y=50, solid line at x=25
    assert g.crosses_stop_line((20, 40), (20, 60)) is True
    assert g.crosses_stop_line((20, 40), (20, 49)) is False
    assert g.crosses_stop_line((20, 40), (21, 49)) is False
    assert g.crosses_stop_line((10, 50), (10, 60)) is True  # starts on the line
    assert g.crosses_solid_line((10, 50), (30, 50)) is True   # crosses x=25
    assert g.crosses_solid_line((40, 50), (50, 50)) is False  # stays right of x=25
    assert g.crosses_solid_line((25, 50), (30, 50)) is True   # starts on the line
    assert g.crosses_solid_line((20, 10), (30, 10)) is True   # crosses even while driving along
    assert g.crosses_solid_line((5, 10), (20, 10)) is False   # never reaches the line


def test_scale_boundary_case():
    # 16:9 frame at smaller size; scaling is per-axis and must map exactly back
    # to the reference space (ref 3840x2160).
    g = Geometry(_cfg(), frame_w=1000, frame_h=562)
    fx = 50 * 1000 / 3840.0
    fy = 50 * 562 / 2160.0
    rx, ry = g.to_ref((fx, fy))
    assert abs(rx - 50.0) < 1e-6 and abs(ry - 50.0) < 1e-6
    assert g.is_on_road((fx, fy)) is True
    assert g.is_in_intersection((fx, fy)) is True   # ref (50,50) in [40,60]^2


def test_coordinate_consistency():
    """Same world point must answer identically at any frame resolution."""
    g1 = Geometry(_cfg())                        # scale 1
    g2 = Geometry(_cfg(), frame_w=1920, frame_h=1080)  # 0.5 scale

    for ref_pt in [(50, 50), (25, 25), (80, 90), (150, 50)]:
        fr_pt = (ref_pt[0] * 0.5, ref_pt[1] * 0.5)
        assert g1.is_on_road(ref_pt) == g2.is_on_road(fr_pt)
        assert g1.get_lane(ref_pt) == g2.get_lane(fr_pt)
        assert g1.is_in_crosswalk(ref_pt) == g2.is_in_crosswalk(fr_pt)
        assert g1.is_in_intersection(ref_pt) == g2.is_in_intersection(fr_pt)
        assert g1.is_in_u_turn_zone(ref_pt) == g2.is_in_u_turn_zone(fr_pt)

    # crossing consistency: ref (25,45)->(25,55) crosses stop line y=50
    ref_a, ref_b = (25, 45), (25, 55)
    fr_a = (ref_a[0] * 0.5, ref_a[1] * 0.5)
    fr_b = (ref_b[0] * 0.5, ref_b[1] * 0.5)
    assert g1.crosses_stop_line(ref_a, ref_b) is True
    assert g1.crosses_stop_line(ref_a, ref_b) == g2.crosses_stop_line(fr_a, fr_b)
    assert g2.crosses_solid_line((13, 5), (13, 45)) == g1.crosses_solid_line((26, 10), (26, 90))


def test_traffic_light_unknown():
    g = Geometry(_cfg())
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    assert g.get_traffic_light_state(frame, [0, 0, 5, 5]) == "UNKNOWN"


def test_shared_camera_level_geometry():
    """One camera-level scene_config drives EVERY video identically.

    There is no per-video override: a leftover 'videos' key must be ignored
    and the same shared features must answer the same queries for any video
    (sample or hidden) — no separate config per video is ever required.
    """
    shared = _cfg()
    assert "videos" not in shared
    g1 = Geometry(shared)

    # even if some unrelated 'videos' data existed, geometry must not branch on it
    with_videos = _cfg()
    with_videos["videos"] = {"CAM_A.MP4": {"lanes": []}}
    g2 = Geometry(with_videos)

    for pt in [(20, 20), (25, 25), (50, 50), (95, 5), (85, 85)]:
        assert (g1.get_lane(pt) == g2.get_lane(pt)
                and g1.is_in_crosswalk(pt) == g2.is_in_crosswalk(pt)
                and g1.is_on_road(pt) == g2.is_on_road(pt)
                and g1.is_in_exclusion(pt) == g2.is_in_exclusion(pt)
                and g1.is_in_u_turn_zone(pt) == g2.is_in_u_turn_zone(pt))
    assert g1.get_lane((20, 20)) == "L1"

    # same config in any actual frame resolution (hidden test videos included)
    g3 = Geometry(shared, frame_w=1920, frame_h=1080)
    assert g3.get_lane((10, 10)) == g1.get_lane((20, 20))  # ref (20,20) * 0.5
    assert g3.is_in_crosswalk((12.5, 12.5)) == g1.is_in_crosswalk((25, 25))


def centroid(poly):
    return (sum(p[0] for p in poly) / len(poly), sum(p[1] for p in poly) / len(poly))


def interior(poly):
    """Guaranteed-interior point for a simple polygon: barycenter of the
    largest fan triangle (p0, pi, pi+1). Falls back to vertex mean otherwise."""
    n = len(poly)
    best, best_area = None, -1.0
    for i in range(1, n - 1):
        ax, ay = poly[0]; bx, by = poly[i]; cx2, cy2 = poly[i + 1]
        area = abs((bx - ax) * (cy2 - ay) - (by - ay) * (cx2 - ax))
        if area > best_area:
            best_area = area
            best = ((ax + bx + cx2) / 3.0, (ay + by + cy2) / 3.0)
    return best or centroid(poly)


def test_real_scene_config():
    """Camera-level shared config: structural sanity checks derived from the
    calibrated geometry itself (no visual knowledge required)."""
    pkg = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg_path = os.path.join(pkg, "scene_config.json")
    assert os.path.exists(cfg_path)
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    assert "videos" not in cfg          # no per-video configs, camera-level only

    ref = cfg["provenance"]["reference_resolution"]
    assert ref == [3840, 2160]
    g = Geometry.from_json(cfg_path, frame_w=3840, frame_h=2160)

    # road polygon: complete polygon, no self-destroyed shape
    road = cfg["road_polygon"]
    assert road["enabled"] and road["confidence"] == "calibrated"
    assert len(road["points"]) >= 4
    rpts = road["points"]
    for v in rpts:
        assert g.is_on_road(v) is True        # boundary inclusive -> vertices on road

    # lanes: parse + centroid inside its own lane + valid direction
    assert len(cfg["lanes"]) >= 1
    seen_lanes = set()
    for l in cfg["lanes"]:
        assert l["enabled"] and isinstance(l["expected_direction"], (int, float))
        assert 0.0 <= l["expected_direction"] < 360.0
        lid = l["lane_id"]
        assert lid and lid not in seen_lanes
        seen_lanes.add(lid)
        poly = l["polygon"]
        assert len(poly) >= 3
        assert g.get_lane(interior(poly)) == lid

    # crosswalks parse and their interiors are detected as in-crosswalk
    assert len(cfg["crosswalks"]) >= 1
    for cw in cfg["crosswalks"]:
        assert cw["enabled"] and len(cw["polygon"]) >= 3
        assert g.is_in_crosswalk(interior(cw["polygon"])) is True

    # zones parse; interiors land inside their own zone
    for key, zone_key in [("intersection_zones", "is_in_intersection"),
                          ("u_turn_zones", "is_in_u_turn_zone"),
                          ("exclusion_regions", "is_in_exclusion")]:
        for z in cfg.get(key, []):
            assert z["enabled"] and len(z["polygon"]) >= 3
            assert getattr(g, zone_key)(interior(z["polygon"])) is True

    # lines: 2 distinct endpoints
    for key in ("stop_lines", "solid_lines"):
        for ln in cfg.get(key, []):
            a, b = ln["line"]
            assert a != b

    # traffic-light ROIs parse; state is still never faked
    assert len(cfg["traffic_light_rois"]) >= 1
    assert g.get_traffic_light_state(None, [0, 0, 1, 1]) == "UNKNOWN"

    # scale-consistency with the real config: L0 centroid at half resolution
    l0 = next(l for l in cfg["lanes"] if l["lane_id"] == "L0")
    cx, cy = centroid(l0["polygon"])
    g2 = Geometry.from_json(cfg_path, frame_w=1920, frame_h=1080)
    assert g2.get_lane((cx * 0.5, cy * 0.5)) == g.get_lane((cx, cy)) == "L0"
    assert g2.is_in_crosswalk((cx * 0.5, cy * 0.5)) == g.is_in_crosswalk((cx, cy))


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"OK: {len(tests)} tests passed")


if __name__ == "__main__":
    main()