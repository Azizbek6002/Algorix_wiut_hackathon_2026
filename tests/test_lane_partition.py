"""Regression tests: L0/L3 lane partitioning (mutual exclusivity + flow fit).

Phase 10A: the two lane polygons must be disjoint (no position can be in both)
and each must contain the traffic of its own flow corridor. Points below are
taken from the observed 335-flow / 164-flow corridors of the real scene.

Run:  python tests/test_lane_partition.py
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.geometry import Geometry, point_in_polygon

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG_PATH = os.path.join(PKG, "scene_config.json")

# bottom-center points inside the observed L0-flow and L3-flow corridors
L0_POINTS = [(400, 560), (600, 620), (740, 560), (900, 700), (1100, 750)]
L3_POINTS = [(500, 300), (940, 460), (1200, 560), (1400, 620), (2540, 980)]


def _polygons(cfg):
    lanes = {l["lane_id"]: l["polygon"] for l in cfg["lanes"]}
    return lanes["L0"], lanes["L3"]


def test_lanes_disjoint_on_real_config():
    cfg = json.load(open(CFG_PATH, encoding="utf-8"))
    p0, p3 = _polygons(cfg)
    # coarse grid over the full reference frame: nothing may be in both lanes
    for y in range(0, 2160, 96):
        for x in range(0, 3840, 96):
            a = point_in_polygon((x, y), p0)
            b = point_in_polygon((x, y), p3)
            assert not (a and b), f"overlap at {x},{y}"


def test_point_inside_l0_not_l3():
    cfg = json.load(open(CFG_PATH, encoding="utf-8"))
    p0, p3 = _polygons(cfg)
    for pt in L0_POINTS:
        assert point_in_polygon(pt, p0), f"{pt} should be in L0"
        assert not point_in_polygon(pt, p3), f"{pt} must not be in L3"


def test_point_inside_l3_not_l0():
    cfg = json.load(open(CFG_PATH, encoding="utf-8"))
    p0, p3 = _polygons(cfg)
    for pt in L3_POINTS:
        assert point_in_polygon(pt, p3), f"{pt} should be in L3"
        assert not point_in_polygon(pt, p0), f"{pt} must not be in L0"


def test_representative_l0_trajectory():
    cfg = json.load(open(CFG_PATH, encoding="utf-8"))
    g = Geometry.from_json(CFG_PATH, frame_w=3840, frame_h=2160)
    for pt in L0_POINTS:
        assert g.get_lane(pt) == "L0", f"L0-flow point {pt} misassigned"


def test_representative_l3_trajectory():
    cfg = json.load(open(CFG_PATH, encoding="utf-8"))
    g = Geometry.from_json(CFG_PATH, frame_w=3840, frame_h=2160)
    for pt in L3_POINTS:
        assert g.get_lane(pt) == "L3", f"L3-flow point {pt} misassigned"


def test_boundary_behaviour_deterministic():
    cfg = json.load(open(CFG_PATH, encoding="utf-8"))
    rng = list(range(0, 3840, 320))
    points = [(x, y) for x in rng for y in range(0, 2160, 320)]
    g1 = Geometry.from_json(CFG_PATH, frame_w=3840, frame_h=2160)
    g2 = Geometry.from_json(CFG_PATH, frame_w=3840, frame_h=2160)
    first = [g1.get_lane(p) for p in points]
    second = [g2.get_lane(p) for p in points]
    third = [g1.get_lane(p) for p in points]
    assert first == second == third, "lane assignment must be a pure function"


def test_expected_direction_unchanged():
    cfg = json.load(open(CFG_PATH, encoding="utf-8"))
    dirs = {l["lane_id"]: l["expected_direction"] for l in cfg["lanes"]
            if l["lane_id"] in ("L0", "L3")}
    assert dirs["L0"] == 24.5
    assert dirs["L3"] == 196.4


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"OK: {len(tests)} tests passed")


if __name__ == "__main__":
    main()