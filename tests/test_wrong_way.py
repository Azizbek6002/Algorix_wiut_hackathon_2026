"""Deterministic unit tests for src/wrong_way.py (PHASE 9 detector).

Run:  python tests/test_wrong_way.py
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.geometry import Geometry  # noqa: E402
from src.motion import MotionEngine, MotionState  # noqa: E402
from src.trajectory import TrajectoryPoint, TrackTrajectory  # noqa: E402
from src.wrong_way import WrongWayDetector, lane_direction_to_motion_deg  # noqa: E402

FULL = [[0, 0], [3840, 0], [3840, 2160], [0, 2160]]   # whole frame in ref coords
LANE_WHOLE = FULL


def make_geometry(expected_direction=90.0, lane_poly=LANE_WHOLE,
                  u_turn_poly=None):
    cfg = {
        "provenance": {"reference_resolution": [3840, 2160]},
        "lanes": [{"lane_id": "L", "expected_direction": expected_direction,
                   "polygon": lane_poly, "enabled": True}],
        "road_polygon": {"points": FULL, "enabled": True},
        "u_turn_zones": [{"polygon": u_turn_poly, "enabled": True}]
                         if u_turn_poly else [],
    }
    return Geometry(cfg, 3840, 2160)


def st(heading, speed=10.0, stationary=False, quality=0.9):
    h = math.radians(heading)
    vx = speed * math.cos(h)
    vy = -speed * math.sin(h)          # math convention: 90 = image top
    return MotionState(t=0.0, vx=vx, vy=vy,
                       speed=(0.0 if stationary else speed), accel=None,
                       heading_deg=heading, stationary=stationary,
                       quality=quality)


def track(pid=1, label="car", x=400.0, y=400.0):
    tr = TrackTrajectory(track_id=pid, label=label)
    tr.append(TrajectoryPoint(t=0.0, x=x, y=y, bottom_y=y,
                              xyxy=(x - 100, y - 200, x + 100, y), conf=0.9))
    return tr


def det(geometry, tracks, states, t_sec=0.0):
    return WrongWayDetector().update(tracks, states, geometry, t_sec)


def test_heading_convention_helper():
    # image (y-down) lane direction 90 == down-screen == motion math 270
    assert abs(lane_direction_to_motion_deg(90.0) - 270.0) < 1e-9
    assert abs(lane_direction_to_motion_deg(0.0) - 0.0) < 1e-9
    assert abs(lane_direction_to_motion_deg(180.0) - 180.0) < 1e-9
    assert abs(lane_direction_to_motion_deg(360.0 + 270.0) - 90.0) < 1e-9


def test_compliant_direction_no_event():
    g = make_geometry(expected_direction=90.0)      # flow down the screen
    r = det(g, {1: track()}, {1: st(heading=270.0)})  # car drives down: compliant
    assert r["evidence"] is False
    assert r["tracks"][1]["reason"] == "below_threshold"
    assert WrongWayDetector().finalize() == []


def test_opposite_direction_gives_evidence():
    g = make_geometry(expected_direction=90.0)
    r = det(g, {1: track()}, {1: st(heading=90.0)})   # up-screen: against the flow
    assert r["evidence"] is True
    assert r["active_tracks"] == [1]
    rec = r["tracks"][1]
    assert rec["active"] is True
    assert rec["lane"] == "L"
    assert rec["heading_deg"] == 90.0
    assert rec["deviation_deg"] == 180.0


def test_stationary_vehicle_no_event():
    g = make_geometry(expected_direction=90.0)
    r = det(g, {1: track()}, {1: st(heading=90.0, stationary=True)})
    assert r["evidence"] is False
    assert r["tracks"][1]["reason"] == "stationary"


def test_low_speed_noise_no_event():
    g = make_geometry(expected_direction=90.0)
    d = WrongWayDetector(min_speed_px_s=4.0)
    r = d.update({1: track()}, {1: st(heading=90.0, speed=1.5)}, g, 0.0)
    assert r["evidence"] is False
    assert r["tracks"][1]["reason"] == "below_min_speed"


def test_heading_jitter_near_0_360_no_false_positive():
    g = make_geometry(expected_direction=0.0)        # compliant = heading ~0
    d = WrongWayDetector()
    for i, h in enumerate([359.9, 0.1, 359.85, 0.0, 359.7, 0.2]):
        r = d.update({1: track()}, {1: st(heading=h)}, g, t_sec=0.1 * i)
        assert r["evidence"] is False, f"jitter at {h} triggered wrong_way"
    assert d.finalize() == []


def test_unknown_lane_no_event():
    lane_poly = [[100, 100], [700, 100], [700, 700], [100, 700]]
    g = make_geometry(expected_direction=90.0, lane_poly=lane_poly)
    r = det(g, {1: track(x=20.0, y=20.0)}, {1: st(heading=90.0)})
    assert r["evidence"] is False
    assert r["tracks"][1]["reason"] == "unknown_lane"


def test_deviation_below_threshold_no_event():
    g = make_geometry(expected_direction=90.0)       # expected math 270
    d = WrongWayDetector(angle_threshold=120.0)
    # 160 vs 270 -> deviation 110 < 120
    r = d.update({1: track()}, {1: st(heading=160.0)}, g, 0.0)
    assert r["evidence"] is False
    assert r["tracks"][1]["reason"] == "below_threshold"
    # 359 vs 270 -> deviation 89 also < 120
    r2 = d.update({1: track()}, {1: st(heading=359.0)}, g, 0.0)
    assert r2["evidence"] is False


def test_sustained_wrong_direction_produces_event():
    g = make_geometry(expected_direction=90.0)
    d = WrongWayDetector()                            # min_on_duration 0.8
    for i in range(21):                               # wrong for t in 0..2.0
        r = d.update({1: track()}, {1: st(heading=90.0)}, g, t_sec=0.1 * i)
        assert r["evidence"] is True
    segs = d.finalize()
    assert len(segs) == 1
    s = segs[0]
    assert s.label == "wrong_way"
    assert s.start == 0.0 and abs(s.end - 2.0) < 1e-9
    assert s.to_list() == [0.0, 2.0, "wrong_way"]


def test_short_wrong_fragment_filtered():
    g = make_geometry(expected_direction=90.0)
    d = WrongWayDetector()                            # min_on_duration 0.8
    for i in range(4):                                # only 0..0.3 s wrong
        d.update({1: track()}, {1: st(heading=90.0)}, g, t_sec=0.1 * i)
    assert d.finalize() == []


def test_u_turn_like_reversal_suppressed_in_zone():
    g = make_geometry(expected_direction=90.0, u_turn_poly=FULL)
    d = WrongWayDetector(suppress_in_u_turn_zone=True)
    for i in range(21):                               # sustained "wrong" heading
        r = d.update({1: track()}, {1: st(heading=90.0)}, g, t_sec=0.1 * i)
        assert r["evidence"] is False
        assert r["tracks"][1]["in_u_turn_zone"] is True
    assert d.finalize() == []                          # never a wrong_way event

    # without the zone suppression the same motion would be flagged
    g2 = make_geometry(expected_direction=90.0, u_turn_poly=None)
    d2 = WrongWayDetector(suppress_in_u_turn_zone=False)
    for i in range(21):
        d2.update({1: track()}, {1: st(heading=90.0)}, g2, t_sec=0.1 * i)
    assert len(d2.finalize()) == 1


def test_lane_specific_directions():
    # two lanes side by side with DIFFERENT expected directions
    lane_a = [[0, 0], [2000, 0], [2000, 1000], [0, 1000]]
    lane_b = [[0, 1200], [2000, 1200], [2000, 2200], [0, 2200]]
    cfg = {
        "provenance": {"reference_resolution": [3840, 2160]},
        "road_polygon": {"points": FULL, "enabled": True},
        "lanes": [
            {"lane_id": "A", "expected_direction": 90.0,
             "polygon": lane_a, "enabled": True},
            {"lane_id": "B", "expected_direction": 0.0,
             "polygon": lane_b, "enabled": True},
        ],
        "u_turn_zones": [],
    }
    g = Geometry(cfg, 3840, 2160)
    d = WrongWayDetector()
    # in lane A (flow down, expected math 270): heading 90 is wrong -> active
    trA = track(1, y=500.0)
    # in lane B (flow right, expected math 0): heading 0 is compliant
    trB = track(2, y=1700.0)
    r = d.update({1: trA, 2: trB},
                 {1: st(heading=90.0), 2: st(heading=0.0)}, g, 0.0)
    assert r["active_tracks"] == [1]
    assert r["tracks"][1]["active"] is True and r["tracks"][1]["lane"] == "A"
    assert r["tracks"][2]["active"] is False and r["tracks"][2]["lane"] == "B"
    assert r["tracks"][2]["reason"] == "below_threshold"


def test_multiple_vehicles_independent():
    g = make_geometry(expected_direction=90.0)
    r = det(g, {1: track(1), 2: track(2)},
            {1: st(heading=90.0), 2: st(heading=270.0)})
    assert r["evidence"] is True
    assert r["active_tracks"] == [1]
    assert r["tracks"][1]["active"] is True
    assert r["tracks"][2]["active"] is False


def test_uses_motion_engine_state():
    # real TrackTrajectory + MotionEngine -> MotionState -> detector (no fake states)
    g = make_geometry(expected_direction=90.0)
    tr = TrackTrajectory(track_id=7, label="car")

    def feed(y0, dy, n=8):
        for i in range(n):
            yy = y0 + dy * i
            tr.append(TrajectoryPoint(t=0.1 * i, x=400.0 + 5 * i, y=yy,
                                      bottom_y=yy,
                                      xyxy=(300, yy - 200, 500, yy), conf=0.9))
        return MotionEngine().update(tr, 0.1 * (n - 1))

    # driving DOWN (heading ~270): compliant
    state_down = feed(200.0, +10.0)
    assert state_down is not None and abs(state_down.speed) > 0
    assert 210.0 <= state_down.heading_deg <= 330.0
    r = det(g, {7: tr}, {7: state_down})
    assert r["evidence"] is False

    # driving UP (heading ~90): wrong way
    tr2 = TrackTrajectory(track_id=7, label="car")
    for i in range(8):
        yy = 700.0 - 10.0 * i
        tr2.append(TrajectoryPoint(t=0.1 * i, x=400.0 + 5 * i, y=yy,
                                   bottom_y=yy, xyxy=(300, yy - 200, 500, yy),
                                   conf=0.9))
    state_up = MotionEngine().update(tr2, 0.7)
    assert state_up is not None
    assert 60.0 <= state_up.heading_deg <= 120.0
    r2 = det(g, {7: tr2}, {7: state_up})
    assert r2["evidence"] is True
    assert r2["tracks"][7]["deviation_deg"] >= 120.0


def test_deterministic_output():
    g = make_geometry(expected_direction=90.0)
    seq = [0.0, 0.1, 0.2, 0.9, 1.0, 1.4, 1.5, 2.0]

    outs = set()
    for _ in range(3):
        d = WrongWayDetector()
        reports = []
        for t in seq:
            reports.append(d.update({1: track()}, {1: st(heading=90.0)},
                                    g, t_sec=t)["evidence"])
        segs = [[round(s.start, 3), round(s.end, 3), s.label]
                for s in d.finalize()]
        outs.add((tuple(reports), tuple(map(tuple, segs))))
    assert len(outs) == 1


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"OK: {len(tests)} tests passed")


if __name__ == "__main__":
    main()