"""Deterministic unit tests for src/motion.py (no pytest dependency).

Run:  python tests/test_motion.py
"""

from __future__ import annotations

import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.motion import MotionEngine, _wrap_deg
from src.trajectory import TrackTrajectory, TrajectoryPoint


def _track(pts, track_id=1, keep_sec=5.0):
    tr = TrackTrajectory(track_id=track_id, label="car", keep_sec=keep_sec)
    for t, x, y in pts:
        tr.append(TrajectoryPoint(t=t, x=x, y=y, bottom_y=y + 10.0,
                                  xyxy=(x - 30, y, x + 30, y + 10), conf=0.9))
    return tr


def _feed_seq(pts, engine=None, dt=0.1):
    eng = engine or MotionEngine(window_s=0.5, min_points=2)
    tr = _track(pts)
    out = []
    n = len(pts)
    for i in range(2, n + 1):
        st = eng.update(tr, pts[i - 1][0])
        out.append(st)
    return eng, out


def _positions_from_headings(headings, v=30.0, dt=0.1):
    x, y = 0.0, 0.0
    pts = [(0.0, x, y)]
    for k, h in enumerate(headings):
        rad = math.radians(h)
        x += math.cos(rad) * v * dt
        y -= math.sin(rad) * v * dt
        pts.append((round((k + 1) * dt, 4), round(x, 4), round(y, 4)))
    return pts


def test_straight_line_east():
    pts = [(i * 0.1, 3.0 * i, 10.0) for i in range(7)]
    eng, out = _feed_seq(pts)
    last = out[-1]
    assert last.stationary is False
    assert abs(last.vx - 30.0) < 1.0
    assert abs(last.vy) < 0.5
    assert abs(last.speed - 30.0) < 1.0
    assert last.heading_deg is not None and abs(last.heading_deg - 0.0) < 2.0


def test_moving_up_heading_north():
    pts = [(i * 0.1, 50.0, 100.0 - 2.0 * i) for i in range(7)]
    eng, out = _feed_seq(pts)
    last = out[-1]
    assert last.heading_deg is not None and abs(last.heading_deg - 90.0) < 2.0
    assert abs(last.speed - 20.0) < 1.0


def test_stationary_detection_and_heading_kept():
    eng = MotionEngine(window_s=0.5, min_points=2)
    move = _track([(i * 0.1, 3.0 * i, 50.0) for i in range(5)])
    for i in range(1, 6):
        eng.update(move, i * 0.1)
    st = eng.get(1)
    assert st.stationary is False
    heading_when_moving = st.heading_deg
    assert heading_when_moving is not None

    jitter_x = [0, 1, -1, 0, 1, -1, 0, 0]
    jitter_y = [1, 0, 1, -1, 0, -1, 0, 1]
    stop_pts = [(0.6 + j * 0.1, 40.0 + xo, 50.0 + yo)
                for j, (xo, yo) in enumerate(zip(jitter_x, jitter_y))]
    eng2 = MotionEngine(window_s=0.5, min_points=2)
    eng2.update(_track(stop_pts[:3]), 0.8)
    st2 = eng2.get(1)
    assert st2.stationary is True
    assert st2.speed == 0.0
    assert st2.heading_deg is None  # never moved: no stable heading yet


def test_stationary_heading_no_drift():
    eng = MotionEngine(window_s=0.5, min_points=2)
    move = _track([(i * 0.1, 3.0 * i, 50.0) for i in range(5)])
    for i in range(1, 6):
        eng.update(move, i * 0.1)
    h0 = eng.get(1).heading_deg
    stop_pts = [(0.6 + j * 0.1, 41.0 + (j % 2), 50.0 + (j % 2)) for j in range(6)]
    tr_stop = _track(stop_pts)
    for k in range(2, len(stop_pts) + 1):
        eng.update(tr_stop, stop_pts[k - 1][0])
    assert eng.get(1).heading_deg == h0  # strict: stationary must not drift


def test_acceleration_sign():
    pts = [(i * 0.1, 1.0 * (i ** 2), 10.0) for i in range(7)]  # quadratic -> accel>0
    eng, out = _feed_seq(pts)
    accels = [s.accel for s in out if s.accel is not None]
    assert accels and all(a > 0 for a in accels[1:])
    assert out[-1].speed > out[0].speed


def test_heading_cross_north_no_wrap_jump():
    raw = [350.0, 355.0, 358.0, 2.0, 5.0]
    pts = _positions_from_headings(raw)
    eng = MotionEngine(window_s=0.8, min_points=3)
    _, out = _feed_seq(pts, engine=eng, dt=0.1)
    out = [s for s in out if s is not None]
    assert len(out) >= 4
    last_raw = raw[-1]
    for st in out:
        assert st.heading_deg is not None
        assert abs(_wrap_deg(st.heading_deg - last_raw)) < 30.0
    prev = out[0].heading_deg
    for st in out[1:]:
        delta = _wrap_deg(st.heading_deg - prev)
        assert 0.0 <= delta < 30.0  # smooth forward rotation across 0/360, no jump
        prev = st.heading_deg


def test_reset_clears_memory():
    eng = MotionEngine(window_s=0.5, min_points=2)
    pts = [(i * 0.1, 3.0 * i, 10.0) for i in range(5)]
    eng.update(_track(pts), 0.4)
    assert eng.get(1) is not None
    eng.reset()
    assert eng.get(1) is None


def test_use_bottom_center_equivalent_offset():
    eng_off = MotionEngine(window_s=0.5, min_points=2, use_bottom_center=True)
    eng_cen = MotionEngine(window_s=0.5, min_points=2, use_bottom_center=False)
    pts = [(i * 0.1, 3.0 * i, 10.0) for i in range(6)]
    _feed_seq(pts, engine=eng_off)
    _feed_seq(pts, engine=eng_cen)
    assert abs(eng_off.get(1).speed - eng_cen.get(1).speed) < 1e-6


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"OK: {len(tests)} tests passed")


if __name__ == "__main__":
    main()