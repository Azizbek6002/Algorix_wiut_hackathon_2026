"""Deterministic unit tests for src/failure_to_yield.py (PHASE 14 detector).

Run:  python tests/test_failure_to_yield.py
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.failure_to_yield import (  # noqa: E402
    FailureToYieldDetector, _angle_between_deg, crosswalk_distance_px, pair_key)
from src.motion import MotionState  # noqa: E402
from src.trajectory import TrajectoryPoint, TrackTrajectory  # noqa: E402


CW = [(740.0, 700.0), (1060.0, 700.0), (1060.0, 1450.0), (740.0, 1450.0)]


class FakeGeometry:
    """Minimal crosswalk geometry: a single rectangle, ref == full-res."""

    def __init__(self, poly=None, sx=1.0, sy=1.0):
        self.crosswalks = [poly or CW]
        self.sx, self.sy = float(sx), float(sy)

    def to_ref(self, p):
        return (p[0] / self.sx, p[1] / self.sy)

    def is_in_crosswalk(self, p):
        return _point_in_polygon(self.to_ref(p), self.crosswalks[0])


def _point_in_polygon(p, poly):
    x, y = p
    inside = False
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def st(heading, speed=10.0, stationary=False, quality=0.9, accel=None):
    h = math.radians(heading)
    vx = speed * math.cos(h)
    vy = -speed * math.sin(h)
    return MotionState(t=0.0, vx=vx, vy=vy, speed=(0.0 if stationary else speed),
                       accel=accel, heading_deg=heading, stationary=stationary,
                       quality=quality)


def _append(tr, t, pos):
    x, y = pos
    tr.append(TrajectoryPoint(t=t, x=x, y=y, bottom_y=y,
                              xyxy=(x - 60, y - 120, x + 60, y), conf=0.9))


def feed(frames, det=None, geometry=None):
    det = det if det is not None else FailureToYieldDetector()
    geom = geometry if geometry is not None else FakeGeometry()
    tracks: dict = {}
    hist: list = []
    last = None
    for t, entries in frames:
        for tid, label, pos, state in entries:
            tr = tracks.get(tid)
            if tr is None:
                tr = TrackTrajectory(track_id=tid, label=label)
                tracks[tid] = tr
            _append(tr, t, pos)
        states = {}
        for tid, label, pos, state in entries:
            if state is not None and tid in tracks:
                states[tid] = state
        last = det.update(tracks, states, geom, t)
        hist.append((t, last))
    return last, det, hist


def events_of(det):
    return [(round(s.start, 4), round(s.end, 4), s.label)
            for s in det.finalize()]


# ---------------------------------------------------------------------------
# scenario builder: a pedestrian crosses a vertical crosswalk strip (heading
# 270, moving DOWN toward the road lane y=1100) while a car drives RIGHT
# (heading 0) along y=1100 toward the pedestrian. Contact of paths near (850,
# 1100). See the failure_to_yield layout in the test docstrings.

def conflict_scene(t_max=8.0, veh_speed=100.0, ped_speed=25.0,
                   veh_brake=None, veh_stop_at=None, veh_id=1, ped_id=2):
    frames = []
    for i in range(int(t_max * 10)):
        t = 0.1 * i
        veh_x = 100.0 + veh_speed * t
        ped_y = 1000.0 + ped_speed * t
        a = st(0.0, veh_speed)
        if veh_stop_at is not None and t >= veh_stop_at:
            a = st(0.0, 0.0, stationary=True)
        elif veh_brake is not None and veh_brake <= t < veh_brake + 2.0:
            a = st(0.0, max(5.0, veh_speed - 50.0 * (t - veh_brake)),
                   accel=(-80.0))
        elif veh_brake is not None and veh_brake + 2.0 <= t < veh_brake + 3.5:
            a = st(0.0, 60.0)
        b = st(270.0, ped_speed)
        frames.append((t, [(veh_id, "car", (veh_x, 1100.0), a),
                           (ped_id, "person", (850.0, ped_y), b)]))
    return frames


def _conflict_frame_window(det, hist, pred):
    return [(t, r) for t, r in hist if pred(t, r)]


# ---------------------------------------------------------------------------
# tests

def test_01_vehicle_approaches_crossing_pedestrian():
    last, det, hist = feed(conflict_scene())
    segs = det.finalize()
    assert len(segs) == 1 and segs[0].label == "failure_to_yield"
    assert 3.5 < segs[0].start < 4.5
    # end == last frame where TTC <= 3.5 (t=7.0); at t=7.1 TTC jumps to 3.7
    assert 6.5 < segs[0].end <= 7.2


def test_02_dangerous_ttc_drives_evidence():
    last, det, hist = feed(conflict_scene())
    ttc_frames = [(t, r) for t, r in hist
                  if r["active_pairs"] and math.isfinite(
                      r["pairs"]["1-2"]["ttc_sec"])]
    assert ttc_frames
    assert min(r["pairs"]["1-2"]["ttc_sec"] for _, r in ttc_frames) <= 3.5
    assert any(r["evidence"] for _, r in hist)


def test_03_vehicle_continues_without_braking():
    last, det, hist = feed(conflict_scene())
    seeing = [(t, r) for t, r in hist
              if t < 7.0 and r["active_pairs"] and not r["pairs"]["1-2"]["veh_yielding"]]
    assert seeing
    assert len(det.finalize()) == 1


def test_04_vehicle_brakes_sufficiently_no_event():
    last, det, hist = feed(conflict_scene(veh_brake=4.0))
    assert det.finalize() == []
    assert any(r["pairs"]["1-2"]["veh_yielding"] for _, r in hist)
    # the braking response must be recorded as the frame-level rejection reason
    # (later frames overtake the persistent hist reason, so scan the frames)
    assert any(r["pairs"]["1-2"]["reason"] == "vehicle_yielding" for _, r in hist)


def test_05_vehicle_stops_before_crossing_no_event():
    last, det, hist = feed(conflict_scene(veh_stop_at=4.0))
    assert det.finalize() == []
    assert "vehicle_not_moving" in det._hist["1-2"]["reason"]


def test_06_pedestrian_outside_crosswalk_no_event():
    frames = []
    for i in range(60):
        t = 0.1 * i
        veh_x = 100.0 + 100.0 * t
        frames.append((t, [(1, "car", (veh_x, 1100.0), st(0.0, 100.0)),
                           (2, "person", (720.0, 1125.0), st(270.0, 25.0))]))
    last, det, hist = feed(frames)
    assert det.finalize() == []
    assert all(not r["pairs"]["1-2"]["ped_in_crosswalk"]
               for _, r in hist if "1-2" in r["pairs"])
    assert "pedestrian_not_in_crosswalk" in det._hist["1-2"]["reason"]


def test_07_pedestrian_stationary_no_event():
    frames = []
    for i in range(60):
        t = 0.1 * i
        frames.append((t, [(1, "car", (100.0 + 100.0 * t, 1100.0), st(0.0, 100.0)),
                           (2, "person", (850.0, 1125.0), st(0.0, 0.0, stationary=True))]))
    last, det, hist = feed(frames)
    assert det.finalize() == []
    assert "pedestrian_stationary" in det._hist["1-2"]["reason"]


def test_08_vehicle_stationary_no_event():
    frames = []
    for i in range(60):
        t = 0.1 * i
        frames.append((t, [(1, "car", (850.0, 1100.0), st(0.0, 0.0, stationary=True)),
                           (2, "person", (850.0, 1125.0), st(270.0, 25.0))]))
    last, det, hist = feed(frames)
    assert det.finalize() == []
    assert "vehicle_not_moving" in det._hist["1-2"]["reason"]


def test_09_large_distance_no_event():
    frames = []
    for i in range(60):
        t = 0.1 * i
        # vehicle crosses far away: the pedestrian is outside interaction range
        frames.append((t, [(1, "car", (100.0 + 100.0 * t, 1100.0), st(0.0, 100.0)),
                           (2, "person", (3000.0, 1100.0), st(270.0, 20.0))]))
    last, det, hist = feed(frames)
    assert det.finalize() == []
    assert not any(r["active_pairs"] for _, r in hist)


def test_10_large_ttc_no_event():
    frames = []
    for i in range(80):
        t = 0.1 * i
        veh_x = 100.0 + 25.0 * t
        frames.append((t, [(1, "car", (veh_x, 1100.0), st(0.0, 25.0)),
                           (2, "person", (850.0, 1100.0), st(270.0, 10.0))]))
    last, det, hist = feed(frames)
    assert det.finalize() == []
    assert not any(r["active_pairs"] for _, r in hist)


def test_11_moving_apart_no_event():
    frames = []
    for i in range(60):
        t = 0.1 * i
        # pedestrian faster than the vehicle on the same line -> separating
        frames.append((t, [(1, "car", (100.0 + 40.0 * t, 1100.0), st(0.0, 40.0)),
                           (2, "person", (850.0 + 60.0 * t, 1100.0), st(0.0, 60.0))]))
    last, det, hist = feed(frames)
    assert det.finalize() == []
    assert not any(r["active_pairs"] for _, r in hist)


def test_12_pedestrian_past_crossing_no_event():
    frames = []
    for i in range(60):
        t = 0.1 * i
        frames.append((t, [(1, "car", (100.0 + 100.0 * t, 1100.0), st(0.0, 100.0)),
                           (2, "person", (900.0, 500.0), st(270.0, 20.0))]))
    last, det, hist = feed(frames)
    assert det.finalize() == []
    assert all(not r["pairs"]["1-2"]["ped_in_crosswalk"]
               for _, r in hist if "1-2" in r["pairs"])


def test_13_vehicle_past_crossing_no_event():
    frames = []
    for i in range(60):
        t = 0.1 * i
        frames.append((t, [(1, "car", (2000.0 - 40.0 * t, 1100.0), st(180.0, 40.0)),
                           (2, "person", (900.0, 1150.0), st(270.0, 20.0))]))
    last, det, hist = feed(frames)
    assert det.finalize() == []
    assert "vehicle_past" in det._hist["1-2"]["reason"] or \
        all(not r["active_pairs"] for _, r in hist)


def test_14_no_conflicting_trajectory_no_event():
    frames = []
    for i in range(60):
        t = 0.1 * i
        # vehicle passes below; pedestrian crosses above -> big predicted gap
        frames.append((t, [(1, "car", (100.0 + 80.0 * t, 1600.0), st(0.0, 80.0)),
                           (2, "person", (850.0, 1250.0), st(90.0, 25.0))]))
    last, det, hist = feed(frames)
    assert det.finalize() == []


def test_15_only_vehicle_pedestrian_pairs_counted():
    frames = []
    for i in range(20):
        t = 0.1 * i
        frames.append((t, [(1, "car", (100.0 + 100.0 * t, 1100.0), st(0.0, 100.0)),
                           (3, "car", (850.0, 1100.0), st(0.0, 100.0)),
                           (2, "person", (900.0, 1150.0), st(270.0, 25.0))]))
    last, det, hist = feed(frames)
    keys = set(last["pairs"].keys())
    assert keys == {"1-2", "2-3"}     # car-car (1-3) is dropped explicitly
    for key in keys:
        assert last["pairs"][key]["veh_class"] == "car"
        assert last["pairs"][key]["ped_class"] == "person"


def test_16_pair_order_unordered():
    f_a = conflict_scene(veh_id=5, ped_id=6)
    f_b = conflict_scene(veh_id=6, ped_id=5)
    _, d_a, _ = feed(f_a)
    _, d_b, _ = feed(f_b)
    assert events_of(d_a) == events_of(d_b)
    assert pair_key(5, 6) == pair_key(6, 5) == "5-6"
    assert set(d_a._hist) == {"5-6"} and set(d_b._hist) == {"5-6"}


def test_17_missing_track_graceful():
    frames = conflict_scene()
    for i, (t, entries) in enumerate(frames):
        if i == 12:                       # pedestrian missing on one frame
            frames[i] = (t, [e for e in entries if e[0] != 2])
    last, det, hist = feed(frames)
    assert det.finalize() and not det == None  # noqa: no crash + still detects


def test_18_heading_wrap():
    frames = []
    for i in range(20):
        t = 0.1 * i
        h = 1.0 if i % 2 else 359.0        # heading jumps across the wrap seam
        frames.append((t, [(1, "car", (2000.0, 1100.0), st(h, 40.0)),
                           (2, "person", (900.0, 1150.0), st(270.0, 20.0))]))
    last, det, hist = feed(frames)
    assert det.finalize() == []            # far apart -> no conflict, no crash
    assert _angle_between_deg(10.0, 0.0, 10.0, 0.0) == 0.0
    assert _angle_between_deg(1.0, 0.0, -1.0, 0.0) == 180.0


def test_19_deterministic_repeated_run():
    outs = set()
    for _ in range(3):
        last, det, hist = feed(conflict_scene())
        outs.add((tuple(events_of(det)),
                  tuple(sorted({k: r["conflict"] for k, r in
                                last["pairs"].items()}.items()))))
    assert len(outs) == 1


def test_20_temporal_confirmation_required():
    # conflict lasts only ~4 frames -> never reaches min_on_duration
    frames = []
    for i in range(20):
        t = 0.1 * i
        if 12 <= i <= 15:
            veh_x, ped_y = 700.0 + 100.0 * (i - 12), 1125.0
            b = st(270.0, 25.0)
        else:
            veh_x, ped_y = 2000.0, 1125.0
            b = st(270.0, 25.0)
        frames.append((t, [(1, "car", (veh_x, 1100.0), st(0.0, 100.0)),
                           (2, "person", (ped_y - 275.0, ped_y), b)]))
    last, det, hist = feed(frames)
    assert any(r["active_pairs"] for _, r in hist)
    assert det.finalize() == []


def test_21_min_duration_filter_and_configurability():
    # 8 frames of conflict -> 0.7 s of continuous evidence
    frames = [
        (0.1 * i, [(1, "car", (700.0 + 50.0 * (0.1 * i), 1100.0), st(0.0, 50.0)),
                   (2, "person", (850.0, 1125.0), st(270.0, 25.0))])
        for i in range(8)]
    d_default = FailureToYieldDetector()
    feed(frames, d_default)
    # 0.7 s >= min_on_duration 0.6 and >= min_duration 0.5 -> kept
    assert len(d_default.finalize()) == 1
    d_long = FailureToYieldDetector(min_duration=0.9)
    feed(frames, d_long)
    assert d_long.finalize() == []             # 0.7 s < min_duration 0.9 -> dropped
    d_short = FailureToYieldDetector(min_duration=0.2, min_on_duration=0.1)
    feed(frames, d_short)
    assert len(d_short.finalize()) == 1        # short segment kept when mins lowered


def test_22_multiple_independent_pedestrian_vehicle_pairs():
    frames = []
    for i in range(60):
        t = 0.1 * i
        frames.append((t, [
            (1, "car", (100.0 + 100.0 * t, 1100.0), st(0.0, 100.0)),
            (2, "person", (850.0, 1000.0 + 25.0 * t), st(270.0, 25.0)),
            # second independent conflict: car4 drives LEFT (heading 180) along
            # y=1300 toward pedestrian3 crossing at x=900 (active t ~4.7..5.9)
            (4, "car", (2000.0 - 160.0 * t, 1300.0), st(180.0, 160.0)),
            (3, "person", (900.0, 1180.0 + 25.0 * t), st(270.0, 25.0)),
        ]))
    last, det, hist = feed(frames)
    both = [r["active_pairs"] for _, r in hist
            if len(r["active_pairs"]) >= 2]
    assert both
    assert det.finalize()                   # at least one temporal event


def test_23_pair_disappearance():
    last, det, hist = feed(conflict_scene(t_max=8.0))
    for i in range(5):
        t = 9.0 + 0.1 * i
        det.update({}, {}, FakeGeometry(), t)
    segs = det.finalize()
    assert len(segs) == 1
    assert segs[0].end <= 7.5


def test_24_no_pair_state_inheritance():
    """Two separate conflict episodes (gap > pair_expire) -> two segments."""
    ep1 = conflict_scene(t_max=8.0)
    gap = [(8.5 + 0.1 * i, []) for i in range(30)]     # no tracks: prune now
    ep2 = []
    for i in range(50):
        t = 11.5 + 0.1 * i
        veh_x = 100.0 + 100.0 * (t - 11.5)
        ped_y = 1000.0 + 25.0 * (t - 11.5)
        ep2.append((t, [(1, "car", (veh_x, 1100.0), st(0.0, 100.0)),
                        (2, "person", (850.0, ped_y), st(270.0, 25.0))]))
    _, det, _ = feed(ep1 + gap + ep2)
    segs = det.finalize()
    assert len(segs) == 2
    assert segs[1].start > 11.0


def test_25_crosswalk_margin_widens_interaction_area():
    a_frames = []
    for i in range(40):
        t = 0.1 * i
        # pedestrian standing 20px LEFT of the 740x-left strip edge, moving
        a_frames.append((t, [(1, "car", (100.0 + 100.0 * t, 1100.0), st(0.0, 100.0)),
                             (2, "person", (720.0, 1125.0), st(270.0, 25.0))]))
    _, d0, h0 = feed(a_frames, FailureToYieldDetector(crosswalk_margin_px=0.0))
    _, dm, hm = feed(a_frames, FailureToYieldDetector(crosswalk_margin_px=30.0))
    assert all(not r["pairs"]["1-2"]["ped_in_crosswalk"] for _, r in h0)
    assert any(r["pairs"]["1-2"]["ped_in_crosswalk"] for _, r in hm)
    assert d0.finalize() == [] and dm.finalize()   # margin unlocks the conflict


def test_26_stationary_grace_mid_crosswalk():
    frames = []
    for i in range(40):
        t = 0.1 * i
        moving = i <= 1
        b = st(270.0, 25.0) if moving else st(0.0, 0.0, stationary=True)
        frames.append((t, [(1, "car", (850.0, 1100.0), st(0.0, 100.0)),
                           (2, "person", (850.0, 1000.0 + 25.0 * (0.1 * i) if moving
                                          else 1005.0), b)]))
    last, det, hist = feed(frames)
    t1 = next((r for t, r in hist if abs(t - 1.0) < 0.05))["pairs"]["1-2"]
    t2 = next((r for t, r in hist if abs(t - 2.0) < 0.05))["pairs"]["1-2"]
    assert t1["crossing"] is True        # still within stationary_grace (1.5s)
    assert t2["crossing"] is False       # grace expired -> pedestrian_stationary


def test_27_braking_memory_suppresses_after_recovery():
    frames = conflict_scene(veh_brake=4.0)
    last, det, hist = feed(frames)
    at45 = next((r for t, r in hist if abs(t - 4.5) < 0.05))["pairs"]["1-2"]
    at75 = next((r for t, r in hist if abs(t - 7.5) < 0.05))["pairs"]["1-2"]
    assert at45["veh_yielding"] is True   # accel None already but memory kept
    assert at75["veh_yielding"] is False  # last brake 5.9 -> 1.6 s ago -> cleared


def test_28_away_pedestrian_no_event():
    # pedestrian directly ahead of the vehicle, in the crosswalk, fleeing along
    # the same line (heading 0, 80 px/s) while the car closes at 200 px/s:
    # every pairwise gate passes once distance <= 350 (t >= 2.42) and the only
    # thing suppressing the event is the pedestrian_moving_away heading gate.
    frames = []
    for i in range(40):
        t = 0.1 * i
        frames.append((t, [(1, "car", (100.0 + 200.0 * t, 1100.0), st(0.0, 200.0)),
                           (2, "person", (740.0 + 80.0 * t, 1100.0), st(0.0, 80.0))]))
    last, det, hist = feed(frames)
    assert det.finalize() == []
    assert "pedestrian_moving_away" in det._hist["1-2"]["reason"]


def test_29_report_shape_and_fields():
    last, det, hist = feed(conflict_scene())
    assert set(last.keys()) == {"t_sec", "evidence", "active_pairs",
                                "pairs", "rejected"}
    rec = last["pairs"]["1-2"]
    for f in ("veh_id", "ped_id", "veh_class", "ped_class", "distance_px",
              "closing_speed_px_s", "relative_speed_px_s", "ttc_sec",
              "min_predicted_distance_px", "veh_speed", "veh_accel",
              "ped_speed", "ped_in_crosswalk", "crossing", "veh_yielding",
              "conflict", "reason", "first_t", "last_t"):
        assert f in rec
    assert rec["veh_class"] == "car" and rec["ped_class"] == "person"


def test_30_crosswalk_distance_helper():
    geom = FakeGeometry()
    assert crosswalk_distance_px(geom, (850.0, 1000.0)) == 0.0   # inside
    assert crosswalk_distance_px(geom, (720.0, 1125.0)) == 20.0   # just outside
    assert crosswalk_distance_px(FakeGeometry(poly=[(0.0, 0.0), (10.0, 0.0),
                                                    (0.0, 10.0)]),
                                 (50.0, 50.0)) > 0.0
    assert crosswalk_distance_px(FakeGeometry(poly=[(0.0, 0.0), (10.0, 0.0),
                                                    (0.0, 10.0)]),
                                 (5.0, 2.0)) == 0.0


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"OK: {len(tests)} tests passed")


if __name__ == "__main__":
    main()