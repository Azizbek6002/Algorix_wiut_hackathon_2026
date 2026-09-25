"""Deterministic unit tests for src/near_miss.py (PHASE 12 detector).

Run:  python tests/test_near_miss.py
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.motion import MotionState  # noqa: E402
from src.near_miss import NearMissDetector, pair_key  # noqa: E402
from src.trajectory import TrajectoryPoint, TrackTrajectory  # noqa: E402


def st(heading, speed=10.0, stationary=False, quality=0.9):
    h = math.radians(heading)
    vx = speed * math.cos(h)
    vy = -speed * math.sin(h)          # math convention: 90 = image top
    return MotionState(t=0.0, vx=vx, vy=vy,
                       speed=(0.0 if stationary else speed), accel=None,
                       heading_deg=heading, stationary=stationary,
                       quality=quality)


def _append(tr, t, pos):
    x, y = pos
    tr.append(TrajectoryPoint(t=t, x=x, y=y, bottom_y=y,
                              xyxy=(x - 60, y - 120, x + 60, y), conf=0.9))


def feed(frames, det=None):
    """Feed rows (t, [(tid, label, pos, state), ...]) through a detector.
    Returns (last_report, detector, evidence_history)."""
    det = det if det is not None else NearMissDetector()
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
        last = det.update(tracks, states, None, t)
        hist.append((t, last))
    return last, det, hist


def pair(t, id_a, pos_a, h_a, id_b, pos_b, h_b, speed=25.0,
         speed_b=None, label_a="car", label_b="car", stationary=False):
    if speed_b is None:
        speed_b = speed
    return (t, [(id_a, label_a, pos_a, st(h_a, speed, stationary)),
                (id_b, label_b, pos_b, st(h_b, speed_b, stationary))])


# ---------------------------------------------------------------------------
# scenario builders (deterministic synthetic paths; full-res pixels)

def offset_headon_scene(dur=9.5, speed=25.0):
    """Two cars on parallel lines 40 px apart, closing head-on: dangerous
    approach that never actually collides -> clear near-miss."""
    return [pair(0.1 * i, 1, (250 + speed * 0.1 * i, 1000.0), 0.0,
                 2, (750 - speed * 0.1 * i, 1040.0), 180.0, speed)
            for i in range(int(dur / 0.1) + 1)]


def crossing_scene():
    """Perpendicular near-crossing: a rises (90, slow) while b crosses right
    (0, fast); their extrapolated lines pass ~20 px apart."""
    return [pair(0.1 * i, 1, (430.0, 1100.0 - 30.0 * 0.1 * i), 90.0,
                 2, (380.0 + 80.0 * 0.1 * i, 1060.0), 0.0,
                 speed=30.0, speed_b=80.0)
            for i in range(8)]


def vehicle_person_scene():
    """Car crossing a pedestrian's path (person heading 90): small predicted
    gap (~25 px), small TTC."""
    return [pair(0.1 * i, 1, (440.0 + 30.0 * 0.1 * i, 900.0), 0.0,
                 2, (500.0, 970.0 - 20.0 * 0.1 * i), 90.0,
                 speed=30.0, speed_b=20.0,
                 label_a="car", label_b="person")
            for i in range(10)]


def headon_60_scene():
    """Fast offset head-on (dy=30 px) with immediate small TTC."""
    return [pair(0.1 * i, 1, (200.0 + 60.0 * 0.1 * i, 1000.0), 0.0,
                 2, (350.0 - 60.0 * 0.1 * i, 1030.0), 180.0, speed=60.0)
            for i in range(13)]


def _track_map(frames):
    tracks: dict = {}
    for t, entries in frames:
        for tid, label, pos, state in entries:
            tr = tracks.get(tid)
            if tr is None:
                tr = TrackTrajectory(track_id=tid, label=label)
                tracks[tid] = tr
            _append(tr, t, pos)
    return tracks


# ---------------------------------------------------------------------------
# the required 20 tests

def test_01_two_vehicles_approach_near_miss_evidence():
    last, det, hist = feed(offset_headon_scene())
    assert any(rep["evidence"] for _, rep in hist), "closing pair must fire"
    assert last["evidence"] is True
    key = pair_key(1, 2)
    assert last["active_pairs"] == [key]
    rec = last["pairs"][key]
    assert rec["reason"] == "ok"
    assert 15.0 < rec["min_predicted_distance_px"] <= 100.0
    assert rec["collision_gated"] is False
    ev_times = [t for t, rep in hist if rep["evidence"]]
    assert len(ev_times) > 10, "evidence must be sustained (>= 1 s)"
    segs = det.finalize()
    assert len(segs) == 1 and segs[0].label == "near_miss"
    assert segs[0].end > segs[0].start


def test_02_large_ttc_no_event():
    frames = [pair(0.1 * i, 1, (1000 + 25 * 0.1 * i, 1000.0), 0.0,
                   2, (1400 - 25 * 0.1 * i, 1040.0), 180.0)
              for i in range(21)]
    last, det, _ = feed(frames)
    assert last["evidence"] is False
    assert last["pairs"][pair_key(1, 2)]["reason"] == "ttc_too_large"
    assert det.finalize() == []


def test_03_large_predicted_distance_no_event():
    frames = [pair(0.2 * i, 1, (100 + 120 * 0.2 * i, 1000.0), 0.0,
                   2, (405 - 120 * 0.2 * i, 1300.0), 180.0, speed=120.0)
              for i in range(5)]
    last, det, hist = feed(frames)
    assert last["evidence"] is False
    # pick the frame where the TTC gate already passes -> the failing gate is
    # then necessarily the predicted-minimum-distance one (lines 300 px apart)
    near = min(hist, key=lambda h: h[1]["pairs"][pair_key(1, 2)]["ttc_sec"])
    rec = near[1]["pairs"][pair_key(1, 2)]
    assert rec["ttc_sec"] <= 3.0
    assert rec["reason"] == "predicted_too_large"
    assert rec["min_predicted_distance_px"] > 100.0
    assert det.finalize() == []


def test_04_objects_moving_apart_no_event():
    frames = [pair(0.1 * i, 1, (200 + 10 * 0.1 * i, 1000.0), 0.0,
                   2, (600 + 40 * 0.1 * i, 1000.0), 0.0, speed=40.0)
              for i in range(15)]
    last, det, _ = feed(frames)
    assert last["evidence"] is False
    assert last["pairs"][pair_key(1, 2)]["reason"] == "not_approaching"
    assert det.finalize() == []


def test_05_parallel_vehicles_no_event():
    frames = [pair(0.1 * i, 1, (100 + 25 * 0.1 * i, 1000.0), 0.0,
                   2, (100 + 25 * 0.1 * i, 1100.0), 0.0)
              for i in range(15)]
    last, det, _ = feed(frames)
    assert last["evidence"] is False
    assert last["pairs"][pair_key(1, 2)]["reason"] == "not_approaching"
    assert det.finalize() == []


def test_06_perpendicular_near_crossing_evidence():
    last, det, hist = feed(crossing_scene())
    fired = [rep for t, rep in hist if rep["evidence"]]
    assert fired, "crossing near miss must produce evidence"
    key = pair_key(1, 2)
    mid = fired[len(fired) // 2]                     # snapshot mid-evidence
    rec = mid["pairs"][key]
    assert 15.0 < rec["min_predicted_distance_px"] <= 100.0
    assert rec["ttc_sec"] <= 3.0
    assert rec["closing_speed_px_s"] >= 5.0
    assert rec["collision_gated"] is False


def test_07_actual_collision_no_near_miss():
    frames = [pair(0.1 * i, 1, (250 + 25 * 0.1 * i, 1000.0), 0.0,
                   2, (350 - 25 * 0.1 * i, 1000.0), 180.0)
              for i in range(21)]
    last, det, hist = feed(frames)
    assert last["evidence"] is False
    assert all(not rep["evidence"] for _, rep in hist), \
        "a potential collision must never be reported as near miss"
    # before both objects become collocated the extrapolated paths meet -> the
    # collision gate rejects it (min_predicted_distance -> 0)
    key = pair_key(1, 2)
    assert any(rep["pairs"][key]["reason"] in ("collision_predicted", "collision_distance")
               for _, rep in hist)
    assert last["pairs"][key]["collision_gated"] is True
    assert det.finalize() == []


def test_08_already_overlapping_no_near_miss():
    frames = [pair(0.1 * i, 1, (500 + 25 * 0.1 * i, 1000.0), 0.0,
                   2, (508 - 25 * 0.1 * i, 1000.0), 180.0)
              for i in range(5)]
    last, det, hist = feed(frames)
    assert last["evidence"] is False
    assert all(not rep["evidence"] for _, rep in hist)
    key = pair_key(1, 2)
    assert any(rep["pairs"][key]["reason"] == "collision_predicted"
               for _, rep in hist)
    assert last["pairs"][key]["collision_gated"] is True
    assert det.finalize() == []


def test_09_stationary_objects_no_false_positive():
    frames = [pair(0.1 * i, 1, (300.0, 1000.0), 0.0, 2, (480.0, 1000.0), 0.0,
                   stationary=True) for i in range(6)]
    last, det, _ = feed(frames)
    assert last["evidence"] is False
    assert last["pairs"][pair_key(1, 2)]["reason"] == "stationary"
    assert det.finalize() == []


def test_10_vehicle_pedestrian():
    last, det, hist = feed(vehicle_person_scene())
    assert any(rep["evidence"] for _, rep in hist), "car vs pedestrian must fire"
    key = pair_key(1, 2)
    rec = last["pairs"][key]
    assert {rec["class_a"], rec["class_b"]} == {"car", "person"}
    assert rec["active"] is True


def test_11_pair_order_one_logical_interaction():
    frames_a = crossing_scene()
    frames_b = []
    for t, entries in frames_a:
        new_entries = [(9 if tid == 1 else 5, label, pos, state)
                       for tid, label, pos, state in entries]
        frames_b.append((t, new_entries))
    last_a, det_a, _ = feed(frames_a)
    last_b, det_b, _ = feed(frames_b)
    assert len(last_a["pairs"]) == 1 and len(last_b["pairs"]) == 1
    assert pair_key(1, 2) in last_a["pairs"]
    assert pair_key(5, 9) in last_b["pairs"]
    assert [s.to_list() for s in det_a.finalize()] == \
           [s.to_list() for s in det_b.finalize()]


def test_12_pair_disappears_temporal_end():
    det = NearMissDetector()
    last, _, hist = feed(headon_60_scene(), det)
    assert last["evidence"] is True
    last_ev_t = max(t for t, rep in hist if rep["evidence"])
    tracks = _track_map(headon_60_scene())
    # track 2 disappears; only track 1 remains -> evidence must end
    for i in range(8):
        t = 1.3 + 0.1 * i
        _append(tracks[1], t, (200 + 60 * t, 1000.0))
        rep = det.update({1: tracks[1]}, {1: st(0.0, 60.0)}, None, t)
        assert rep["evidence"] is False, "pair gone -> no near-miss evidence"
    segs = det.finalize()
    assert len(segs) == 1
    s = segs[0]
    assert s.label == "near_miss"
    assert s.end <= last_ev_t + 1e-6, "run must end at the last evidence"
    assert s.end > s.start


def test_13_track_id_missing_graceful():
    tr = TrackTrajectory(track_id=1, label="car")
    _append(tr, 0.0, (100, 1000))
    det = NearMissDetector()
    r = det.update({1: tr}, {1: st(0.0), 99: st(180.0)}, None, 0.0)
    assert r["evidence"] is False and r["pairs"] == {}
    r2 = det.update({}, {}, None, 0.1)          # empty world
    assert r2["evidence"] is False and r2["pairs"] == {}
    assert det.finalize() == []


def test_14_heading_wrap():
    frames = [pair(0.1 * i, 1, (100 + 25 * 0.1 * i, 1000.0), 359.0,
                   2, (100 + 25 * 0.1 * i, 1060.0), 1.0)
              for i in range(6)]
    last, det, _ = feed(frames)
    rec = last["pairs"][pair_key(1, 2)]
    assert rec["heading_difference_deg"] == 2.0, "wrap: 359 vs 1 -> 2 deg"
    assert last["evidence"] is False
    assert det.finalize() == []


def test_15_insufficient_motion_history():
    frames = [(0.1 * i, [(1, "car", (100 + 25 * 0.1 * i, 1000.0), None),
                         (2, "car", (400 - 25 * 0.1 * i, 1000.0), st(180.0))])
              for i in range(3)]
    last, det, _ = feed(frames)
    assert last["evidence"] is False
    assert last["pairs"][pair_key(1, 2)]["reason"] == "insufficient_history"
    assert det.finalize() == []


def test_16_deterministic_repeated_input():
    outs = set()
    for _ in range(3):
        last, det, _ = feed(offset_headon_scene(dur=8.0))
        key = pair_key(1, 2)
        outs.add((round(last["pairs"][key]["ttc_sec"], 6),
                  round(last["pairs"][key]["life"]["min_ttc"], 6),
                  tuple((round(s.start, 6), round(s.end, 6), s.label)
                        for s in det.finalize())))
    assert len(outs) == 1


def test_17_temporal_confirmation():
    last, det, _ = feed(offset_headon_scene())
    segs = det.finalize()
    assert len(segs) == 1
    s = segs[0]
    assert s.label == "near_miss"
    assert s.end - s.start >= 0.8     # sustained >= min_on_duration
    assert 4.5 < s.start < 8.0        # starts when TTC drops into the band


def test_18_short_fragment_filtered():
    last, det, hist = feed(crossing_scene())
    fire_count = sum(rep["evidence"] for _, rep in hist)
    assert fire_count >= 1, "the crossing does produce momentary evidence"
    assert det.finalize() == [], "sub-second fragment must be filtered out"


def test_19_multiple_independent_pairs():
    frames = []
    for i in range(8):
        t = 0.1 * i
        far = (5000.0 + 10.0 * t, 2000.0)      # never close to anyone
        frames.append((t, [(1, "car", (430.0, 1100.0 - 30.0 * t), st(90.0, 30.0)),
                           (2, "car", (380.0 + 80.0 * t, 1060.0), st(0.0, 80.0)),
                           (3, "car", far, st(0.0, 10.0))]))
    last, det, hist = feed(frames)
    assert any(rep["active_pairs"] == [pair_key(1, 2)] for _, rep in hist)
    assert last["pairs"][pair_key(1, 3)]["active"] is False
    assert len(det.finalize()) <= 1, "benign pairs must not add segments"


def test_20_pair_state_not_inherited():
    """Pair (1,2) has a near miss, the state is pruned, and the SAME key later
    reappears -> it must start from a FRESH history (own first_t / extremes)."""
    det = NearMissDetector(pair_expire_sec=2.0)
    first = headon_60_scene()
    last, det, _ = feed(first, det)
    early_min_ttc = last["pairs"][pair_key(1, 2)]["life"]["min_ttc"]
    assert early_min_ttc < 2.0
    last_ev_t = max(t for t in [0.1 * i for i in range(13)])

    tracks = _track_map(first)
    # idle: only track 1 is fed (pair never computed -> history gets stale)
    for i in range(5):
        t = 2.5 + i
        _append(tracks[1], t, (2000.0 + 10.0 * i, 3000.0))
        det.update({1: tracks[1]}, {1: st(0.0, 10.0)}, None, t)
    assert t - last_ev_t > 2.0, "gap must exceed pair_expire_sec"

    # same key dangerous AGAIN -> fresh history
    second = headon_60_scene()
    second = [(12.0 + t, entries) for t, entries in second]
    last2, det, _ = feed(second, det)
    rec2 = last2["pairs"][pair_key(1, 2)]
    assert rec2["life"]["first_t"] >= 11.99, "history must be fresh, not reused"
    assert rec2["life"]["min_ttc"] < 2.0
    assert abs(rec2["life"]["min_ttc"] - early_min_ttc) < 0.3


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"OK: {len(tests)} tests passed")


if __name__ == "__main__":
    main()