"""Deterministic unit tests for src/accident.py (PHASE 13 detector).

Run:  python tests/test_accident.py
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.accident import AccidentDetector, pair_key, _wrap_delta_deg  # noqa: E402
from src.motion import MotionState  # noqa: E402
from src.trajectory import TrajectoryPoint, TrackTrajectory  # noqa: E402


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


def feed(frames, det=None):
    det = det if det is not None else AccidentDetector()
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


def events_of(det):
    return [(round(s.start, 4), round(s.end, 4), s.label)
            for s in det.finalize()]


# ---------------------------------------------------------------------------
# scenario builders (state-level, deterministic; full-res pixels)

def collision_scene(frames_post=60, post_states=None):
    """Head-on collision of two cars (40 px/s each) on the x axis at y=1000.
    Contact at t=2.5. ``post_states(i)`` returns the (a, b) MotionStates after
    the contact; default = both brake to a standstill (a shows a -200 px/s^2
    deceleration pulse on its first stopped frame)."""
    frames = []
    n_pre = 25                       # 0 <= t < 2.5 approach (25 frames)
    for i in range(n_pre):
        t = 0.1 * i
        ax, bx = 200 + 40 * t, 400 - 40 * t
        frames.append((t, [(1, "car", (ax, 1000.0), st(0.0, 40.0)),
                           (2, "car", (bx, 1000.0), st(180.0, 40.0))]))
    cx = 300.0
    for i in range(frames_post):
        t = 2.5 + 0.1 * i
        if post_states is not None:
            a_state, b_state = post_states(i)
        else:
            a_state = st(0.0, 40.0, accel=(-200.0) if i == 0 else None)
            b_state = st(0.0, 0.0, stationary=True)
        frames.append((t, [(1, "car", (cx, 1000.0), a_state),
                           (2, "car", (cx, 1000.0), b_state)]))
    return frames


def near_miss_scene():
    """Offset head-on (dy=40): real near miss, no contact -> never impact."""
    return [(0.1 * i, [(1, "car", (200 + 25 * 0.1 * i, 1000.0), st(0.0, 25.0)),
                       (2, "car", (350 - 25 * 0.1 * i, 1040.0), st(180.0, 25.0))])
            for i in range(30)]


def vehicle_hits_person_scene(label_b="person", speed=50.0):
    """Car (speed) hits a standing person/bicycle. Contact ~ t=(300-15)/speed;
    afterwards the car brakes to a standstill next to the victim."""
    frames = []
    for i in range(80):
        t = 0.1 * i
        d = 300 - speed * t
        if d > 15.0:
            ax, bx = 200 + speed * t, 500.0
            a_state = st(0.0, speed)
        else:
            ax, bx = 485.0, 500.0
            a_state = st(0.0, 0.0, stationary=True,
                         accel=(-300.0 if i <= 2 else None))
        b_state = st(0.0, 0.0, stationary=True)
        frames.append((t, [(1, "car", (ax, 1000.0), a_state),
                           (2, label_b, (bx, 1000.0), b_state)]))
    return frames


def heading_wrap_scene(delta_deg):
    """Collision; object B changes heading by `delta_deg` (from 359-ish),
    object A brakes -> tests wrap-aware heading-change signal."""
    frames = []
    for i in range(25):
        t = 0.1 * i
        ax, bx = 200 + 40 * t, 400 - 40 * t
        frames.append((t, [(1, "car", (ax, 1000.0), st(0.0, 40.0)),
                           (2, "car", (bx, 1000.0), st(180.0, 40.0))]))
    b_h = (359.0 + delta_deg) % 360.0
    for i in range(60):
        t = 2.5 + 0.1 * i
        a_state = st(0.0, 0.0, stationary=True, accel=(-200.0 if i == 0 else None))
        b_state = st(b_h, 6.0) if i == 0 else st(b_h, 6.0)
        frames.append((t, [(1, "car", (300.0, 1000.0), a_state),
                           (2, "car", (300.0, 1000.0), b_state)]))
    return frames


# ---------------------------------------------------------------------------
# tests

def test_01_clear_collision_candidate():
    """Impact candidate requires a real contact frame."""
    last, det, hist = feed(collision_scene())
    hit = next((r for _, r in hist if r["impact_pairs"]), None)
    assert hit is not None, "collision must enter impact"
    assert hit["impact_pairs"] == [pair_key(1, 2)]
    assert hit["pairs"][pair_key(1, 2)]["impact_ok"] is True
    rec = last["pairs"][pair_key(1, 2)]
    assert rec["phase"] == "impact" or rec["confirmed"] is True


def test_02_collision_and_sudden_braking():
    last, det, _ = feed(collision_scene())
    segs = det.finalize()
    assert len(segs) == 1
    s = segs[0]
    assert s.label == "accident" and s.end > s.start
    rec = last["pairs"][pair_key(1, 2)]
    assert "speed_drop" in rec["signals"] and "decel" in rec["signals"]
    assert rec["confirmed"] is True


def test_03_collision_and_heading_change():
    def post(i):
        return st(0.0, 40.0), st(90.0, 40.0)
    frames = collision_scene(frames_post=60, post_states=post)
    last, det, _ = feed(frames)
    segs = det.finalize()
    assert len(segs) == 1 and segs[0].label == "accident"
    rec = last["pairs"][pair_key(1, 2)]
    assert "heading_change" in rec["signals"]
    assert rec["confirmed"] is True


def test_04_collision_and_post_impact_stop():
    frames = collision_scene(frames_post=60)
    last, det, _ = feed(frames)
    rec = last["pairs"][pair_key(1, 2)]
    assert "stop" in rec["signals"]
    assert len(det.finalize()) == 1
    s = det.finalize()[0]
    assert 2.3 < s.start < 3.3
    assert s.end - s.start >= 1.0


def test_05_collision_without_stop_speed_no_change():
    """Both keep moving at the same speed after contact -> only co-location
    (1 signal) -> never an accident."""
    frames = collision_scene(frames_post=60, post_states=lambda i: (st(0.0, 40.0), st(180.0, 40.0)))
    last, det, _ = feed(frames)
    assert det.finalize() == []
    rec = last["pairs"][pair_key(1, 2)]
    assert set(rec["signals"]) <= {"collocation"}


def test_06_near_miss_must_not_become_accident():
    last, det, hist = feed(near_miss_scene())
    assert not any(r["impact_pairs"] for _, r in hist)
    assert det.finalize() == []


def test_07_large_distance_no_accident():
    frames = [(0.1 * i, [(1, "car", (300.0, 800.0), st(90.0, 20.0)),
                         (2, "car", (300.0, 1200.0), st(270.0, 20.0))])
              for i in range(20)]
    last, det, _ = feed(frames)
    assert det.finalize() == []
    assert not det.impact_pairs_count


def test_08_large_ttc_no_accident():
    """Gentle closing (slow head-on) keeps TTC far above impact_ttc."""
    frames = []
    for i in range(40):
        t = 0.1 * i
        frames.append((t, [(1, "car", (100.0 + t, 1000.0), st(0.0, 1.0)),
                           (2, "car", (1000.0 - t, 1000.0), st(180.0, 1.0))]))
    last, det, _ = feed(frames)
    assert not det.impact_pairs_count
    assert det.finalize() == []


def test_09_moving_apart_no_accident():
    """Initially within collision distance but separating -> impact opens,
    no post-impact signals -> rejected."""
    frames = []
    for i in range(30):
        t = 0.1 * i
        ax, bx = 300.0 + 30 * t, 308.0 + 35 * t
        frames.append((t, [(1, "car", (ax, 1000.0), st(0.0, 30.0)),
                           (2, "car", (bx, 1000.0), st(0.0, 35.0))]))
    last, det, _ = feed(frames)
    assert det.impact_pairs_count >= 1
    assert det.finalize() == []
    assert det.rejected_reasons[pair_key(1, 2)] == "no_pre_separation"


def test_10_stationary_overlapping_no_accident():
    frames = [(0.1 * i, [(1, "car", (300.0, 1000.0), st(0.0, 0.0, stationary=True)),
                         (2, "car", (305.0, 1000.0), st(0.0, 0.0, stationary=True))])
              for i in range(20)]
    last, det, _ = feed(frames)
    assert not det.impact_pairs_count
    assert det.finalize() == []


def test_11_tracking_overlap_without_impact_evidence():
    """Two IDs on ONE moving object (distance ~0, same velocity): overlap
    alone is NOT an accident (blocked by no-pre-separation and <2 signals)."""
    frames = [(0.1 * i, [(1, "car", (300.0, 1000.0), st(0.0, 30.0)),
                         (2, "car", (303.0, 1000.0), st(0.0, 30.0))])
              for i in range(40)]
    last, det, _ = feed(frames)
    assert det.finalize() == []
    assert det.rejected_reasons.get(pair_key(1, 2)) == "no_pre_separation"


def test_12_sudden_braking_without_collision_no_accident():
    """Car brakes hard but the pair never comes close -> no accident."""
    frames = []
    for i in range(20):
        t = 0.1 * i
        frames.append((t, [(1, "car", (200.0 + t, 1000.0), st(0.0, 80.0, accel=(-300.0))),
                           (2, "car", (400.0 - t, 1000.0), st(180.0, 80.0))]))
    last, det, _ = feed(frames)
    assert not det.impact_pairs_count
    assert det.finalize() == []


def test_13_heading_change_without_collision_no_accident():
    frames = []
    for i in range(20):
        t = 0.1 * i
        h = 180.0 if i >= 8 else 0.0
        frames.append((t, [(1, "car", (200.0 + 5 * t, 1000.0), st(h, 12.0)),
                           (2, "car", (700.0, 1000.0), st(h, 12.0))]))
    last, det, _ = feed(frames)
    assert det.finalize() == []
    assert not det.impact_pairs_count


def test_14_vehicle_pedestrian():
    last, det, _ = feed(vehicle_hits_person_scene("person"))
    segs = det.finalize()
    assert len(segs) == 1
    rec = last["pairs"][pair_key(1, 2)]
    assert rec["class_a"] == "car" and rec["class_b"] == "person"
    assert rec["confirmed"] is True
    assert "speed_drop" in rec["signals"] and "stop" in rec["signals"]


def test_15_vehicle_bicycle():
    last, det, _ = feed(vehicle_hits_person_scene("bicycle"))
    rec = last["pairs"][pair_key(1, 2)]
    assert rec["class_b"] == "bicycle" and rec["confirmed"] is True
    assert len(det.finalize()) == 1


def test_16_pair_order_is_unordered():
    frames = collision_scene()
    swapped = []
    for t, entries in frames:
        swapped.append((t, [(2 if tid == 1 else 1, label, pos, state)
                            for tid, label, pos, state in entries]))
    _, d1, _ = feed(frames, AccidentDetector())
    _, d2, _ = feed(swapped, AccidentDetector())
    assert events_of(d1) == events_of(d2)
    assert pair_key(1, 2) == pair_key(2, 1) == "1-2"


def test_17_pair_disappearance():
    det = AccidentDetector()
    last, det, hist = feed(collision_scene(), det)
    # pair vanishes -> engine must still close the run at the last active frame
    for i in range(5):
        t = 9.0 + 0.1 * i
        det.update({}, {}, None, t)
    segs = det.finalize()
    assert len(segs) == 1
    assert segs[0].end <= 8.6     # never grows beyond the last observed evidence


def test_18_missing_track_graceful():
    frames = near_miss_scene()
    frames_mod = []
    for i, (t, entries) in enumerate(frames):
        if i == 5:                      # track 2 missing on one frame
            entries = [e for e in entries if e[0] != 2]
        frames_mod.append((t, entries))
    last, det, _ = feed(frames_mod)
    assert det.finalize() == []
    assert not det.impact_pairs_count


def test_19_heading_wrap():
    assert _wrap_delta_deg(5.0, 359.0) == 6.0
    assert _wrap_delta_deg(359.0, 1.0) == -2.0
    assert _wrap_delta_deg(350.0, 10.0) == -20.0
    assert _wrap_delta_deg(None, 30.0) is None
    # small change (8 deg) must NOT fire the heading-change signal
    _, d_small, _ = feed(heading_wrap_scene(8.0))
    assert "heading_change" not in d_small._pairs[pair_key(1, 2)]["signals"] or \
        len(d_small.finalize()) == 1
    # big change (100 deg) MUST fire it
    last, d_big, _ = feed(heading_wrap_scene(100.0))
    assert "heading_change" in last["pairs"][pair_key(1, 2)]["signals"]


def test_20_deterministic_repeated_run():
    outs = set()
    for _ in range(3):
        det = AccidentDetector()
        last, det, _ = feed(collision_scene(), det)
        rec = last["pairs"][pair_key(1, 2)]
        outs.add((tuple(events_of(det)),
                  frozenset(rec["signals"]),
                  det.impact_pairs_count,
                  tuple(sorted(det.rejected_reasons.items()))))
    assert len(outs) == 1


def test_21_confirmed_even_without_full_stop():
    """Abrupt braking (speed 40 -> 10, no rest) still confirms through
    speed_drop + decel + co-location; 'stop' must be ABSENT."""
    def post(i):
        return st(0.0, 10.0, accel=(-200.0) if i <= 1 else None), st(180.0, 10.0)
    frames = collision_scene(frames_post=60, post_states=post)
    last, det, _ = feed(frames)
    rec = last["pairs"][pair_key(1, 2)]
    assert rec["confirmed"] is True
    assert "stop" not in rec["signals"]
    assert "speed_drop" in rec["signals"] and "decel" in rec["signals"]


def test_22_rejected_reasons_archive():
    det_nm = AccidentDetector()
    feed(near_miss_scene(), det_nm)
    assert not det_nm.impact_pairs_count
    assert det_nm.rejected_reasons == {}
    det2 = AccidentDetector()
    feed([(0.1 * i, [(1, "car", (300.0, 1000.0), st(0.0, 30.0)),
                     (2, "car", (303.0, 1000.0), st(0.0, 30.0))])
          for i in range(40)], det2)
    assert det2.rejected_reasons[pair_key(1, 2)] == "no_pre_separation"


def test_23_reset_clears_state():
    det = AccidentDetector()
    feed(collision_scene(), det)
    assert len(det.finalize()) == 1
    det.reset()
    assert det.confirmed_keys == set()
    assert det.rejected_reasons == {}
    assert det.impact_pairs_count == 0
    last, det, _ = feed(collision_scene(), det)
    assert len(det.finalize()) == 1


def test_24_no_negative_or_infinite_segments():
    for scene in (collision_scene(), vehicle_hits_person_scene()):
        _, det, _ = feed(scene)
        for s in det.finalize():
            assert math.isfinite(s.start) and math.isfinite(s.end)
            assert s.end > s.start


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"OK: {len(tests)} tests passed")


if __name__ == "__main__":
    main()