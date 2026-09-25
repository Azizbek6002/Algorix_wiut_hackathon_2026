"""Deterministic unit tests for src/illegal_u_turn.py (PHASE 10B detector).

Run:  python tests/test_illegal_u_turn.py
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.geometry import Geometry  # noqa: E402
from src.illegal_u_turn import IllegalUTurnDetector  # noqa: E402
from src.motion import MotionState  # noqa: E402
from src.trajectory import TrajectoryPoint, TrackTrajectory  # noqa: E402

FULL = [[0, 0], [3840, 0], [3840, 2160], [0, 2160]]
ZONE = [[500, 500], [1500, 500], [1500, 1500], [500, 1500]]   # reference-space


def make_geometry(zone_poly=ZONE):
    cfg = {
        "provenance": {"reference_resolution": [3840, 2160]},
        "lanes": [],
        "road_polygon": {"points": FULL, "enabled": True},
        "u_turn_zones": [{"polygon": zone_poly, "enabled": True}]
                         if zone_poly else [],
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


def run(scene, det=None, tid=1, label="car"):
    """feed (t, heading, pos) frames through a fresh detector; return the report
    of the LAST frame + the finalized events."""
    det = det or IllegalUTurnDetector()
    g, frames = scene
    tr = TrackTrajectory(track_id=tid, label=label)
    last = None
    for t, h, pos, *_ in frames:
        tr.append(TrajectoryPoint(t=t, x=pos[0], y=pos[1], bottom_y=pos[1],
                                  xyxy=(pos[0] - 60, pos[1] - 120,
                                        pos[0] + 60, pos[1]),
                                  conf=0.9))
        last = det.update({tid: tr}, {tid: st(h)}, g, t)
    return last, det.finalize()


# ---------------------------------------------------------------------------
# scenario builders (deterministic, well-shaped synthetic paths)

def uturn_scene(with_zone_poly=True, extra_180_frames=10):
    """Right-handed ~180deg reversal THROUGH the zone: outside approach (0 deg),
    an in-zone loop rotating 0->180, then a straight 180-deg exit leg."""
    frames = [(0.1 * i, 0.0, (280 + 36 * i, 1000)) for i in range(7)]
    loop = [(0.0, (560, 1010)), (30.0, (660, 1040)), (60.0, (760, 1090)),
            (90.0, (820, 1160)), (120.0, (800, 1240)), (150.0, (720, 1300)),
            (180.0, (600, 1330))]
    for k, (h, pos) in enumerate(loop):
        frames.append((0.7 + 0.1 * k, h, pos))
    for i in range(extra_180_frames):                   # exit leg at 180
        frames.append((1.4 + 0.1 * i, 180.0, (600 - 40 * (i + 1), 1330)))
    return (make_geometry(ZONE if with_zone_poly else None), frames)


def straight_scene(n=16):
    frames = [(0.1 * i, 0.0, (300 + 30 * i, 1040)) for i in range(n)]
    return (make_geometry(ZONE), frames)


def ninety_scene():
    """90deg right turn: exits at heading 90, in-zone all the way."""
    frames = [(0.1 * i, 0.0, (300 + 30 * i, 500)) for i in range(7)]
    for i in range(5):
        frames.append((0.7 + 0.1 * i, 18.0 * (i + 1), (510 + 10 * i, 500 + 40 * i)))
    for i in range(4):
        frames.append((1.2 + 0.1 * i, 90.0, (560 + 30 * i, 700 + 30 * i)))
    return (make_geometry(ZONE), frames)


def noisy_scene(n=20):
    """moving straight but heading jitters (small noise, a few 60deg spikes)."""
    jitter = [15.0, -10.0, 20.0, 355.0, -15.0, 12.0, 60.0, -20.0, 340.0, 15.0,
              350.0, -18.0, 14.0, 358.0, 20.0, -12.0, 355.0, 16.0, 340.0, 10.0]
    frames = [(0.1 * i, jitter[i % len(jitter)], (300 + 30 * i, 1040))
              for i in range(n)]
    return (make_geometry(ZONE), frames)


# ---------------------------------------------------------------------------
# the required minimum of 13 tests

def test_straight_vehicle_no_event():
    last, segs = run(straight_scene())
    assert last["evidence"] is False
    assert segs == []


def test_90_degree_turn_no_uturn():
    last, segs = run(ninety_scene())
    assert last["evidence"] is False
    assert last["tracks"][1]["reason"] in ("turn_too_small", "reversal_too_small")
    assert segs == []


def test_180_degree_turn_gives_uturn_evidence():
    last, segs = run(uturn_scene())
    assert last["evidence"] is True
    rec = last["tracks"][1]
    assert rec["active"] is True
    assert rec["reversal_deg"] is not None and rec["reversal_deg"] >= 180 - 60
    assert rec["cum_turn_deg"] is not None and abs(rec["cum_turn_deg"]) >= 120.0


def test_uturn_outside_configured_zone_no_event():
    last, segs = run(uturn_scene(with_zone_poly=False))
    assert last["evidence"] is False
    assert last["tracks"][1]["reason"] == "outside_u_turn_zone"
    assert segs == []


def test_uturn_inside_configured_zone_evidence():
    _, segs = run(uturn_scene(with_zone_poly=True))
    # a full sustained u-turn crossing the zone must yield a confirmed segment
    assert len(segs) == 1
    s = segs[0]
    assert s.label == "illegal_u_turn"
    assert 0.0 < s.start and s.end > s.start


def test_stationary_object_no_event():
    g = make_geometry(ZONE)
    det = IllegalUTurnDetector()
    tr = TrackTrajectory(track_id=1, label="car")
    # parked vehicle: never moving
    for i in range(12):
        t = 0.1 * i
        tr.append(TrajectoryPoint(t=t, x=1000, y=1000, bottom_y=1000,
                                  xyxy=(900, 800, 1100, 1000), conf=0.9))
        r = det.update({1: tr}, {1: st(180.0, stationary=True)}, g, t)
        assert r["evidence"] is False, "stationary must never fire"
        assert r["tracks"][1]["reason"] == "stationary"
    assert det.finalize() == []

    # pivoting in place: rotates 180 but does not travel -> no movement
    det2 = IllegalUTurnDetector()
    tr2 = TrackTrajectory(track_id=1, label="car")
    for i in range(4):
        tr2.append(TrajectoryPoint(t=0.1 * i, x=1000, y=1000, bottom_y=1000,
                                   xyxy=(900, 800, 1100, 1000), conf=0.9))
    for i in range(12):
        t = 0.4 + 0.1 * i
        h = 180.0 * min(1.0, i / 11.0)
        tr2.append(TrajectoryPoint(t=t, x=1000, y=1000, bottom_y=1000,
                                   xyxy=(900, 800, 1100, 1000), conf=0.9))
        r = det2.update({1: tr2}, {1: st(h, speed=20.0)}, g, t)
        assert r["evidence"] is False, "pivot without travel must not fire"
    assert det2.finalize() == []
    assert r["tracks"][1]["reason"] == "arc_too_short"


def test_noisy_heading_no_false_positive():
    last, segs = run(noisy_scene())
    assert last["evidence"] is False
    assert segs == []


def test_insufficient_history_no_event():
    frames = straight_scene(n=5)[1]          # only 5 frames / 0.4 s
    g = make_geometry(ZONE)
    last, segs = run((g, frames))
    assert last["evidence"] is False
    assert last["tracks"][1]["reason"] == "insufficient_history"
    assert segs == []


def test_final_direction_not_sufficiently_opposite():
    """rotation reaches 120deg (shape ok) but reversal < 135 -> no U-turn."""
    frames = [(0.1 * i, 0.0, (280 + 36 * i, 1040)) for i in range(7)]
    for i in range(7):
        frames.append((0.7 + 0.1 * i, 20.0 * (i + 1), (600 - 45 * i, 1030 + 30 * i)))
    for i in range(3):
        frames.append((1.4 + 0.1 * i, 120.0, (240 - 30 * i, 1250)))
    last, segs = run((make_geometry(ZONE), frames))
    assert last["evidence"] is False
    assert last["tracks"][1]["reason"] == "reversal_too_small"
    assert segs == []


def test_multiple_vehicles_independent():
    g, frames_a = uturn_scene()
    _, frames_b = straight_scene(n=len(frames_a))
    det = IllegalUTurnDetector()
    tr_a = TrackTrajectory(track_id=1, label="car")
    tr_b = TrackTrajectory(track_id=2, label="car")
    last = None
    for i in range(len(frames_a)):
        t = frames_a[i][0]
        _, ha, pa = frames_a[i]
        _, hb, pb = frames_b[i]
        tr_a.append(TrajectoryPoint(t=t, x=pa[0], y=pa[1], bottom_y=pa[1],
                                    xyxy=(pa[0] - 60, pa[1] - 120,
                                          pa[0] + 60, pa[1]), conf=0.9))
        tr_b.append(TrajectoryPoint(t=t, x=pb[0], y=pb[1], bottom_y=pb[1],
                                    xyxy=(pb[0] - 60, pb[1] - 120,
                                          pb[0] + 60, pb[1]), conf=0.9))
        last = det.update({1: tr_a, 2: tr_b},
                          {1: st(ha), 2: st(hb)}, g, t)
    assert last["evidence"] is True
    assert last["active_tracks"] == [1]
    assert last["tracks"][1]["active"] is True
    assert last["tracks"][2]["active"] is False


def test_deterministic_repeated_input():
    outs = set()
    for _ in range(3):
        last, segs = run(uturn_scene(extra_180_frames=8))
        evidence_seq = [last["evidence"]]
        outs.add((tuple(sorted(last["tracks"].keys())),
                  tuple((round(s.start, 3), round(s.end, 3), s.label)
                        for s in segs)))
    assert len(outs) == 1


def test_temporal_confirmation():
    last, segs = run(uturn_scene(extra_180_frames=10))
    # sustained evidence (>= min_on_duration 0.8 s) -> exactly one segment
    assert len(segs) == 1
    s = segs[0]
    assert s.label == "illegal_u_turn"
    assert s.end - s.start >= 0.8


def test_short_false_positive_fragment_filtered():
    """the 180 reversal is reached inside the zone only at the very last frame
    (single active frame) -> below min_on_duration, no event."""
    frames = [(0.1 * i, 0.0, (300 + 22 * i, 1040)) for i in range(11)]   # 1.0 s
    for i in range(4):           # quick in-zone rotation 45..180
        frames.append((1.1 + 0.1 * i, 45.0 + 45.0 * i,
                       (560 + 6 * i, 1060 + 10 * i)))
    g = make_geometry(ZONE)
    det = IllegalUTurnDetector()
    tr = TrackTrajectory(track_id=1, label="car")
    active_count = 0
    for t, h, pos in frames:
        tr.append(TrajectoryPoint(t=t, x=pos[0], y=pos[1], bottom_y=pos[1],
                                  xyxy=(pos[0] - 60, pos[1] - 120,
                                        pos[0] + 60, pos[1]), conf=0.9))
        r = det.update({1: tr}, {1: st(h)}, g, t)
        active_count += int(r["evidence"])
    assert active_count <= 1, "reversal must be a momentary peak, not sustained"
    assert det.finalize() == []      # single-frame blip filtered by the engine


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"OK: {len(tests)} tests passed")


if __name__ == "__main__":
    main()