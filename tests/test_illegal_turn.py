"""Deterministic unit tests for src/illegal_turn.py (PHASE 11 detector).

Run:  python tests/test_illegal_turn.py
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.geometry import Geometry  # noqa: E402
from src.illegal_turn import IllegalTurnDetector  # noqa: E402
from src.motion import MotionState  # noqa: E402
from src.trajectory import TrajectoryPoint, TrackTrajectory  # noqa: E402

FULL = [[0, 0], [3840, 0], [3840, 2160], [0, 2160]]
IZ = [[700, 700], [1300, 700], [1300, 1300], [700, 1300]]   # reference-space
IZ_FAR = [[1700, 700], [2300, 700], [2300, 1300], [1700, 1300]]


def make_geometry(iz_poly=IZ):
    cfg = {
        "provenance": {"reference_resolution": [3840, 2160]},
        "lanes": [],
        "road_polygon": {"points": FULL, "enabled": True},
        "intersection_zones": [{"polygon": iz_poly, "enabled": True}]
                              if iz_poly else [],
        "u_turn_zones": [],
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
    det = det or IllegalTurnDetector()
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

def turn_scene(iz_poly=IZ, extra_90_frames=10):
    """Rightwards 90deg turn through the intersection zone (kind 'left')."""
    frames = [(0.1 * i, 0.0, (280 + 40 * i, 1000)) for i in range(6)]
    loop = [(0.0, (760, 1000)), (30.0, (840, 1070)), (60.0, (900, 1160)),
            (90.0, (930, 1240))]
    for k, (h, pos) in enumerate(loop):
        frames.append((0.6 + 0.1 * k, h, pos))
    for i in range(extra_90_frames):
        frames.append((1.0 + 0.1 * i, 90.0, (940 + 30 * i, 1280)))
    return (make_geometry(iz_poly), frames)


def straight_scene(n=16):
    frames = [(0.1 * i, 0.0, (300 + 30 * i, 1000)) for i in range(n)]
    return (make_geometry(IZ), frames)


def lane_change_scene():
    """gentle S-path: net heading change stays well below 45deg."""
    hs = [0.0, 3.0, 8.0, 14.0, 18.0, 15.0, 10.0, 5.0, 0.0, -3.0, 0.0, 0.0, 0.0, 0.0]
    frames = [(0.1 * i, hs[i], (300 + 30 * i, 1000 - math.sin(i / 3.0) * 20))
              for i in range(len(hs))]
    return (make_geometry(IZ), frames)


def noisy_scene(n=20):
    jitter = [12.0, -15.0, 20.0, -8.0, 340.0, 14.0, 40.0, -18.0, 350.0, 12.0,
              348.0, -14.0, 10.0, 358.0, 15.0, -10.0, 352.0, 18.0, 6.0, -12.0]
    frames = [(0.1 * i, (jitter[i % len(jitter)]), (300 + 30 * i, 1000))
              for i in range(n)]
    return (make_geometry(IZ), frames)


def uturn_excluded_scene():
    """in-intersection 0->180deg reversal: must be deferred, not illegal_turn."""
    frames = [(0.1 * i, 0.0, (300 + 40 * i, 1000)) for i in range(6)]
    frames += [(0.6, 0.0, (760, 1000)), (0.7, 30.0, (840, 1070)),
               (0.8, 60.0, (900, 1160)), (0.9, 90.0, (920, 1240)),
               (1.0, 120.0, (860, 1280)), (1.1, 150.0, (790, 1280)),
               (1.2, 170.0, (750, 1270)), (1.3, 180.0, (770, 1260)),
               (1.4, 180.0, (810, 1250)), (1.5, 180.0, (860, 1250)),
               (1.6, 180.0, (910, 1250))]
    return (make_geometry(IZ), frames)


# ---------------------------------------------------------------------------
# the required minimum of 15 tests

def test_straight_movement_no_event():
    last, segs = run(straight_scene())
    assert last["evidence"] is False
    assert last["turn_candidates"] == []
    assert last["tracks"][1]["reason"] == "change_too_small"
    assert segs == []


def test_small_heading_change_no_event():
    last, segs = run(lane_change_scene())
    assert last["evidence"] is False
    assert last["turn_candidates"] == []
    assert last["tracks"][1]["reason"] == "change_too_small"


def test_90_degree_turn_without_intersection_no_event():
    last, segs = run(turn_scene(iz_poly=IZ_FAR))     # turn happens far from IZ
    assert last["evidence"] is False
    assert last["turn_candidates"] == []
    assert last["tracks"][1]["reason"] == "outside_intersection"
    assert segs == []


def test_90_degree_turn_inside_intersection_is_candidate():
    last, segs = run(turn_scene(iz_poly=IZ))
    assert last["evidence"] is False                 # no evidence without rule
    assert last["turn_candidates"] == [1]
    rec = last["tracks"][1]
    assert rec["turn_candidate"] is True
    assert rec["zone_index"] == 0
    assert abs(rec["signed_change_deg"]) >= 45.0
    assert rec["turn_kind"] == "left"


def test_intersection_unknown_allowed_conf_no_illegal_turn():
    g, frames = turn_scene(iz_poly=IZ)
    # default (None) -> UNKNOWN -> candidate but never evidence
    last, segs = run((g, frames), det=IllegalTurnDetector())
    assert last["turn_candidates"] == [1]
    assert last["evidence"] is False
    assert last["tracks"][1]["allowed_status"] == "UNKNOWN"
    assert last["tracks"][1]["reason"] == "turn_rule_unknown"
    assert segs == []
    # explicit UNKNOWN value behaves identically
    d2 = IllegalTurnDetector(allowed_turns={"0": "UNKNOWN"})
    last2, segs2 = run((g, frames), det=d2)
    assert last2["turn_candidates"] == [1]
    assert last2["evidence"] is False
    assert last2["tracks"][1]["allowed_status"] == "UNKNOWN"
    assert segs2 == []


def test_explicitly_allowed_turn_no_illegal_turn():
    d = IllegalTurnDetector(allowed_turns={"0": ["left"]})   # this turn is left
    last, segs = run(turn_scene(iz_poly=IZ), det=d)
    assert last["turn_candidates"] == [1]
    assert last["evidence"] is False
    assert last["tracks"][1]["allowed_status"] == "allowed"
    assert last["tracks"][1]["reason"] == "allowed_turn"
    assert segs == []


def test_explicitly_forbidden_turn_gives_evidence():
    d = IllegalTurnDetector(allowed_turns={"0": ["right"]})  # left turn forbidden
    last, segs = run(turn_scene(iz_poly=IZ), det=d)
    assert last["evidence"] is True
    assert last["active_tracks"] == [1]
    assert last["tracks"][1]["allowed_status"] == "forbidden"
    assert last["tracks"][1]["active"] is True
    assert len(segs) == 1
    assert segs[0].label == "illegal_turn"


def test_uturn_like_180_no_illegal_turn():
    last, segs = run(uturn_excluded_scene())
    assert last["evidence"] is False
    assert last["turn_candidates"] == []
    assert last["tracks"][1]["reason"] == "u_turn"
    assert segs == []


def test_stationary_object_no_event():
    g = make_geometry(IZ)
    det = IllegalTurnDetector()
    tr = TrackTrajectory(track_id=1, label="car")
    for i in range(12):
        t = 0.1 * i
        tr.append(TrajectoryPoint(t=t, x=1000, y=1000, bottom_y=1000,
                                  xyxy=(900, 800, 1100, 1000), conf=0.9))
        r = det.update({1: tr}, {1: st(90.0, stationary=True)}, g, t)
        assert r["evidence"] is False
        assert r["tracks"][1]["reason"] == "stationary"
    assert det.finalize() == []


def test_heading_noise_no_event():
    last, segs = run(noisy_scene())
    assert last["evidence"] is False
    assert last["turn_candidates"] == []
    assert segs == []


def test_insufficient_history_no_event():
    g, frames = straight_scene(n=6)
    last, segs = run((g, frames))
    assert last["evidence"] is False
    assert last["tracks"][1]["reason"] == "insufficient_history"
    assert segs == []


def test_deterministic_repeated_input():
    outs = set()
    for _ in range(3):
        d = IllegalTurnDetector(allowed_turns={"0": ["right"]})
        last, segs = run(turn_scene(iz_poly=IZ), det=d)
        outs.add((tuple(last["active_tracks"]),
                  tuple(sorted(last["turn_candidates"])),
                  tuple((round(s.start, 3), round(s.end, 3), s.label)
                        for s in segs)))
    assert len(outs) == 1


def test_multiple_vehicles_independent():
    g, frames = turn_scene(iz_poly=IZ)
    _, frames_b = straight_scene(n=len(frames))
    d = IllegalTurnDetector(allowed_turns={"0": ["right"]})
    tr_a = TrackTrajectory(track_id=1, label="car")
    tr_b = TrackTrajectory(track_id=2, label="car")
    last = None
    for i in range(len(frames)):
        t = frames[i][0]
        _, ha, pa = frames[i]
        _, hb, pb = frames_b[i]
        tr_a.append(TrajectoryPoint(t=t, x=pa[0], y=pa[1], bottom_y=pa[1],
                                    xyxy=(pa[0] - 60, pa[1] - 120,
                                          pa[0] + 60, pa[1]), conf=0.9))
        tr_b.append(TrajectoryPoint(t=t, x=pb[0], y=pb[1], bottom_y=pb[1],
                                    xyxy=(pb[0] - 60, pb[1] - 120,
                                          pb[0] + 60, pb[1]), conf=0.9))
        last = d.update({1: tr_a, 2: tr_b},
                        {1: st(ha), 2: st(hb)}, g, t)
    assert last["active_tracks"] == [1]
    assert last["turn_candidates"] == [1]
    assert last["tracks"][1]["active"] is True
    assert last["tracks"][2]["active"] is False


def test_temporal_confirmation():
    d = IllegalTurnDetector(allowed_turns={"0": ["right"]})
    last, segs = run(turn_scene(iz_poly=IZ, extra_90_frames=16), det=d)
    assert len(segs) == 1
    assert segs[0].label == "illegal_turn"
    assert segs[0].end - segs[0].start >= 0.8


def test_short_false_positive_fragment_filtered():
    """reversal-into-turn happens only at the final frames -> < min_on_duration
    -> the single fragment never becomes an event."""
    frames = [(0.1 * i, 0.0, (300 + 22 * i, 1000)) for i in range(11)]
    frames += [(1.1, 45.0, (580, 1080)), (1.2, 90.0, (600, 1160))]
    g = make_geometry(IZ)
    d = IllegalTurnDetector(allowed_turns={"0": ["right"]})
    tr = TrackTrajectory(track_id=1, label="car")
    active_count = 0
    for t, h, pos in frames:
        tr.append(TrajectoryPoint(t=t, x=pos[0], y=pos[1], bottom_y=pos[1],
                                  xyxy=(pos[0] - 60, pos[1] - 120,
                                        pos[0] + 60, pos[1]), conf=0.9))
        r = d.update({1: tr}, {1: st(h)}, g, t)
        active_count += int(r["evidence"])
    assert active_count <= 1
    assert d.finalize() == []


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"OK: {len(tests)} tests passed")


if __name__ == "__main__":
    main()