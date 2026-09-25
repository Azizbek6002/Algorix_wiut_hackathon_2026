"""Deterministic unit tests for src/interaction.py (PHASE 7: pairwise + TTC).

Run:  python tests/test_interaction.py
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.interaction import (  # noqa: E402
    EPS,
    PairwiseInteractionEngine,
    heading_difference_deg,
)


def st(t=0.0, vx=0.0, vy=0.0, speed=None, heading=None, stationary=False):
    """Build a MotionState quickly (same fields as src/motion.MotionState)."""
    from src.motion import MotionState
    return MotionState(t=t, vx=vx, vy=vy,
                       speed=speed if speed is not None else math.hypot(vx, vy),
                       accel=None, heading_deg=heading, stationary=stationary,
                       quality=1.0)


def test_toward_each_other():
    eng = PairwiseInteractionEngine()
    a = st(vx=100, vy=0, heading=0.0)    # east
    b = st(vx=-100, vy=0, heading=180.0)  # west, heading at A
    p = eng.compute(1, 2, t_sec=10.0, pos_a=(0, 500), pos_b=(1000, 500),
                    state_a=a, state_b=b, class_a="car", class_b="car")
    assert p.distance_px == 1000.0
    assert p.relative_speed_px_s == 200.0
    assert p.closing_speed_px_s == 200.0
    assert p.relative_velocity_px_s == (-200.0, 0.0)
    assert abs(p.ttc_sec - 5.0) < 1e-9
    assert p.approaching is True
    assert abs(p.time_to_closest_approach_sec - 5.0) < 1e-9
    assert abs(p.min_predicted_distance_px - 0.0) < 1e-9
    assert p.heading_difference_deg == 180.0
    assert p.same_class is True and p.class_a == "car" and p.class_b == "car"


def test_moving_apart():
    eng = PairwiseInteractionEngine()
    p = eng.compute(1, 2, 0.0, (0, 0), (100, 0),
                    st(vx=-100, heading=180.0), st(vx=100, heading=0.0))
    assert p.distance_px == 100.0
    assert p.closing_speed_px_s == -200.0
    assert math.isinf(p.ttc_sec)
    assert p.approaching is False
    assert p.time_to_closest_approach_sec == 0.0
    assert p.min_predicted_distance_px == 100.0     # they only get farther
    assert p.heading_difference_deg == 180.0


def test_parallel_same_speed():
    eng = PairwiseInteractionEngine()
    # same velocity -> zero relative velocity, never converge
    p = eng.compute(1, 2, 0.0, (0, 100), (0, 200),
                    st(vx=100, heading=0.0), st(vx=100, heading=0.0))
    assert p.distance_px == 100.0
    assert p.relative_speed_px_s == 0.0
    assert p.closing_speed_px_s == 0.0
    assert math.isinf(p.ttc_sec)
    assert p.approaching is False
    assert p.time_to_closest_approach_sec == 0.0
    assert p.min_predicted_distance_px == 100.0
    assert p.heading_difference_deg == 0.0


def test_perpendicular_pass_by():
    eng = PairwiseInteractionEngine()
    # A rides east along y=0, B crosses from above downwards -> closest pass
    # is at t_ca=2.5 s with min distance ~70.7 px: a pass-by, NOT a collision.
    a = st(vx=100, vy=0, heading=0.0)
    b = st(vx=0, vy=100, heading=270.0)
    p = eng.compute(1, 2, 0.0, (0, 0), (300, -200), a, b)
    assert abs(p.distance_px - math.hypot(300, 200)) < 1e-9
    assert abs(p.time_to_closest_approach_sec - 2.5) < 1e-9
    assert abs(p.min_predicted_distance_px - math.hypot(50, 50)) < 1e-9
    assert p.approaching is True            # closing at this moment
    assert p.ttc_sec > p.time_to_closest_approach_sec
    assert abs(p.ttc_sec - 2.6) < 1e-9
    assert p.heading_difference_deg == 90.0
    assert p.min_predicted_distance_px > 0.0  # pass-by vs real collision


def test_stationary_vs_moving():
    eng = PairwiseInteractionEngine()
    car = st(vx=100, heading=0.0)             # moving east at x=0
    parked = st(vx=0, vy=0, stationary=True, heading=180.0)  # facing west
    p = eng.compute(7, 8, 0.0, (100, 0), (0, 0), parked, car)
    # moving object is 100 px left of the parked one, closing at 100 px/s
    assert p.distance_px == 100.0
    assert p.closing_speed_px_s == 100.0
    assert abs(p.ttc_sec - 1.0) < 1e-9
    assert p.approaching is True
    assert abs(p.min_predicted_distance_px - 0.0) < 1e-9
    assert p.heading_difference_deg == 180.0

    # parked vehicle with no known heading -> heading diff unknown, not a crash
    p2 = eng.compute(7, 8, 0.0, (100, 0), (0, 0),
                     st(vx=0, vy=0, stationary=True, heading=None), car)
    assert p2.heading_difference_deg is None
    assert abs(p2.ttc_sec - 1.0) < 1e-9


def test_already_very_close():
    eng = PairwiseInteractionEngine()
    # d below the tiny-distance floor, still closing -> immediate impact, no div-by-zero
    p = eng.compute(1, 2, 0.0, (0, 0), (5e-4, 0),
                    st(vx=100), st(vx=-100))
    assert p.distance_px == 5e-4
    assert p.ttc_sec == 0.0
    assert p.approaching is True

    # tiny distance but zero closing -> inf, and no ZeroDivision anywhere
    q = eng.compute(1, 2, 0.0, (0, 0), (5e-4, 0),
                    st(vx=100), st(vx=100))
    assert q.distance_px == 5e-4
    assert math.isinf(q.ttc_sec)
    assert q.approaching is False


def test_zero_relative_velocity():
    eng = PairwiseInteractionEngine()
    p = eng.compute(1, 2, 0.0, (100, 100), (300, 100),
                    st(vx=0, vy=0, stationary=True),
                    st(vx=0, vy=0, stationary=True))
    assert p.relative_speed_px_s == 0.0
    assert p.closing_speed_px_s == 0.0
    assert math.isinf(p.ttc_sec)
    assert p.approaching is False
    assert p.time_to_closest_approach_sec == 0.0
    assert p.min_predicted_distance_px == 200.0
    assert p.heading_difference_deg is None


def test_heading_wrap():
    assert heading_difference_deg(359.0, 1.0) == 2.0
    assert heading_difference_deg(1.0, 359.0) == 2.0
    assert heading_difference_deg(350.0, 10.0) == 20.0
    assert heading_difference_deg(0.0, 359.0) == 1.0
    assert heading_difference_deg(180.0, 0.0) == 180.0
    assert heading_difference_deg(270.0, 90.0) == 180.0
    assert heading_difference_deg(45.0, 45.0) == 0.0
    assert heading_difference_deg(None, 90.0) is None
    assert heading_difference_deg(90.0, None) is None
    assert heading_difference_deg(None, None) is None


def test_insufficient_history():
    eng = PairwiseInteractionEngine()
    # no MotionStates at all (insufficient history) -> distance still reported,
    # motion-based fields degrade gracefully, no crash.
    p = eng.compute(3, 4, 12.5, (0, 0), (10, 0), None, None,
                    class_a="truck", class_b="car")
    assert p.distance_px == 10.0
    assert p.class_a == "truck" and p.class_b == "car"
    assert p.same_class is False
    assert p.relative_speed_px_s == 0.0
    assert math.isinf(p.ttc_sec)
    assert p.approaching is False
    assert p.heading_difference_deg is None

    # only one object has motion history -> still deterministic
    q = eng.compute(3, 4, 12.5, (0, 0), (10, 0), None, st(vx=100), )
    assert q.distance_px == 10.0
    assert math.isinf(q.ttc_sec)


def test_determinism_and_engine_reuse():
    args = dict(track_id_a=1, track_id_b=2, t_sec=0.0,
                pos_a=(0.0, 0.0), pos_b=(1000.0, 0.0),
                state_a=st(vx=100), state_b=st(vx=-100))
    eng = PairwiseInteractionEngine()
    first = eng.compute(**args)
    for _ in range(5):
        again = eng.compute(**args)
        assert again == first
    other = PairwiseInteractionEngine().compute(**args)
    assert other == first


def test_pairs_unique_and_sorted():
    eng = PairwiseInteractionEngine()
    items = [
        (1, "car", (0, 0), st(vx=100)),
        (2, "car", (1000, 0), st(vx=-100)),
        (3, "ped", (200, 200), None),
    ]
    pairs = eng.pairs(items, t_sec=0.0)
    assert [ (p.track_id_a, p.track_id_b) for p in pairs ] == [(1, 2), (1, 3), (2, 3)]
    assert all(p.track_id_a < p.track_id_b for p in pairs)
    # ids are the per-track integers passed in, never a fabricated self-pair
    assert all(p.track_id_a != p.track_id_b for p in pairs)
    toward = pairs[0]
    assert toward.same_class is True and abs(toward.ttc_sec - 5.0) < 1e-9


def test_pairs_sorts_unsorted_input():
    eng = PairwiseInteractionEngine()
    # deliberately not sorted by id -> output is still (id_a < id_b) ordered
    items = [
        (7, "car", (0, 0), st(vx=100)),
        (5, "car", (1000, 0), st(vx=-100)),
        (9, "ped", (200, 200), None),
    ]
    pairs = eng.pairs(items, t_sec=0.0)
    assert [ (p.track_id_a, p.track_id_b) for p in pairs ] == [(5, 7), (5, 9), (7, 9)]
    assert all(p.track_id_a < p.track_id_b for p in pairs)


def test_eps_and_tiny_racing():
    # closing speed exactly on the EPS boundary -> treated as no approach
    eng = PairwiseInteractionEngine()
    p = eng.compute(1, 2, 0.0, (0, 0), (1000, 0),
                    st(vx=EPS / 2), st(vx=0))
    assert p.closing_speed_px_s <= EPS
    assert math.isinf(p.ttc_sec)
    # but a marginally large distance with real closing yields the exact ratio
    q = eng.compute(1, 2, 0.0, (0, 0), (1000, 0),
                    st(vx=EPS * 10), st(vx=0))
    assert q.approaching is True


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"OK: {len(tests)} tests passed")


if __name__ == "__main__":
    main()