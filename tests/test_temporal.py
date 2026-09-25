"""Deterministic unit tests for src/temporal.py (PHASE 8: temporal engine).

Run:  python tests/test_temporal.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.temporal import EventSegment, TemporalEventEngine  # noqa: E402


def tups(segments):
    return [tuple(s) for s in segments]  # (start, end, label)


def test_single_short_false_positive():
    eng = TemporalEventEngine(min_on_duration=0.5, allowed_gap=2.0,
                              merge_gap=1.0, min_duration=0.0)
    eng.update("near_miss", 0.0, evidence=True)
    eng.update("near_miss", 1.0, evidence=False)
    eng.update("near_miss", 2.0, evidence=False)   # gap exceeded -> run closes
    assert eng.finalize() == []                     # never confirmed


def test_minimum_duration():
    eng = TemporalEventEngine(min_on_duration=0.0, allowed_gap=0.5,
                              merge_gap=1.0, min_duration=1.0)
    eng.update("x", 0.0, evidence=True)             # 0.0 s event -> dropped
    eng.update("x", 1.0, evidence=False)            # closes [0,0]
    eng.update("x", 3.0, evidence=True)
    for t in (3.1, 3.5, 4.0, 4.2):
        eng.update("x", t, evidence=True)
    eng.update("x", 5.0, evidence=False)            # closes [3,4.2], dur 1.2
    assert tups(eng.finalize()) == [(3.0, 4.2, "x")]


def test_delayed_start_confirmation_uses_real_start():
    eng = TemporalEventEngine(min_on_duration=2.0, allowed_gap=10.0,
                              merge_gap=0.0, min_duration=0.0)
    for t in range(6):                              # 6 active updates, 0..5
        eng.update("alert", float(t), evidence=True)
    # confirmed only after active_total >= 2.0, but the segment's start must be
    # the real first active frame (0.0), not the confirmation moment (~2.0)
    segs = eng.finalize()
    assert tups(segs) == [(0.0, 5.0, "alert")]


def test_delayed_end():
    eng = TemporalEventEngine(min_on_duration=0.0, allowed_gap=3.0,
                              merge_gap=1.0, min_duration=0.0)
    for t in range(6):                              # active 0..5
        eng.update("x", float(t), evidence=True)
    eng.update("x", 6.0, evidence=False)            # gap 1 < 3 -> still open
    eng.update("x", 7.0, evidence=False)            # gap 2 < 3 -> still open
    eng.update("x", 8.0, evidence=False)            # gap 3 >= 3 -> closes
    segs = eng.finalize()
    # end = last active frame (5), NOT the close time (8)
    assert tups(segs) == [(0.0, 5.0, "x")]


def test_allowed_gap_resumes_same_run():
    eng = TemporalEventEngine(min_on_duration=0.0, allowed_gap=3.0,
                              merge_gap=0.0, min_duration=0.0)
    for t in (0.0, 1.0, 2.0):
        eng.update("x", t, evidence=True)
    eng.update("x", 3.0, evidence=False)            # gap 1 -> run survives
    for t in (4.0, 5.0, 6.0):
        eng.update("x", t, evidence=True)           # resumes inside the gap
    eng.update("x", 7.0, evidence=False)            # gap 1 -> survives
    eng.update("x", 8.0, evidence=False)            # gap 2 -> survives
    eng.update("x", 9.0, evidence=False)            # gap 3 -> closes
    # one continuous event [0, 6], NOT two fragments
    assert tups(eng.finalize()) == [(0.0, 6.0, "x")]


def test_merge_of_same_class_segments():
    eng = TemporalEventEngine(min_on_duration=0.0, allowed_gap=0.5,
                              merge_gap=4.0, min_duration=0.0)
    for t in (0.0, 1.0, 2.0):
        eng.update("x", t, evidence=True)
    eng.update("x", 3.0, evidence=False)            # closes run [0,2]
    eng.update("x", 4.0, evidence=False)
    for t in (5.0, 6.0, 7.0):
        eng.update("x", t, evidence=True)
    eng.update("x", 8.0, evidence=False)            # closes run [5,7]
    # gap 5-2=3 <= merge_gap 4 -> one segment
    assert tups(eng.finalize()) == [(0.0, 7.0, "x")]


def test_no_merge_when_gap_too_large():
    eng = TemporalEventEngine(min_on_duration=0.0, allowed_gap=0.5,
                              merge_gap=1.0, min_duration=0.0)
    for t in (0.0, 1.0, 2.0):
        eng.update("x", t, evidence=True)
    eng.update("x", 3.0, evidence=False)            # closes [0,2]
    for t in (5.0, 6.0, 7.0):
        eng.update("x", t, evidence=True)
    eng.update("x", 8.0, evidence=False)            # closes [5,7]
    # gap 3 > merge_gap 1 -> kept separate
    assert tups(eng.finalize()) == [(0.0, 2.0, "x"), (5.0, 7.0, "x")]


def test_threshold_hysteresis():
    eng = TemporalEventEngine(min_on_duration=0.0, allowed_gap=0.0,
                              merge_gap=1.0, min_duration=0.0,
                              threshold_on=0.8, threshold_off=0.5)
    eng.update("x", 0.0, score=0.70)    # idle: below on-threshold -> nothing
    eng.update("x", 1.0, score=0.90)    # crosses on  -> run starts (also at 1.0? end=1.0)
    eng.update("x", 2.0, score=0.60)    # in run: still >= off (0.5) -> continues
    eng.update("x", 3.0, score=0.40)    # in run: below off -> absent -> closes
    segs = eng.finalize()
    # without hysteresis 0.60 at t=2 would end it; hysteresis keeps [1,2]
    assert tups(segs) == [(1.0, 2.0, "x")]


def test_hysteresis_stays_off_after_end():
    eng = TemporalEventEngine(min_on_duration=0.0, allowed_gap=0.0,
                              merge_gap=1.0, min_duration=0.0,
                              threshold_on=0.8, threshold_off=0.5)
    eng.update("x", 1.0, score=0.60)    # run closed, idle again: 0.6 < 0.8 -> nothing
    assert eng.finalize() == []


def test_threshold_off_defaults_to_on():
    eng = TemporalEventEngine(min_on_duration=0.0, allowed_gap=0.0,
                              merge_gap=1.0, min_duration=0.0,
                              threshold_on=0.8, threshold_off=None)
    eng.update("x", 0.0, score=0.90)
    eng.update("x", 1.0, score=0.85)    # in run: >= on-threshold (0.8) -> continues
    eng.update("x", 2.0, score=0.70)    # in run: below on-threshold -> closes
    segs = eng.finalize()
    assert tups(segs) == [(0.0, 1.0, "x")]


def test_score_path_disabled_when_thresholds_none():
    eng = TemporalEventEngine(min_on_duration=0.0, allowed_gap=0.1,
                              merge_gap=0.0, min_duration=0.0)
    eng.update("x", 0.0, score=0.99)    # thresholds None -> score ignored
    eng.update("x", 1.0, evidence=False)
    assert eng.finalize() == []


def test_same_class_non_overlap():
    # overlapping/adjacent same-class segments must collapse into one: the
    # engine merges anything within merge_gap, so the output never overlaps.
    eng = TemporalEventEngine(min_on_duration=0.0, allowed_gap=0.4,
                              merge_gap=10.0, min_duration=0.0)
    for t in (0.0, 1.0, 2.0):
        eng.update("same", t, evidence=True)
    eng.update("same", 3.0, evidence=False)         # closes [0,2]
    for t in (3.5, 4.0, 5.0):
        eng.update("same", t, evidence=True)        # a second candidate
    eng.update("same", 6.0, evidence=False)         # closes [3.5,5]
    segs = eng.finalize()
    assert tups(segs) == [(0.0, 5.0, "same")]       # merged, not overlapping
    # generic invariant: no two same-label segments overlap
    from collections import defaultdict
    by = defaultdict(list)
    for s in segs:
        by[s.label].append(s)
    for label, ss in by.items():
        ss.sort(key=lambda x: x.start)
        for (a, b) in zip(ss, ss[1:]):
            assert b.start >= a.end, f"{label} overlaps"


def test_simultaneous_different_classes():
    eng = TemporalEventEngine(min_on_duration=0.0, allowed_gap=10.0,
                              merge_gap=0.0, min_duration=0.0)
    for t in range(6):                               # A alone 0..3
        eng.update("red_light", float(t), evidence=True)
    for t in range(2, 8):                            # B coexists 2..5+
        eng.update("near_miss", float(t), evidence=True)
    segs = eng.finalize()
    # different classes coexist; near_miss is flushed to its last active 7.0,
    # red_light to its last active 5.0; output sorted by (label, start)
    assert tups(segs) == [(2.0, 7.0, "near_miss"), (0.0, 5.0, "red_light")]


def test_reset():
    eng = TemporalEventEngine(min_on_duration=0.0, allowed_gap=0.1,
                              merge_gap=1.0, min_duration=0.0)
    eng.update("x", 0.0, evidence=True)
    eng.update("x", 1.0, evidence=True)
    eng.update("x", 1.5, evidence=False)             # closes [0,1]
    assert tups(eng.finalize()) == [(0.0, 1.0, "x")]
    eng.reset()
    assert eng.finalize() == []                      # fresh video, nothing left
    eng.update("x", 0.0, evidence=True)
    eng.update("x", 1.0, evidence=True)
    eng.update("x", 1.5, evidence=False)
    assert tups(eng.finalize()) == [(0.0, 1.0, "x")]  # replay identical


def test_finalize_flushes_active_run():
    eng = TemporalEventEngine(min_on_duration=0.0, allowed_gap=100.0,
                              merge_gap=0.0, min_duration=0.0)
    for t in range(4):
        eng.update("x", float(t), evidence=True)     # never absent -> still open
    segs = eng.finalize()
    assert tups(segs) == [(0.0, 3.0, "x")]           # flushed to last active frame
    assert tups(eng.finalize()) == [(0.0, 3.0, "x")]  # idempotent


def test_causality_confirmation_uses_past_only():
    eng = TemporalEventEngine(min_on_duration=1.0, allowed_gap=100.0,
                              merge_gap=0.0, min_duration=0.0)
    eng.update("x", 0.0, evidence=True)
    # mid-stream decision: only 0.0 seen, never confirmed -> nothing yet
    assert tups(eng.finalize()) == []
    # continuing from a NEW run (0.0 was flushed as unconfirmed = discarded)
    for t in (1.0, 2.0, 3.0):
        eng.update("x", t, evidence=True)
    segs = eng.finalize()
    # confirmed from its own real start 1.0 -- the earlier 0.0 blip did NOT leak
    assert tups(segs) == [(1.0, 3.0, "x")]


def test_causality_monotonic_guard():
    eng = TemporalEventEngine()
    eng.update("x", 5.0, evidence=True)
    try:
        eng.update("x", 3.0, evidence=False)
    except ValueError:
        return
    raise AssertionError("non-monotonic timestamps must raise ValueError")


def test_deterministic_repeated_sequence():
    feed = [
        ("x", 0.0, True), ("x", 0.5, True), ("x", 1.0, False),
        ("y", 2.0, True), ("y", 2.5, True), ("x", 3.0, True),
        ("x", 3.5, False), ("y", 4.0, False),
    ]
    results = set()
    for _ in range(3):
        eng = TemporalEventEngine(min_on_duration=0.0, allowed_gap=0.6,
                                  merge_gap=1.0, min_duration=0.5)
        for label, t, ev in feed:
            eng.update(label, t, evidence=ev)
        results.add(tuple(tups(eng.finalize())))
        # same engine, replayed identically after reset
        eng.reset()
        for label, t, ev in feed:
            eng.update(label, t, evidence=ev)
        results.add(tuple(tups(eng.finalize())))
    assert len(results) == 1


def test_per_label_overrides():
    eng = TemporalEventEngine(min_on_duration=1.0, allowed_gap=0.1,
                              merge_gap=1.0, min_duration=1.0,
                              per_label={
                                  "fast": {"min_on_duration": 0.0,
                                           "min_duration": 0.0},
                              })
    # global min_on_duration=1.0 confirms "slow" only after 1 s of evidence
    eng.update("slow", 0.0, evidence=True)
    eng.update("slow", 0.5, evidence=True)
    eng.update("slow", 1.0, evidence=False)          # closed while unconfirmed
    # "fast" overrides: instant confirmation + no min_duration
    eng.update("fast", 0.0, evidence=True)
    eng.update("fast", 0.8, evidence=True)
    eng.update("fast", 1.0, evidence=False)
    segs = eng.finalize()
    assert tups(segs) == [(0.0, 0.8, "fast")]
    # and the 1-s "slow" run survives since it >= global min_duration
    eng2 = TemporalEventEngine(min_on_duration=1.0, allowed_gap=0.1,
                              merge_gap=1.0, min_duration=1.0)
    for t in (0.0, 0.6, 1.2):
        eng2.update("slow", t, evidence=True)
    eng2.update("slow", 1.3, evidence=False)
    assert tups(eng2.finalize()) == [(0.0, 1.2, "slow")]


def test_disabled_params_via_none():
    # min_on_duration=None + allowed_gap=None: instant start, instant end
    eng = TemporalEventEngine(min_on_duration=None, allowed_gap=None,
                              merge_gap=None, min_duration=None)
    eng.update("x", 0.0, evidence=True)
    eng.update("x", 1.0, evidence=True)
    eng.update("x", 2.0, evidence=False)
    assert tups(eng.finalize()) == [(0.0, 1.0, "x")]

    # merge_gap=None -> never merge
    eng2 = TemporalEventEngine(min_on_duration=None, allowed_gap=None,
                               merge_gap=None, min_duration=None)
    eng2.update("x", 0.0, evidence=True);  eng2.update("x", 1.0, evidence=True)
    eng2.update("x", 2.0, evidence=False)
    eng2.update("x", 5.0, evidence=True);  eng2.update("x", 6.0, evidence=True)
    eng2.update("x", 7.0, evidence=False)
    assert tups(eng2.finalize()) == [(0.0, 1.0, "x"), (5.0, 6.0, "x")]


def test_event_segment_unpacking():
    s = EventSegment("accident", 1.2, 3.4)
    start, end, label = s
    assert (start, end, label) == (1.2, 3.4, "accident")
    assert s.to_list() == [1.2, 3.4, "accident"]
    assert abs(s.duration - 2.2) < 1e-9


def test_zero_duration_segments_dropped():
    # a single confirmed frame would be [0, 0]; the engine instead reports
    # nothing (the harness needs 0 <= start < end)
    eng = TemporalEventEngine(min_on_duration=0.0, allowed_gap=10.0,
                              merge_gap=0.0, min_duration=0.0)
    eng.update("x", 0.0, evidence=True)
    assert eng.finalize() == []


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"OK: {len(tests)} tests passed")


if __name__ == "__main__":
    main()