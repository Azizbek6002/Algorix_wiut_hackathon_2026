"""Deterministic unit tests for src/trajectory.py (no pytest dependency).

Run:  python tests/test_trajectory.py
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.trajectory import Detection, TrajectoryEngine, frame_presence_stats


def _det(x1, y1, x2, y2, conf, label, tid):
    return Detection(xyxy=(x1, y1, x2, y2), conf=conf, label=label, tid=tid)


def test_detection_geometry():
    d = _det(10, 20, 30, 60, 0.9, "car", 1)
    assert d.center == (20.0, 40.0)
    assert d.bottom_center == (20.0, 60.0)
    assert d.area == 20.0 * 40.0
    assert d.width == 20.0 and d.height == 40.0


def test_engine_single_track_history():
    eng = TrajectoryEngine(keep_sec=4.0)
    eng.update([_det(0, 0, 10, 20, 0.9, "car", 17)], t_sec=10.0)
    eng.update([_det(5, 5, 15, 25, 0.8, "car", 17)], t_sec=10.1)
    eng.update([_det(10, 10, 20, 30, 0.7, "car", 17)], t_sec=10.2)
    active = eng.active()
    assert len(active) == 1
    tr = active[0]
    assert tr.track_id == 17 and tr.label == "car"
    assert [p.t for p in tr.points] == [10.0, 10.1, 10.2]
    assert tr.last.xyxy == (10.0, 10.0, 20.0, 30.0)


def test_engine_prunes_stale_track():
    eng = TrajectoryEngine(keep_sec=2.0)
    eng.update([_det(0, 0, 10, 20, 0.9, "car", 1)], t_sec=0.0)
    eng.update([_det(0, 0, 10, 20, 0.9, "car", 2)], t_sec=0.1)
    assert len(eng.active()) == 2
    eng.update([_det(0, 0, 10, 20, 0.9, "car", 3)], t_sec=5.0)
    assert [tr.track_id for tr in eng.active()] == [3]


def test_history_window_retention():
    tr_engine = TrajectoryEngine(keep_sec=10.0)
    for i in range(6):
        t = float(i)
        tr_engine.update([_det(i * 2, 0, i * 2 + 10, 20, 0.9, "car", 5)], t_sec=t)
    tr = tr_engine.get(5)
    assert len(tr.points) == 6
    window = tr.recent(window_sec=4.0)
    assert len(window) == 5
    assert all(p.t >= 1.0 for p in window)


def test_frame_presence_stats():
    seq = [[1, 2], [1, 3], [2], [1, 2, 3], [1, 2, 3]]
    st = frame_presence_stats(seq)
    assert st["unique_ids"] == 3
    assert st["appeared_once_ids"] == []
    assert st["churn_share"] == 0.0
    st2 = frame_presence_stats([[9], [9], [8]])
    assert st2["appeared_once_ids"] == [8]
    assert st2["churn_share"] == 0.5


def test_determinism():
    feeds = [
        [(10, 0, 20, 30, 0.9, "car", 1)],
        [(11, 1, 21, 31, 0.9, "car", 1), (5, 5, 10, 20, 0.8, "person", 2)],
        [(12, 2, 22, 32, 0.9, "car", 1)],
    ]
    def run_once():
        eng = TrajectoryEngine(keep_sec=10.0)
        for t, batch in enumerate(feeds):
            dets = [_det(*b) for b in batch]
            eng.update(dets, float(t))
        out = []
        for tr in eng.active():
            out.append([(p.t, p.x, p.y, p.bottom_y) for p in tr.points])
        return out

    assert run_once() == run_once()


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"OK: {len(tests)} tests passed")


if __name__ == "__main__":
    main()