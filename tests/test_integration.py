"""Integration test for solution.detect_events — no GPU, no weights.

Drives the REAL pipeline end-to-end (VideoReader -> mocked Detector ->
EventManager -> legacy flag engine -> post-processing) over a tiny synthetic
.mp4. The detections are hand-crafted so the legacy rule engine must emit
`stopped_vehicle` and `wrong_way`, proving the whole wiring works and that the
output obeys the harness contract.

Also runs the optional PHASE detector path (TCV_ENABLE_PHASE_DETECTORS=1) to
make sure trajectory/motion/interaction + geometry + all four PHASE detectors
execute without error over the same mocked feed.

Run:  python tests/test_integration.py
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np  # noqa: E402

_FPS = 10.0
_N_FRAMES = 320          # 32 s
_W, _H = 640, 360


def make_video(path: str) -> dict:
    import cv2
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(path, fourcc, _FPS, (_W, _H))
    frame = np.zeros((_H, _W, 3), dtype=np.uint8)
    for _ in range(_N_FRAMES):
        vw.write(frame)
    vw.release()
    return {"fps": _FPS, "width": _W, "height": _H, "n_frames": _N_FRAMES,
            "duration": _N_FRAMES / _FPS}


class FakeDetector:
    """Yields deterministic tracked detections, indexed by call order."""

    imgsz = 320

    def __init__(self):
        self.calls = 0

    def track(self, frame_bgr, persist: bool = True) -> list[dict]:
        i = self.calls
        self.calls += 1
        # track 1: static car (bottom-center ~(330,340)) -> stopped_vehicle >= 10 s
        d1 = {"xyxy": (300.0, 300.0, 360.0, 340.0), "conf": 0.9,
              "label": "car", "id": 1}
        # track 2: car moving right for the first ~1 s (heading 0 deg, dev 165
        # from dominant flow 195) -> wrong_way while moving
        offset = 5.0 * min(i, 10)
        d2 = {"xyxy": (100.0 + offset, 250.0, 160.0 + offset, 290.0),
              "conf": 0.9, "label": "car", "id": 2}
        return [d1, d2]


def run_with_fake(fake: FakeDetector) -> list[list]:
    import src.pipeline.pipeline as pp
    prev = pp._MODEL
    pp._MODEL = fake
    try:
        from solution import detect_events
        return detect_events(video_path)
    finally:
        pp._MODEL = prev


video_path = None
_tmpdir = None


def setup_module(module):
    global video_path, _tmpdir
    _tmpdir = tempfile.mkdtemp(prefix="tcv_test_")
    video_path = os.path.join(_tmpdir, "synthetic.mp4")
    make_video(video_path)


def teardown_module(module):
    if _tmpdir:
        shutil.rmtree(_tmpdir, ignore_errors=True)


# ---- tests -----------------------------------------------------------------
def test_integration_legacy_pipeline_emits_expected_events():
    from _validation import validate_events
    events = run_with_fake(FakeDetector())
    assert validate_events(events, _N_FRAMES / _FPS) == []
    labels = {ev[2] for ev in events}
    assert "stopped_vehicle" in labels, f"expected stopped_vehicle, got {events}"
    assert "wrong_way" in labels, f"expected wrong_way, got {events}"
    stopped = [ev for ev in events if ev[2] == "stopped_vehicle"]
    wrong = [ev for ev in events if ev[2] == "wrong_way"]
    # stationary car must sit for >= 10 s; the fake moves for 10 sampled calls
    # (the detector is call-indexed, not frame-indexed), so the wrong_way run
    # is a few seconds long regardless of stride
    assert max(e - s for s, e, _ in stopped) >= 10.0
    assert all(0.5 <= e - s <= 5.0 for s, e, _ in wrong)


def test_integration_deterministic():
    first = run_with_fake(FakeDetector())
    second = run_with_fake(FakeDetector())
    assert first == second


def test_integration_phase_detectors_run_clean():
    """TCV_ENABLE_PHASE_DETECTORS=1 must run trajectory/motion/interaction +
    geometry + all four PHASE detectors without error over the same feed."""
    os.environ["TCV_ENABLE_PHASE_DETECTORS"] = "1"
    try:
        from solution import detect_events
        events = detect_events(video_path)
    finally:
        del os.environ["TCV_ENABLE_PHASE_DETECTORS"]
    from _validation import validate_events
    assert validate_events(events, _N_FRAMES / _FPS) == []


def main():
    setup()
    try:
        tests = [v for k, v in sorted(globals().items())
                 if k.startswith("test_")]
        for fn in tests:
            fn()
            print(f"PASS {fn.__name__}")
        print(f"OK: {len(tests)} tests passed")
    finally:
        teardown()


if __name__ == "__main__":
    main()