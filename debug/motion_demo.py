"""Demonstrate MotionEngine on the real recorded track #2 (stationary car).

Requires the output of `python debug/track_verify.py` (track_verify_out.txt).
Run:  python debug/motion_demo.py
"""

import os
import re
import sys

PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PKG_DIR not in sys.path:
    sys.path.insert(0, PKG_DIR)

from src.motion import MotionEngine
from src.trajectory import TrackTrajectory, TrajectoryPoint


def load_track(out_file):
    with open(out_file, "rb") as f:
        raw = f.read()
    try:
        data = raw.decode("utf-8")
        if "\x00" in data:
            raise UnicodeDecodeError("utf-8", raw, 0, 1, "nul bytes")
    except UnicodeDecodeError:
        data = raw.decode("utf-16")
    m = re.search(r"=== EXAMPLE[^\n]*\n.*?\n", data, re.S)
    if not m:
        raise SystemExit("no example block; run debug/track_verify.py first")
    pat = re.compile(
        r"t=([\d.]+)\s+position=\(([\d.]+), ([\d.]+)\)\s+bottom=\(([\d.]+), ([\d.]+)\)")
    tr = TrackTrajectory(track_id=2, label="car", keep_sec=6.0)
    for t, x, y, bx, by in pat.findall(data):
        t, x, y = float(t), float(x), float(y)
        bx, by = float(bx), float(by)
        tr.append(TrajectoryPoint(t=t, x=x, y=y, bottom_y=by,
                                  xyxy=(x - 100, y - 90, x + 100, by), conf=0.9))
    return tr


def main():
    out_file = os.path.join(PKG_DIR, "track_verify_out.txt")
    tr = load_track(out_file)
    eng = MotionEngine(window_s=0.6, min_points=3, noise_px=4.0)
    print("track_id: 2  class: car  (0.0-0.9s moving, then stationary)\n")
    print(f"{'t':>6} {'speed_px/s':>10} {'heading':>8} {'accel':>9} {'state':>11} {'quality':>8}")
    for p in tr.points:
        st = eng.update(tr, p.t)
        if st is None:
            print(f"{p.t:6.2f} {'-':>10} {'-':>8} {'-':>9} {'warmup':>11} {'-':>8}")
            continue
        hdg = f"{st.heading_deg:.1f}" if st.heading_deg is not None else "-"
        acc = f"{st.accel:.1f}" if st.accel is not None else "-"
        print(f"{st.t:6.2f} {st.speed:10.1f} {hdg:>8} {acc:>9} "
              f"{'STATIONARY' if st.stationary else 'moving':>11} {st.quality:8.2f}")


if __name__ == "__main__":
    main()