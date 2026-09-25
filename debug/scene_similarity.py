"""Numeric check: do the four sample videos plausibly share one fixed camera?

The samples are treated as ONE fixed camera by the team (same resolution
3840x2160, geometry calibrated once into a shared camera-level scene_config).
This script is a crude proxy (64x36 grids of a single mid frame): if the
static top-strip correlation between videos is low, verify the shared
geometry visually against a real frame of each sample before trusting the
topology (differences may be day/season/weather content, not a different
camera). It is a WARNING gadget, not the arbiter: the default is the shared
camera-level config.

Run from the package dir:  python debug/scene_similarity.py
"""

from __future__ import annotations

import os
import sys

import cv2
import numpy as np

VIDEO_DIR = r"C:\Projects\Traffic Computer Vision\video"
VIDEOS = ["C3896.MP4", "C3897.MP4", "C3902.MP4", "C3905.MP4"]
GRID_W, GRID_H = 64, 36
TOP_ROWS = 8  # upper ~22% of the frame: static backdrop, no traffic


def load_mid(path: str) -> np.ndarray | None:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    mid = max(0, total // 2)
    cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
    ok, fr = cap.read()
    cap.release()
    if not ok:
        return None
    gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (GRID_W, GRID_H)).astype(np.float32)


def corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel() - a.mean()
    b = b.ravel() - b.mean()
    den = np.linalg.norm(a) * np.linalg.norm(b)
    if den < 1e-9:
        return 0.0
    return float(np.dot(a, b) / den)


def main():
    samples = {n: load_mid(os.path.join(VIDEO_DIR, n)) for n in VIDEOS}
    if any(g is None for g in samples.values()):
        sys.exit("failed to load mid frames")
    names = VIDEOS

    print(f"grid {GRID_W}x{GRID_H}; top-strip = rows 0..{TOP_ROWS - 1}\n")
    for region, rows in (("full-frame", None), ("top-strip (static)", slice(0, TOP_ROWS))):
        print(f"pairwise {region} correlation (mid frames):")
        print(f"{'':>10}" + "".join(f"{n[:6]:>9}" for n in names))
        for a in names:
            row = f"{a[:6]:>10}"
            ga = samples[a] if rows is None else samples[a][rows]
            for b in names:
                if a == b:
                    row += f"{'1.000':>9}"
                    continue
                gb = samples[b] if rows is None else samples[b][rows]
                row += f"{corr(ga, gb):9.3f}"
            print(row)
        print()

    top = [corr(samples[a][slice(0, TOP_ROWS)], samples[b][slice(0, TOP_ROWS)])
           for i, a in enumerate(names) for b in names[i + 1:]]
    print("conclusion (top-strip = static backdrop):")
    if top:
        print(f"  min cross-video static-backdrop correlation = {min(top):.3f}")
        print("  default: ONE fixed camera -> SHARED camera-level scene_config")
        print("  => " + ("shared geometry plausible"
                         if min(top) >= 0.85 else
                         "caution: backdrop differs between samples -> compare the "
                         "shared geometry visually on a real frame of each video "
                         "(road/lanes alignment) before trusting it; weather/season/"
                         "daylight content also lowers correlation"))


if __name__ == "__main__":
    main()