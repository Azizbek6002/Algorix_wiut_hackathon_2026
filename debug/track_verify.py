"""Verify ByteTrack id persistence + feed TrajectoryEngine (PHASE 1-3 check).

Run from the package dir:  python debug/track_verify.py [video] [max_frames] [stride]

GPU inference, one detection pass per sampled frame.
"""

from __future__ import annotations

import json
import os
import sys

import cv2

PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PKG_DIR not in sys.path:
    sys.path.insert(0, PKG_DIR)

from src.detector import Detector
from src.trajectory import Detection, TrajectoryEngine, frame_presence_stats

DEFAULT_VIDEO = r"C:\Projects\Traffic Computer Vision\video\C3905.MP4"
IMGSZ = 800
CONF = 0.25
DEVICE = "cuda:0"


def main():
    video = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_VIDEO
    max_frames = int(sys.argv[2]) if len(sys.argv) > 2 else 90
    stride = int(sys.argv[3]) if len(sys.argv) > 3 else 3

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        print("cannot open video"); sys.exit(1)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    small_h = max(1, int(IMGSZ * H / W))
    sx, sy = W / IMGSZ, H / small_h

    det = Detector(model_path=os.path.join(PKG_DIR, "weights", "yolo11x.pt"),
                   conf=CONF, device=DEVICE, imgsz=IMGSZ)
    eng = TrajectoryEngine(keep_sec=6.0)
    id_seqs: list[list[int]] = []

    idx = 0
    while True:
        ok, fr = cap.read()
        if not ok or (max_frames and idx >= max_frames):
            break
        if idx % stride != 0:
            idx += 1
            continue
        small = cv2.resize(fr, (IMGSZ, small_h))
        dets_raw = det.track(small, persist=True)
        dets = []
        for d in dets_raw:
            x1, y1, x2, y2 = d["xyxy"]
            d = dict(d)
            d["xyxy"] = (x1 * sx, y1 * sy, x2 * sx, y2 * sy)
            dets.append(Detection.from_dict(d))
        eng.update(dets, idx / fps)
        id_seqs.append(sorted({dt.tid for dt in dets if dt.tid is not None}))
        idx += 1
    cap.release()

    t_end = (idx - 1) / fps if idx else 0.0
    stats = frame_presence_stats(id_seqs)

    tracks = eng.active()
    lines = []
    for tr in tracks:
        pts = [p for p in tr.points if p.t <= t_end]
        first, last = pts[0], pts[-1]
        lines.append({
            "track_id": tr.track_id, "class": tr.label, "n_points": len(pts),
            "t_first": round(first.t, 3), "t_last": round(last.t, 3),
            "last_center": (round(last.x, 1), round(last.y, 1)),
            "last_bbox": tuple(round(v, 1) for v in last.xyxy),
        })

    longest = max(tracks, key=lambda tr: sum(1 for p in tr.points if p.t <= t_end),
                  default=None)
    report = {
        "config": {"video": os.path.basename(video), "frames_to": idx,
                   "stride": stride, "imgsz": IMGSZ, "device": DEVICE,
                   "W": W, "H": H, "fps": round(fps, 2)},
        "stability": stats,
        "summary_tracks": lines,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))

    print("\n=== EXAMPLE: longest-lived track ===")
    if longest is not None:
        print(f"track_id: {longest.track_id}")
        print(f"class:    {longest.label}")
        for p in longest.points:
            print(f"  t={p.t:.3f}  position=({p.x:.1f}, {p.y:.1f})  "
                  f"bottom=({p.x:.1f}, {p.bottom_y:.1f})  bbox={tuple(round(v,1) for v in p.xyxy)}")


if __name__ == "__main__":
    main()