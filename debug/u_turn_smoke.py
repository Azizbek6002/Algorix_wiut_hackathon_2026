"""illegal_u_turn smoke test on a real video (YOLO11x + ByteTrack + full stack).

    python debug/u_turn_smoke.py [video] [stride] [max_frames]

Defaults: C3905.MP4, stride 3, frame cap 0 (full video). Prints every detected
illegal_u_turn candidate with its full per-track evidence: start/end, track id,
duration, initial and final (window-entry / window-exit) headings, heading
change, zone flag, and speed / movement evidence.

CANDIDATES ARE NOT CONFIRMED GROUND TRUTH: visual verification on the video is
required before trusting them (the calibrated u-turn zones are still estimates).

Note: GPU inference dominates the runtime (see TCV_DEVICE/TCV_IMGSZ envs).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2

from src.detector import Detector
from src.geometry import Geometry
from src.illegal_u_turn import IllegalUTurnDetector
from src.motion import MotionEngine
from src.trajectory import Detection, TrajectoryEngine
from src.wrong_way import DEFAULT_VEHICLE_LABELS  # noqa: E402

VIDEO_DIR = r"C:\Projects\Traffic Computer Vision\video"
SCENE_CFG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "scene_config.json")

IMG_SZ = int(os.environ.get("TCV_IMGSZ", "800"))
CONF = float(os.environ.get("TCV_CONF", "0.25"))
ALLOWED_GAP = 0.6


def _close(cand, candidates):
    if cand and cand["frames"]:
        cand["end"] = cand["last"]
        candidates.append(cand)
        return {}
    return cand


def main() -> int:
    video = sys.argv[1] if len(sys.argv) > 1 else os.path.join(VIDEO_DIR, "C3905.MP4")
    stride = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    max_frames = int(sys.argv[3]) if len(sys.argv) > 3 else 0

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        print("cannot open", video)
        return 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"video={os.path.basename(video)} {W}x{H} fps={fps:.2f} frames={n_frames} "
          f"duration={n_frames / fps:.1f}s stride={stride}")

    geometry = Geometry.from_json(SCENE_CFG, W, H)
    print(f"u-turn zones: {len(geometry.u_turn_zones)} enabled")
    for z in geometry.u_turn_zones:
        xs = [p[0] for p in z]
        ys = [p[1] for p in z]
        zz = [[min(xs), min(ys)], [max(xs), max(ys)]]
        print(f"  zone bbox (ref-space): x {zz[0][0]:.0f}-{zz[1][0]:.0f} "
              f"y {zz[0][1]:.0f}-{zz[1][1]:.0f} ({len(z)} pts)")

    det = Detector(model_path=r"weights/yolo11x.pt", conf=CONF,
                   device=os.environ.get("TCV_DEVICE", "cuda:0"), imgsz=IMG_SZ)
    traj = TrajectoryEngine(keep_sec=4.0)
    motion = MotionEngine(window_s=0.8, noise_px=4.0, min_points=3)
    ut = IllegalUTurnDetector()

    candidates: list[dict] = []
    cand: dict = {}
    reasons: dict[str, int] = {}
    n_active_frames = 0
    idx = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if idx % stride != 0:
            idx += 1
            continue
        if max_frames and idx >= max_frames:
            break
        small = cv2.resize(fr, (IMG_SZ, max(1, int(IMG_SZ * H / W))))
        sx, sy = W / IMG_SZ, H / small.shape[0]
        dets = det.track(small, persist=True)
        t_sec = idx / fps

        scaled = []
        for d in dets:
            x1, y1, x2, y2 = d["xyxy"]
            scaled.append(Detection(xyxy=(x1 * sx, y1 * sy, x2 * sx, y2 * sy),
                                    conf=d["conf"], label=d["label"], tid=d["id"]))
        traj.update(scaled, t_sec)

        tracks = {tr.track_id: tr for tr in traj.active()}
        states = {}
        for tr in tracks.values():
            if tr.label not in DEFAULT_VEHICLE_LABELS:
                continue
            st = motion.update(tr, t_sec)
            if st is not None:
                states[tr.track_id] = st

        report = ut.update(tracks, states, geometry, t_sec)
        for rec in report["tracks"].values():
            reasons[rec["reason"]] = reasons.get(rec["reason"], 0) + 1

        for tid in list(cand.keys()):
            if tid not in report["active_tracks"]:
                if t_sec - cand[tid]["last"] >= ALLOWED_GAP:
                    cand[tid] = _close(cand[tid], candidates)
        for tid in report["active_tracks"]:
            rec = report["tracks"][tid]
            c = cand.setdefault(tid, {"tid": tid, "start": None, "last": None,
                                      "frames": [], "label": tracks[tid].label})
            if c["start"] is None:
                c["start"] = t_sec
            c["last"] = t_sec
            c["frames"].append({
                "t": t_sec, "entry": rec["entry_heading_deg"],
                "exit": rec["exit_heading_deg"], "reversal": rec["reversal_deg"],
                "cum": rec["cum_turn_deg"], "arc": rec["arc_px"],
                "speed": rec["speed_px_s"], "in_zone": rec["in_zone"],
                "n": rec["samples"]})
        if report["evidence"]:
            n_active_frames += 1
        idx += 1
        if idx % 300 == 0:
            print(f"  ... {idx} frames, {n_active_frames} u-turn frames")
    cap.release()

    for tid in list(cand.keys()):
        cand[tid] = _close(cand[tid], candidates)

    segs = ut.finalize()
    candidates.sort(key=lambda c: c["start"])
    print(f"\nper-track evaluation reasons: {reasons}")
    print(f"illegal_u_turn evidence frames: {n_active_frames}")
    print("events:")
    for s in segs:
        print(f"  [{round(s.start, 3)}, {round(s.end, 3)}] illegal_u_turn")
    if not segs:
        print("  (none)")

    print("\ncandidate U-turns (per sustained evidence run):")
    if not candidates:
        print("  (none)")
    for c in candidates:
        fr = c["frames"]
        peak = max(fr, key=lambda f: f["reversal"] or 0.0)
        mid = fr[len(fr) // 2]
        print(
            f"  tid={c['tid']:<4} label={c['label']:<9} "
            f"[{c['start']:.2f}, {c['end']:.2f}] dur={c['end'] - c['start']:.2f}s "
            f"frames={len(fr)}")
        print(
            f"      initial_h={peak['entry']:.0f} final_h={peak['exit']:.0f} "
            f"reversal={peak['reversal']:.0f}"
            f" cum_turn={peak['cum']:.0f} arc={peak['arc']:.0f}px "
            f"speed={mid['speed']:.0f}px/s zone={mid['in_zone']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())