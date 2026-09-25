"""wrong_way smoke test on a real video (YOLO11x + ByteTrack + full stack).

    python debug/wrong_way_smoke.py [video] [stride] [max_frames]

Defaults: C3905.MP4, stride 3 (= what run_submission uses), no frame cap.
Prints vehicle heading histogram (motion convention) to sanity-check the
lane-direction convention, plus per-lane stats and the final events.

Note: GPU inference dominates the runtime (see TCV_DEVICE/TCV_IMGSZ envs).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2

from src.detector import Detector
from src.geometry import Geometry
from src.motion import MotionEngine
from src.trajectory import Detection, TrajectoryEngine
from src.wrong_way import (  # noqa: E402
    DEFAULT_VEHICLE_LABELS,
    WrongWayDetector,
    lane_direction_to_motion_deg,
)

VIDEO_DIR = r"C:\Projects\Traffic Computer Vision\video"
SCENE_CFG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "scene_config.json")

IMG_SZ = int(os.environ.get("TCV_IMGSZ", "800"))
CONF = float(os.environ.get("TCV_CONF", "0.25"))


def histogram(values, low=0.0, high=360.0, bins=8):
    step = (high - low) / bins
    out = [0] * bins
    for v in values:
        idx = min(bins - 1, int((v % 360.0) / step))
        out[idx] += 1
    return out


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
    for lane in geometry.lanes:
        img_dir = lane["expected_direction"]
        print(f"  lane {lane['lane_id']}: expected(image)={img_dir} "
              f"-> motion(math)={lane_direction_to_motion_deg(img_dir):.0f}")

    det = Detector(model_path=r"weights/yolo11x.pt", conf=CONF,
                   device=os.environ.get("TCV_DEVICE", "cuda:0"), imgsz=IMG_SZ)
    traj = TrajectoryEngine(keep_sec=4.0)
    motion = MotionEngine(window_s=0.8, noise_px=4.0, min_points=3)
    ww = WrongWayDetector()

    heads: list[float] = []
    per_lane = {}
    reasons: dict[str, int] = {}
    evidence_details: list[dict] = []
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
            if st is not None and not st.stationary and st.speed > 1.0:
                heads.append(st.heading_deg)
                lane = geometry.get_lane((tr.last.x, tr.last.bottom_y))
                per_lane.setdefault(lane, []).append(st.heading_deg)

        report = ww.update(tracks, states, geometry, t_sec)
        for rec in report["tracks"].values():
            reasons[rec["reason"]] = reasons.get(rec["reason"], 0) + 1
        for tid in report["active_tracks"]:
            rec = report["tracks"][tid]
            evidence_details.append({
                "t": t_sec, "tid": tid, "label": tracks[tid].label,
                "lane": rec["lane"], "heading": rec["heading_deg"],
                "expected": rec["expected_deg"], "dev": rec["deviation_deg"],
                "speed": rec["speed_px_s"],
            })
        if report["evidence"]:
            n_active_frames += 1
        idx += 1
        if idx % 300 == 0:
            print(f"  ... {idx} frames, {len(heads)} vehicle headings, "
                  f"{n_active_frames} wrong_way frames")
    cap.release()

    print(f"\nvehicle headings (motion/math convention) histogram 45-deg bins:")
    for i, c in enumerate(histogram(heads)):
        print(f"    heading {i * 45:>3}-{i * 45 + 45:>3}: {c}")
    for lane, hs in sorted(per_lane.items(), key=lambda kv: (kv[0] is None, kv[0] or "")):
        print(f"  lane {lane!r}: {len(hs)} headings, 45-deg bins {histogram(hs)}")

    segs = ww.finalize()
    print(f"\nper-track evaluation reasons: {reasons}")
    print(f"wrong_way evidence frames: {n_active_frames}")
    print("events:")
    for s in segs:
        print(f"  [{s.start:.2f}, {s.end:.2f}] wrong_way")
    if not segs:
        print("  (none)")

    print("\nevidence track details:")
    for rec in evidence_details:
        print(
            f"  t={rec['t']:.2f} tid={rec['tid']} lane={rec['lane']!r} "
            f"heading={rec['heading']:.0f} expected={rec['expected']:.0f} "
            f"dev={rec['dev']:.0f} speed={rec['speed']:.1f} label={rec['label']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())