"""Per-lane motion-direction calibration diagnostics (READ-ONLY).

Computes, for each lane, the observed real-world driving heading from tracked
vehicles (MotionEngine, already smoothed) and recommends an `expected_direction`
to store in scene_config.json (image/y-down convention: `img = (360 - h) % 360`).

Does NOT modify scene_config.json — it only prints recommendations.

    python debug/lane_direction_calibration.py [video] [stride] [max_frames]

Filtering (per Phase 9.1 spec): ignores stationary, low-speed, vehicles outside
any lane, vehicles inside U-turn zones, and low-quality motion states.
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2

from src.detector import Detector
from src.geometry import Geometry
from src.motion import MotionEngine
from src.trajectory import Detection, TrajectoryEngine
from src.wrong_way import DEFAULT_VEHICLE_LABELS, WrongWayDetector

VIDEO_DIR = r"C:\Projects\Traffic Computer Vision\video"
SCENE_CFG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "scene_config.json")

IMG_SZ = int(os.environ.get("TCV_IMGSZ", "800"))
CONF = float(os.environ.get("TCV_CONF", "0.25"))
LOW_SPEED_PX_S = 5.0  # below this the heading is unreliable

veh = WrongWayDetector().vehicle_labels


def circular_mean(angles):
    s = sum(math.sin(math.radians(a)) for a in angles)
    c = sum(math.cos(math.radians(a)) for a in angles)
    return math.degrees(math.atan2(s, c)) % 360.0


def circular_median(angles):
    # grid search for the angle minimizing the sum of angular distances
    best_m, best_d = 0.0, float("inf")
    for m10 in range(3600):
        m = m10 / 10.0
        d = sum(min(abs((a - m) % 360.0), 360.0 - abs((a - m) % 360.0)) for a in angles)
        if d < best_d:
            best_d, best_m = d, m
    return best_m


def modal_circular_mean(angles):
    """Robust to counterflow: histogram peak, then circular mean of the ±80° window."""
    bins_n = 36
    hist = [0] * bins_n
    for a in angles:
        hist[int((a % 360.0) / (360.0 / bins_n)) % bins_n] += 1
    peak = max(range(bins_n), key=lambda i: hist[i])
    lo = (peak * 10.0 - 80.0) % 360.0
    hi = lo + 160.0
    inside = []
    for a in angles:
        d = (a - lo) % 360.0
        if d <= (hi - lo):
            inside.append(a)
    return circular_mean(inside), len(inside)


def to_image_dir(motion_deg):
    return (360.0 - motion_deg) % 360.0


def main() -> int:
    video = sys.argv[1] if len(sys.argv) > 1 else os.path.join(VIDEO_DIR, "C3905.MP4")
    stride = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    max_frames = int(sys.argv[3]) if len(sys.argv) > 3 else 0

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        print("cannot open", video)
        return 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"video={os.path.basename(video)} {W}x{H} fps={fps:.2f} stride={stride}")

    geometry = Geometry.from_json(SCENE_CFG, W, H)
    det = Detector(model_path=r"weights/yolo11x.pt", conf=CONF,
                   device=os.environ.get("TCV_DEVICE", "cuda:0"), imgsz=IMG_SZ)
    traj = TrajectoryEngine(keep_sec=4.0)
    motion = MotionEngine(window_s=0.8, noise_px=4.0, min_points=3)

    per_lane: dict[str, list[float]] = {}
    ignored = {"stationary": 0, "u_turn_zone": 0, "unknown_lane": 0,
               "low_speed": 0, "low_quality": 0, "not_vehicle": 0}

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

        for tr in traj.active():
            if tr.label not in veh:
                ignored["not_vehicle"] += 1
                continue
            st = motion.update(tr, t_sec)
            if st is None or st.quality < 0.3:
                ignored["low_quality"] += 1
                continue
            if st.stationary:
                ignored["stationary"] += 1
                continue
            if st.speed < LOW_SPEED_PX_S:
                ignored["low_speed"] += 1
                continue
            pos = (tr.last.x, tr.last.bottom_y)
            lane = geometry.get_lane(pos)
            if lane is None:
                ignored["unknown_lane"] += 1
                continue
            if geometry.is_in_u_turn_zone(pos):
                ignored["u_turn_zone"] += 1
                continue
            per_lane.setdefault(lane, []).append(st.heading_deg)
        idx += 1
        if idx % 600 == 0:
            print(f"  ... {idx} frames processed")
    cap.release()

    print(f"\nigored counts: {ignored}")
    print()
    lane_names = sorted(per_lane, key=lambda l: (int(l[1:]) if l[1:].isdigit() else l))
    for lane in lane_names:
        angles = sorted(per_lane[lane])
        cfg = next((l for l in geometry.lanes if l["lane_id"] == lane), None)
        cur_img = cfg["expected_direction"] if cfg else None
        cur_motion = (360.0 - cur_img) % 360.0 if cur_img is not None else None
        modal, n_win = modal_circular_mean(angles)
        print(f"{lane}:")
        print(f"  current expected_direction: {cur_img}")
        print(f"  observed heading stats (motion convention): "
              f"n={len(angles)}")
        print(f"    circular mean    : {circular_mean(angles):7.1f}")
        print(f"    circular median  : {circular_median(angles):7.1f}")
        print(f"    modal-window mean: {modal:7.1f}  (n={n_win} in ±80deg of peak)")
        print(f"  recommended image direction: {to_image_dir(modal):.1f}")
        print(f"    (current image={cur_img} -> motion {cur_motion}"
              if cur_motion is not None else "    (current image=?)")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())