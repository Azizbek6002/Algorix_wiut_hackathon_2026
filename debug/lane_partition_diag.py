"""Phase 10A diagnostics: L0/L3 overlap analysis vs real vehicle positions.

Collects bottom-center positions of moving vehicles (stable MotionEngine
heading), labels them by observed flow (L0-flow ~heading 335, L3-flow ~heading
164), reports per-lane / overlap membership, and writes:
  - debug/lane_partition_points.csv  (x, y, heading, speed, in_L0, in_L3)
  - debug/lane_partition_overview.png (polygons + class-labelled points)

README-ONLY: does not modify scene_config.json.
"""

from __future__ import annotations

import csv
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2

from src.detector import Detector
from src.geometry import Geometry, point_in_polygon
from src.motion import MotionEngine
from src.trajectory import Detection, TrajectoryEngine
from src.wrong_way import DEFAULT_VEHICLE_LABELS

VIDEO_DIR = r"C:\Projects\Traffic Computer Vision\video"
PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCENE_CFG = os.path.join(PKG, "scene_config.json")

IMG_SZ = int(os.environ.get("TCV_IMGSZ", "800"))
CONF = float(os.environ.get("TCV_CONF", "0.25"))
MIN_SPEED = 8.0
FLOW_L0 = math.radians(335.5)   # L0 expected image 24.5 -> motion (360-24.5)
FLOW_L3 = math.radians(163.6)   # L3 expected image 196.4 -> motion (360-196.4)
WINDOW = math.radians(35.0)


def shoe_area(poly):
    return abs(sum(poly[i][0] * poly[(i + 1) % len(poly)][1]
                   - poly[(i + 1) % len(poly)][0] * poly[i][1]
                   for i in range(len(poly)))) / 2.0


def classify(heading):
    h = math.radians(heading % 360.0)
    if abs((h - FLOW_L0 + math.pi) % (2 * math.pi) - math.pi) <= WINDOW:
        return "L0flow"
    if abs((h - FLOW_L3 + math.pi) % (2 * math.pi) - math.pi) <= WINDOW:
        return "L3flow"
    return None


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

    geometry = Geometry.from_json(SCENE_CFG, W, H)
    lanes = {l["lane_id"]: l for l in geometry.lanes}
    poly0, poly3 = lanes["L0"]["polygon"], lanes["L3"]["polygon"]
    print("L0 polygon:", len(poly0), "points, area(ref px^2)=",
          round(shoe_area(poly0)))
    for p in poly0:
        print("   L0", p)
    print("L3 polygon:", len(poly3), "points, area(ref px^2)=",
          round(shoe_area(poly3)))
    for p in poly3:
        print("   L3", p)

    # ---- pure-geometry overlap estimation (grid sampling, no GPU) ----
    step = 48
    in0 = in3 = inboth = 0
    ov_x0 = ov_y0 = W
    ov_x1 = ov_y1 = 0
    for yy in range(0, H, step):
        for xx in range(0, W, step):
            ref = (float(xx), float(yy))
            a = point_in_polygon(ref, poly0)
            b = point_in_polygon(ref, poly3)
            if a:
                in0 += 1
            if b:
                in3 += 1
            if a and b:
                inboth += 1
                ov_x0, ov_y0 = min(ov_x0, xx), min(ov_y0, yy)
                ov_x1, ov_y1 = max(ov_x1, xx), max(ov_y1, yy)
    cell = step * step
    print(f"\n[grid step={step}] sampled cells: L0~{in0 * cell} px^2, "
          f"L3~{in3 * cell} px^2, overlap~{inboth * cell} px^2 "
          f"({100.0 * inboth / max(1, in0):.1f}% of L0, "
          f"{100.0 * inboth / max(1, in3):.1f}% of L3)")
    print(f"overlap bbox (ref): x[{ov_x0},{ov_x1}] y[{ov_y0},{ov_y1}]")

    det = Detector(model_path=r"weights/yolo11x.pt", conf=CONF,
                   device=os.environ.get("TCV_DEVICE", "cuda:0"), imgsz=IMG_SZ)
    traj = TrajectoryEngine(keep_sec=4.0)
    motion = MotionEngine(window_s=0.8, noise_px=4.0, min_points=3)

    rows: list[dict] = []
    labels: list[str] = []
    preview_frame = None
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
        if preview_frame is None:
            preview_frame = fr.copy()
        small = cv2.resize(fr, (IMG_SZ, max(1, int(IMG_SZ * H / W))))
        sx, sy = W / IMG_SZ, H / small.shape[0]
        dets = det.track(small, persist=True)
        t_sec = idx / fps

        scaled = [Detection(xyxy=(d["xyxy"][0] * sx, d["xyxy"][1] * sy,
                                  d["xyxy"][2] * sx, d["xyxy"][3] * sy),
                             conf=d["conf"], label=d["label"], tid=d["id"])
                  for d in dets]
        traj.update(scaled, t_sec)

        for tr in traj.active():
            if tr.label not in DEFAULT_VEHICLE_LABELS:
                continue
            st = motion.update(tr, t_sec)
            if st is None or st.stationary or st.speed < MIN_SPEED \
                    or st.quality < 0.3 or st.heading_deg is None:
                continue
            pos = (tr.last.x, tr.last.bottom_y)
            if geometry.is_in_u_turn_zone(pos) or not geometry.is_on_road(pos):
                continue
            cls = classify(st.heading_deg)
            if cls is None:
                continue
            in0 = point_in_polygon(geometry.to_ref(pos), poly0)
            in3 = point_in_polygon(geometry.to_ref(pos), poly3)
            rows.append({"x": pos[0], "y": pos[1],
                         "heading": st.heading_deg, "speed": st.speed,
                         "in_L0": int(in0), "in_L3": int(in3)})
            labels.append(f"{cls}:{int(in0)}{int(in3)}")
        idx += 1
    cap.release()

    csv_path = os.path.join(PKG, "debug", "lane_partition_points.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["x", "y", "heading", "speed",
                                          "in_L0", "in_L3"])
        w.writeheader()
        w.writerows(rows)

    n_l0flow = sum(1 for r in rows if abs(r["heading"] - 335.5) <= 35.0
                   or abs(r["heading"] - 335.5 + 360.0) <= 35.0)
    n_l3flow = sum(1 for r in rows if abs(r["heading"] - 163.6) <= 35.0
                   or abs(r["heading"] - 163.6 - 360.0) <= 35.0
                   or abs(r["heading"] - 163.6 + 360.0) <= 35.0)
    print(f"\ntracked classified points: total={len(rows)} "
          f"(L0flow~{n_l0flow}, L3flow~{n_l3flow}) -> saved {csv_path}")

    only0 = [r for r in rows if r["in_L0"] and not r["in_L3"]]
    only3 = [r for r in rows if r["in_L3"] and not r["in_L0"]]
    both_ = [r for r in rows if r["in_L0"] and r["in_L3"]]
    neither = [r for r in rows if not r["in_L0"] and not r["in_L3"]]
    print(f"membership: L0-only={len(only0)} L3-only={len(only3)} "
          f"BOTH={len(both_)} neither={len(neither)}")
    for name, rs in (("L0-only", only0), ("L3-only", only3), ("BOTH", both_)):
        hs = sorted(r["heading"] % 360.0 for r in rs)
        if hs:
            lo = min(hs); hi = max(hs)
            cnt335 = sum(1 for h in hs if abs((h - 335.5 + 180) % 360 - 180) <= 35)
            cnt164 = sum(1 for h in hs if abs((h - 163.6 + 180) % 360 - 180) <= 35)
            print(f"  {name}: n={len(hs)} heading range {lo:.1f}..{hi:.1f} "
                  f"-> L0flow={cnt335} L3flow={cnt164}")

    # overlap bbox from polygon vertex pairs by sampling
    ovg_points = [r for r in rows if r["in_L0"] and r["in_L3"]]
    if ovg_points:
        xs = [r["x"] for r in ovg_points]; ys = [r["y"] for r in ovg_points]
        print(f"overlap observed region: bbox x[{min(xs):.0f},{max(xs):.0f}] "
              f"y[{min(ys):.0f},{max(ys):.0f}] n={len(ovg_points)}")

    # draw overview PNG (downscale to 2)
    if preview_frame is not None:
        scale = 2
        img = preview_frame[::scale, ::scale].copy()
        pts0 = [(int(x / scale), int(y / scale)) for x, y in poly0]
        pts3 = [(int(x / scale), int(y / scale)) for x, y in poly3]
        cv2.polylines(img, [pts0], True, (255, 255, 0), 6)
        cv2.polylines(img, [pts3], True, (0, 255, 255), 6)
        # shade overlap pixels sampled
        step = 16
        for yy in range(0, H, step):
            for xx in range(0, W, step):
                if point_in_polygon(geometry.to_ref((xx, yy)), poly0) and \
                        point_in_polygon(geometry.to_ref((xx, yy)), poly3):
                    cv2.circle(img, (xx // scale, yy // scale), 3, (0, 0, 255), -1)
        for r, lab in zip(rows, labels):
            c = (0, 255, 0) if lab.startswith("L0flow") else (255, 0, 0)
            if lab.endswith(":11"):
                c = (0, 0, 255)
            cv2.circle(img, (int(r["x"] / scale), int(r["y"] / scale)), 4, c, -1)
        out = os.path.join(PKG, "debug", "lane_partition_overview.png")
        cv2.imwrite(out, img)
        print(f"overview saved -> {out}  "
              f"(yellow=L0, cyan=L3, red=overlap, green=L0flow pts, blue=L3flow pts)")
    return 0


if __name__ == "__main__":
    sys.exit(main())