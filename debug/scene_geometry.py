"""Visualize scene geometry from scene_config.json on one mid-frame per video.

Run from the package dir:  python debug/scene_geometry.py

Draws every ENABLED config feature (road, lanes + directions, crosswalks, stop
lines, solid lines, intersection/u-turn zones, traffic-light ROIs) together
with YOLO/ByteTrack detections and their bottom centers, then prints per video:
video, frame, timestamp, resolution, geometry coordinates, output path.

Debug-only output: never used in predictions.
"""

from __future__ import annotations

import json
import math
import os

import cv2
import numpy as np

PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PKG_DIR not in __import__("sys").path:
    __import__("sys").path.insert(0, PKG_DIR)

from src.detector import Detector
from src.geometry import Geometry

DEBUG_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_DIR = r"C:\Projects\Traffic Computer Vision\video"
WEIGHTS = os.path.join(PKG_DIR, "weights", "yolo11x.pt")
CONFIG = os.path.join(PKG_DIR, "scene_config.json")
VIDEOS = ["C3896.MP4", "C3897.MP4", "C3902.MP4", "C3905.MP4"]
OUT_W = 1600
Y_HOME = 30

ROAD_COLOR = (0, 215, 255)
LANE_COLOR = (0, 170, 255)
CROSSWALK_COLOR = (255, 120, 0)
STOP_LINE_COLOR = (0, 255, 255)
SOLID_LINE_COLOR = (255, 255, 255)
INTERSECTION_COLOR = (60, 200, 60)
U_TURN_COLOR = (255, 0, 255)
EXCLUSION_COLOR = (60, 60, 200)
TRAFFIC_LIGHT_COLOR = (255, 255, 0)
BOX_COLOR = (0, 255, 0)
BOTTOM_COLOR = (0, 0, 255)


def put_label(img, text, origin, color, scale=0.6):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    x, y = origin
    cv2.rectangle(img, (x, y - th - 8), (x + tw + 6, y + 4), (0, 0, 0), -1)
    cv2.putText(img, text, (x + 3, y - 4), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, 2, cv2.LINE_AA)


def to_int(poly):
    return np.array(poly, dtype=np.int32).reshape(-1, 1, 2)


def fill_poly(img, pts, color, alpha=0.25):
    overlay = img.copy()
    cv2.fillPoly(overlay, [to_int(pts)], color)
    cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0, img)


def draw_poly(img, pts, color, thickness=3, label=None, fill=0.0):
    if fill > 0:
        fill_poly(img, pts, color, fill)
    cv2.polylines(img, [to_int(pts)], True, color, thickness, cv2.LINE_AA)
    if label is not None:
        put_label(img, label, (int(pts[0][0]), int(pts[0][1])), color)


def draw_line(img, p1, p2, color, thickness=6, label=None):
    cv2.line(img, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])),
             color, thickness, cv2.LINE_AA)
    if label is not None:
        mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
        put_label(img, label, (int(mx), int(my)), color)


def draw_direction_arrow(img, pts, heading_deg, color, length=160):
    cx = float(np.mean([p[0] for p in pts]))
    cy = float(np.mean([p[1] for p in pts]))
    vx, vy = math.cos(math.radians(heading_deg)), math.sin(math.radians(heading_deg))
    cv2.arrowedLine(img, (int(cx), int(cy)), (int(cx + vx * length), int(cy + vy * length)),
                    color, 5, cv2.LINE_AA, tipLength=0.4)


def spts(pts, f):
    return [(float(p[0]) * f, float(p[1]) * f) for p in pts]


def spt(p, f):
    return (int(float(p[0]) * f), int(float(p[1]) * f))


def draw_geometry(img, g: Geometry, f: float = 1.0):
    """f = display_width / full_width; scales full-res geometry to the canvas."""
    if g.road_polygon:
        draw_poly(img, spts(g.road_polygon, f), ROAD_COLOR, 3, "ROAD", fill=0.12)
    for lane in g.lanes:
        poly = spts(lane["polygon"], f)
        draw_poly(img, poly, LANE_COLOR, 3, f"LANE {lane['lane_id']}", fill=0.2)
        draw_direction_arrow(img, poly, lane["expected_direction"], LANE_COLOR)
    for i, cw in enumerate(g.crosswalks):
        draw_poly(img, spts(cw, f), CROSSWALK_COLOR, 3, f"CROSSWALK {i}", fill=0.3)
    for i, sl in enumerate(g.stop_lines):
        draw_line(img, spt(sl[0], f), spt(sl[1], f), STOP_LINE_COLOR, 6, f"STOP {i}")
    for i, sl in enumerate(g.solid_lines):
        draw_line(img, spt(sl[0], f), spt(sl[1], f), SOLID_LINE_COLOR, 6, f"SOLID {i}")
    for i, z in enumerate(g.intersections):
        draw_poly(img, spts(z, f), INTERSECTION_COLOR, 3, f"INTERSECTION {i}", fill=0.25)
    for i, z in enumerate(g.u_turn_zones):
        draw_poly(img, spts(z, f), U_TURN_COLOR, 3, f"U-TURN {i}", fill=0.25)
    for i, z in enumerate(g.exclusion_regions):
        draw_poly(img, spts(z, f), EXCLUSION_COLOR, 3, f"EXCLUSION {i}", fill=0.35)
    for i, roi in enumerate(g.traffic_light_rois):
        x1 = min(roi[0][0], roi[1][0]); y1 = min(roi[0][1], roi[1][1])
        x2 = max(roi[0][0], roi[1][0]); y2 = max(roi[0][1], roi[1][1])
        cv2.rectangle(img, (int(x1 * f), int(y1 * f)), (int(x2 * f), int(y2 * f)),
                      TRAFFIC_LIGHT_COLOR, 3)
        put_label(img, f"TRAFFIC {i}", (int(x1 * f), int(y1 * f) - 8),
                  TRAFFIC_LIGHT_COLOR)


def process(video_path: str, det: Detector) -> dict:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    mid = max(0, n // 2)
    cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"cannot read frame {mid} from {video_path}")

    g = Geometry.from_json(CONFIG, frame_w=W, frame_h=H)
    t_sec = mid / fps

    dets = det.track(frame, persist=False)
    if not dets:
        dets = det.detect(frame)

    f = OUT_W / W
    img = cv2.resize(frame, (OUT_W, int(H * f)))
    scale = lambda xy: (int(xy[0] * f), int(xy[1] * f))

    draw_geometry(img, g, f=f)

    for d in dets:
        p1, p2 = scale((d["xyxy"][0], d["xyxy"][1])), scale((d["xyxy"][2], d["xyxy"][3]))
        cv2.rectangle(img, p1, p2, BOX_COLOR, 2)
        bc = scale(((d["xyxy"][0] + d["xyxy"][2]) / 2, d["xyxy"][3]))
        cv2.circle(img, bc, 7, BOTTOM_COLOR, -1)
        tag = f"{d['label']} {d['conf']:.2f}" if "id" not in d else \
            f"{d['label']}#{d['id']}"
        put_label(img, tag, (p1[0], max(16, p1[1] - 6)), BOX_COLOR, scale=0.5)

    out = os.path.join(DEBUG_DIR,
                       f"scene_geometry_{os.path.splitext(os.path.basename(video_path))[0]}.png")
    cv2.imwrite(out, img, [int(cv2.IMWRITE_PNG_COMPRESSION), 6])

    rec = {
        "video": os.path.basename(video_path),
        "frame": mid,
        "timestamp": round(t_sec, 3),
        "resolution": [W, H],
        "scale_to_ref": {"sx": g.sx, "sy": g.sy},
        "geometry": {
            "road_polygon": g.road_polygon,
            "lanes": g.lanes,
            "crosswalks": g.crosswalks,
            "stop_lines": g.stop_lines,
            "solid_lines": g.solid_lines,
            "intersection_zones": g.intersections,
            "u_turn_zones": g.u_turn_zones,
            "exclusion_regions": g.exclusion_regions,
            "traffic_light_rois": g.traffic_light_rois,
        },
        "detections": len(dets),
        "output": out,
    }
    print(json.dumps(rec, ensure_ascii=False, indent=2))
    print("-----")
    return rec


def main():
    det = Detector(model_path=WEIGHTS, conf=0.25, device="cuda:0", imgsz=800)
    for name in VIDEOS:
        vp = os.path.join(VIDEO_DIR, name)
        try:
            process(vp, det)
        except Exception as exc:
            print(f"[{name}] ERROR: {exc!r}")
            print("-----")


if __name__ == "__main__":
    main()