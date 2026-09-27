"""
Real-Video Annotation & Export Pipeline — WIUT Hackathon 2026 CV Track
=======================================================================
Processes C3905.mp4 using the REAL pipeline:
  - YOLO11x detection + ByteTrack tracking (real objects)
  - Calibrated scene geometry from scene_config.json
  - Green polygon (road zone), Yellow polygon (crosswalk), Red stop line, Cyan ROIs
  - EventManager flags all 14 event classes frame by frame
  - RiskEstimator causal Part B risk score
  - Professional HUD overlay with zone visualizations

Output: /home/azizbek/Видео/test_uchun/00_REAL_FULL_ANALYSIS.mp4

Usage:
    python export_test_video.py [--start 0] [--duration 30] [--stride 3]
"""
from __future__ import annotations

import argparse, os, sys, time
import cv2
import numpy as np

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.config.settings import settings
from src.detection.detector import Detector
from src.events.manager import EventManager
from src.scene.scene import Scene
from src.scene.geometry import Geometry
from src.risk.risk import RiskEstimator

OUTPUT_DIR = "/home/azizbek/Видео/test_uchun"
VIDEO_IN    = "samples/C3905.mp4"

# ── Zone colors (BGR) ─────────────────────────────────────────────────────────
COL_GREEN  = (50,  230,  50)   # road polygon
COL_YELLOW = (0,   220, 255)   # crosswalk zone
COL_RED    = (0,     0, 255)   # stop line
COL_CYAN   = (255, 240,   0)   # traffic-light / sign ROIs

# ── Bounding-box colors per class ─────────────────────────────────────────────
OBJ_COLORS = {
    "car":        (255, 200,  50),
    "bus":        (0,   200, 255),
    "truck":      (255, 120,   0),
    "motorcycle": (200, 100, 255),
    "person":     (50,  255,  80),
}

# ── Event severity / color ────────────────────────────────────────────────────
HIGH_RISK  = {"accident", "wrong_way", "red_light", "near_miss"}
MED_RISK   = {"failure_to_yield", "jaywalking", "illegal_u_turn",
              "illegal_turn", "solid_line_crossing", "stop_line"}

FULL_EVENT_NAMES = {
    "accident":           "TO'QNASHUV (Accident)",
    "near_miss":          "DEYARLI TO'QNASHUV (Near-Miss)",
    "red_light":          "QIZIL CHIROQ (Red Light)",
    "wrong_way":          "NOTO'G'RI YO'NALISH (Wrong Way)",
    "illegal_u_turn":     "NOQONUNIY QAYTISH (Illegal U-Turn)",
    "stopped_vehicle":    "TO'XTAB TURISH (Stopped Vehicle ≥10s)",
    "jaywalking":         "PIYODA (Jaywalking)",
    "failure_to_yield":   "YO'L BERMASLIK (Failure to Yield)",
    "illegal_turn":       "NOQONUNIY BURILISH (Illegal Turn)",
    "solid_line_crossing":"CHIZIQ OSHISH (Solid Line Crossing)",
    "stop_line":          "TO'XTASH CHIZIG'I (Stop Line)",
    "congestion":         "TIRBANDLIK (Congestion)",
    "road_obstacle":      "TO'SIQ (Road Obstacle)",
    "fire_smoke":         "OLOV/TUTUN (Fire/Smoke)",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video", default=VIDEO_IN)
    p.add_argument("--output-dir", default=OUTPUT_DIR)
    p.add_argument("--start", type=float, default=0.0)
    p.add_argument("--duration", type=float, default=30.0, help="0 = full video")
    p.add_argument("--stride", type=int, default=3)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--out-w", type=int, default=1920)
    p.add_argument("--out-h", type=int, default=1080)
    return p.parse_args()


# ── HUD drawing ───────────────────────────────────────────────────────────────

def _fill_bar(frame, x, y, w, h, ratio, color):
    cv2.rectangle(frame, (x, y), (x+w, y+h), (40, 42, 50), -1)
    fw = int(w * max(0.0, min(1.0, ratio)))
    if fw > 0:
        cv2.rectangle(frame, (x, y), (x+fw, y+h), color, -1)
    cv2.rectangle(frame, (x, y), (x+w, y+h), (180, 180, 180), 1)


def draw_hud(frame, t_sec, active, risk, W, H, n_tracks):
    # ── top bar ──────────────────────────────────────────────────
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (W, 82), (12, 14, 20), -1)
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)

    cv2.putText(frame, "ALGORIX CV  |  WIUT HACKATHON 2026",
                (20, 30), cv2.FONT_HERSHEY_DUPLEX, 0.80, (0, 220, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"TIME: {t_sec:06.2f}s   TRACKS: {n_tracks}   DEVICE: {settings.device.upper()}",
                (20, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (180, 180, 180), 1, cv2.LINE_AA)

    # ── risk gauge (top-right) ────────────────────────────────────
    gx = W - 280
    gy = 32
    risk_col = (50, 220, 50) if risk < 0.3 else ((0, 165, 255) if risk < 0.6 else (0, 0, 255))
    cv2.putText(frame, f"AVARIYA XAVFI (P_5s): {risk:.2f}",
                (gx, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1, cv2.LINE_AA)
    _fill_bar(frame, gx, gy, 240, 16, risk, risk_col)

    # ── center alert ──────────────────────────────────────────────
    if active:
        labels = sorted(active)
        # choose worst color
        if any(l in HIGH_RISK for l in labels):
            ac = (0, 0, 255)
        elif any(l in MED_RISK for l in labels):
            ac = (0, 140, 255)
        else:
            ac = (30, 180, 255)

        # Show at most top-3 events to keep text readable
        display = " | ".join(l.upper() for l in labels[:3])
        if len(labels) > 3:
            display += f" +{len(labels)-3}"
        txt = f"⚠  {display}"
        (tw, _), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_DUPLEX, 0.78, 2)
        cx = max(20, W // 2 - tw // 2)
        cv2.rectangle(frame, (cx - 12, 8), (cx + tw + 12, 62), (20, 22, 30), -1)
        cv2.rectangle(frame, (cx - 12, 8), (cx + tw + 12, 62), ac, 2)
        cv2.putText(frame, txt, (cx, 44),
                    cv2.FONT_HERSHEY_DUPLEX, 0.78, ac, 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, "STATUS: TRAFFIC NORMAL",
                    (W // 2 - 150, 44), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (50, 220, 50), 2, cv2.LINE_AA)

    # ── active events panel (left side, below main bar) ──────────
    if active:
        py = 95
        for lbl in sorted(active)[:6]:
            full = FULL_EVENT_NAMES.get(lbl, lbl)
            ec = (0, 0, 220) if lbl in HIGH_RISK else (0, 130, 255)
            cv2.putText(frame, f"▶ {full}",
                        (18, py), cv2.FONT_HERSHEY_SIMPLEX, 0.52, ec, 1, cv2.LINE_AA)
            py += 26

    # ── legend (bottom strip) ─────────────────────────────────────
    by = H - 10
    cv2.rectangle(frame, (0, H - 35), (W, H), (12, 14, 20), -1)
    cv2.putText(frame,
                "■ YASHIL=Yo'l zonasi   ■ SARIQ=Zebra o'tish joyi   ■ QIZIL=To'xtash chizig'i   ■ MOVIY=Svetafor/belgilar",
                (12, by), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (200, 200, 200), 1, cv2.LINE_AA)


# ── Geometry overlay (SLEEK THIN OUTLINES, NO FILLS) ──────────────────────────

# Load precomputed pixel-perfect zone masks if available
_ZONE_ASSETS = None
_ZONE_PATH = os.path.join(ROOT_DIR, "weights", "calibrated_zones_1080p.npz")
if os.path.exists(_ZONE_PATH):
    try:
        _loaded = np.load(_ZONE_PATH)
        _g_raw = _loaded["green"]
        _y_raw = _loaded["yellow"]
        _r_raw = _loaded["red"]
        # Thin edges (1px clean outline)
        k = np.ones((3, 3), np.uint8)
        _g_mid = cv2.bitwise_and(_g_raw, cv2.bitwise_not(cv2.erode(_g_raw, k)))
        _y_mid = cv2.bitwise_and(_y_raw, cv2.bitwise_not(cv2.erode(_y_raw, k)))
        _r_mid = cv2.bitwise_and(_r_raw, cv2.bitwise_not(cv2.erode(_r_raw, k)))
        _ZONE_ASSETS = {
            "g": _g_mid > 0,
            "y": _y_mid > 0,
            "r": _r_mid > 0,
        }
    except Exception as e:
        print("Warning: could not load precomputed zone masks:", e)


def draw_zones(vis_frame, geom, sx, sy):
    """Draw calibrated zones cleanly with thin lines (exact match to user drawing)."""
    if _ZONE_ASSETS is not None and vis_frame.shape[1] == 1920 and vis_frame.shape[0] == 1080:
        # Pixel-perfect exact drawing from user reference:
        vis_frame[_ZONE_ASSETS["g"]] = COL_GREEN
        vis_frame[_ZONE_ASSETS["y"]] = COL_YELLOW
        vis_frame[_ZONE_ASSETS["r"]] = COL_RED
    else:
        # Fallback polygon drawing
        def s(pts):
            return np.array([(int(x * sx), int(y * sy)) for x, y in pts], np.int32)

        if geom.road_polygon:
            pts = s(geom.road_polygon)
            cv2.polylines(vis_frame, [pts], True, COL_GREEN, 1, cv2.LINE_AA)

        for cw in geom.crosswalks:
            pts = s(cw)
            cv2.polylines(vis_frame, [pts], True, COL_YELLOW, 1, cv2.LINE_AA)

        for sl in geom.stop_lines:
            p1 = (int(sl[0][0] * sx), int(sl[0][1] * sy))
            p2 = (int(sl[1][0] * sx), int(sl[1][1] * sy))
            cv2.line(vis_frame, p1, p2, COL_RED, 2, cv2.LINE_AA)

    # Traffic light / sign ROIs (cyan thin boxes)
    tl_labels = ["TL-L", "TL-MAIN", "SIGN:PED", "SIGN:RND"]
    for i, tl in enumerate(geom.traffic_light_rois):
        x1, y1 = int(tl[0][0] * sx), int(tl[0][1] * sy)
        x2, y2 = int(tl[1][0] * sx), int(tl[1][1] * sy)
        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), COL_CYAN, 1, cv2.LINE_AA)
        label = tl_labels[i] if i < len(tl_labels) else f"ROI{i}"
        cv2.putText(vis_frame, label,
                    (x1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX,
                    0.35, COL_CYAN, 1, cv2.LINE_AA)


# ── Box drawing (SLEEK THIN 1PX BOXES) ────────────────────────────────────────

def draw_detections(vis_frame, dets, histories, sx, sy, is_infer):
    for d in dets:
        x1r, y1r, x2r, y2r = d["xyxy"]
        px1, py1 = int(x1r * sx), int(y1r * sy)
        px2, py2 = int(x2r * sx), int(y2r * sy)
        lbl = d.get("label", "vehicle")
        tid = d.get("id", -1)
        cx, cy = (px1 + px2) // 2, py2

        # Motion trail
        if tid not in histories:
            histories[tid] = []
        if is_infer:
            histories[tid].append((cx, cy))
            if len(histories[tid]) > 20:
                histories[tid].pop(0)

        trail = histories[tid]
        for i in range(1, len(trail)):
            cv2.line(vis_frame, trail[i-1], trail[i], (0, 180, 255), 1, cv2.LINE_AA)

        color = OBJ_COLORS.get(lbl, (200, 200, 200))
        # Sleek thin 1px rectangle
        cv2.rectangle(vis_frame, (px1, py1), (px2, py2), color, 1, cv2.LINE_AA)

        # Small bottom-center dot (anchor point)
        cv2.circle(vis_frame, (cx, cy), 3, color, -1)

        # Minimal sleek label tag
        tag = f"#{tid} {lbl[:3].upper()}"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
        bgy1, bgy2 = max(0, py1 - th - 4), py1
        cv2.rectangle(vis_frame, (px1, bgy1), (px1 + tw + 4, bgy2), (20, 22, 28), -1)
        cv2.putText(vis_frame, tag, (px1 + 2, max(th, py1 - 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    stem = os.path.splitext(os.path.basename(args.video))[0]
    dur_tag = f"{int(args.duration)}s" if args.duration > 0 else "full"
    out_name = f"00_REAL_{stem}_{dur_tag}_annotated.mp4"
    out_path = os.path.join(args.output_dir, out_name)

    print("=" * 64)
    print("  ALGORIX CV — REAL VIDEO PIPELINE ANNOTATOR")
    print("=" * 64)
    print(f"  Input  : {args.video}")
    print(f"  Output : {out_path}")
    print(f"  Start  : {args.start}s   Duration: {args.duration}s")
    print(f"  Stride : {args.stride}   ImgSz: {args.imgsz}   Device: {settings.device}")
    print("-" * 64)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print("ERROR: Cannot open video", args.video, file=sys.stderr)
        sys.exit(1)

    fps       = cap.get(cv2.CAP_PROP_FPS) or 30.0
    tot_frames= int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    VW        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    VH        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    OW, OH = args.out_w, args.out_h
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (OW, OH))

    start_fr = int(args.start * fps)
    dur_fr   = int(args.duration * fps) if args.duration > 0 else (tot_frames - start_fr)
    end_fr   = min(tot_frames, start_fr + dur_fr)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_fr)

    # Scale: full video -> output resolution
    sx_out, sy_out = OW / VW, OH / VH

    # ── Modules ────────────────────────────────────────────────────
    import json
    cfg = json.load(open("scene_config.json"))

    # Build Scene from calibrated config
    rp_cfg = cfg.get("road_polygon", {})
    raw_rp = rp_cfg.get("points", []) if isinstance(rp_cfg, dict) else rp_cfg
    road_poly = [tuple(p) for p in raw_rp]

    sl_cfg = cfg.get("stop_lines", [])
    sl_items = sl_cfg.get("lines", []) if isinstance(sl_cfg, dict) else sl_cfg
    stop_lines = []
    for e in sl_items:
        if isinstance(e, dict):
            if "line" in e:
                stop_lines.append([(e["line"][0][0], e["line"][0][1]), (e["line"][1][0], e["line"][1][1])])
            elif "start" in e and "end" in e:
                stop_lines.append([(e["start"][0], e["start"][1]), (e["end"][0], e["end"][1])])

    scene = Scene(width=VW, height=VH,
                  road_poly=road_poly, stop_lines=stop_lines)
    geom  = Geometry.from_json("scene_config.json", frame_w=VW, frame_h=VH)

    detector = Detector(model_path=settings.weights_path,
                        conf=settings.conf, iou=settings.iou,
                        device=settings.device, imgsz=args.imgsz)

    event_mgr = EventManager(scene, settings, width=VW, height=VH)

    risk_est = RiskEstimator()
    risk_est.reset({"video_id": stem, "fps": fps,
                    "width": VW, "height": VH, "n_frames": tot_frames})

    # Scale: inference (imgsz) -> full video coordinates
    small_h = max(1, int(args.imgsz * VH / VW))
    sx_inf  = VW / args.imgsz
    sy_inf  = VH / small_h

    # ── Processing loop ─────────────────────────────────────────────
    last_dets   = []
    last_active = set()
    last_risk   = 0.0
    histories   = {}
    t0 = time.time()
    written = 0

    print("Processing …")
    for fi in range(start_fr, end_fr):
        ret, frame = cap.read()
        if not ret:
            break

        t_sec       = fi / fps
        is_infer    = ((fi - start_fr) % args.stride == 0)

        if is_infer:
            small = cv2.resize(frame, (args.imgsz, small_h))
            dets  = detector.track(small, persist=True)
            for d in dets:
                x1, y1, x2, y2 = d["xyxy"]
                d["xyxy"] = (x1 * sx_inf, y1 * sy_inf,
                             x2 * sx_inf, y2 * sy_inf)
            last_dets = dets

            event_mgr.step(dets, t_sec)

            active = set()
            for lbl, flags in event_mgr.flags_map.items():
                if flags and flags[-1]:
                    active.add(lbl)
            last_active = active

            last_risk = risk_est.step(frame, t_sec)

        # ── Render ─────────────────────────────────────────────────
        vis = cv2.resize(frame, (OW, OH))

        draw_zones(vis, geom, sx_out, sy_out)
        draw_detections(vis, last_dets, histories, sx_out, sy_out, is_infer)
        draw_hud(vis, t_sec, last_active, last_risk, OW, OH, len(last_dets))

        writer.write(vis)
        written += 1

        if written % 30 == 0:
            el  = time.time() - t0
            pct = (fi - start_fr + 1) / (end_fr - start_fr) * 100
            efps = written / el if el > 0 else 0
            ev_str = " | ".join(sorted(last_active)) if last_active else "–"
            print(f"  [{pct:5.1f}%] t={t_sec:5.1f}s  {efps:4.1f}fps  "
                  f"tracks={len(last_dets)}  events=[{ev_str}]")

    cap.release()
    writer.release()
    total_t = time.time() - t0
    sz_mb   = os.path.getsize(out_path) / 1024**2

    # Finalize and print detected events
    dur_total = (end_fr - start_fr) / fps
    detected_events = event_mgr.finalize(dur_total)

    print("-" * 64)
    print(f"Done. {written} frames in {total_t:.1f}s ({written/total_t:.1f} fps)")
    print(f"Saved: {out_path}  ({sz_mb:.1f} MB)")
    print(f"Detected Events Count: {len(detected_events)}")
    for ev in detected_events:
        print(f"  [{ev[0]:6.2f}s - {ev[1]:6.2f}s]  {ev[2].upper()}")
    print("=" * 64)


if __name__ == "__main__":
    main()
