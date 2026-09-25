"""Interactive Video Exporter for WIUT Hackathon 2026 CV Track.

Processes sample video (e.g. samples/C3905.mp4) using the integrated
YOLO11x + ByteTrack + EventManager + Scene Geometry + RiskEstimator pipeline.
Renders detection boxes, tracking IDs, road geometry overlays, active event alerts,
and risk gauges onto the video and exports directly to `/home/azizbek/Видео/test_uchun/`.

Usage:
    python export_test_video.py [--video samples/C3905.mp4] [--duration 20] [--output-dir /home/azizbek/Видео/test_uchun]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import cv2
import numpy as np

# Ensure root dir is in sys.path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.config.settings import settings
from src.detection.detector import Detector
from src.events.manager import EventManager
from src.scene.scene import Scene
from src.scene.geometry import Geometry
from src.risk.risk import RiskEstimator


def parse_args():
    parser = argparse.ArgumentParser(description="Export annotated traffic analysis video.")
    parser.add_argument("--video", type=str, default="samples/C3905.mp4",
                        help="Input video file path.")
    parser.add_argument("--output-dir", type=str, default="/home/azizbek/Видео/test_uchun",
                        help="Destination directory for output video.")
    parser.add_argument("--start-sec", type=float, default=0.0,
                        help="Start time in seconds (default: 0.0).")
    parser.add_argument("--duration", type=float, default=15.0,
                        help="Duration in seconds to process (default: 15.0, 0 for full video).")
    parser.add_argument("--stride", type=int, default=3,
                        help="Frame stride for YOLO inference (default: 3).")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Inference image resolution (default: 640).")
    return parser.parse_args()


def draw_hud(frame, t_sec, fps, active_events, risk_score, W, H):
    # Top overlay bar (Dark glass aesthetic)
    overlay = frame.copy()
    bar_h = 75
    cv2.rectangle(overlay, (0, 0), (W, bar_h), (18, 18, 24), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Title & time
    cv2.putText(frame, f"ALGORIX CV | WIUT HACKATHON 2026", (25, 30),
                cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 220, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"TIME: {t_sec:05.2f}s  |  FPS: {fps:.1f}", (25, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1, cv2.LINE_AA)

    # Active Events / Violations alert
    if active_events:
        alert_text = "ALERT: " + " | ".join(sorted(active_events)).upper()
        alert_color = (0, 0, 255) if "accident" in active_events or "wrong_way" in active_events else (0, 140, 255)
        # Background pill for alert
        (tw, th), _ = cv2.getTextSize(alert_text, cv2.FONT_HERSHEY_DUPLEX, 0.75, 2)
        center_x = W // 2 - tw // 2
        cv2.rectangle(frame, (center_x - 15, 12), (center_x + tw + 15, 58), (30, 30, 45), -1)
        cv2.rectangle(frame, (center_x - 15, 12), (center_x + tw + 15, 58), alert_color, 2)
        cv2.putText(frame, alert_text, (center_x, 42),
                    cv2.FONT_HERSHEY_DUPLEX, 0.75, alert_color, 2, cv2.LINE_AA)
    else:
        status_text = "STATUS: TRAFFIC NORMAL"
        cv2.putText(frame, status_text, (W // 2 - 120, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 100), 2, cv2.LINE_AA)

    # Risk Meter (Part B Anticipation)
    gauge_w, gauge_h = 240, 18
    gx, gy = W - gauge_w - 40, 40
    # Background gauge
    cv2.rectangle(frame, (gx, gy), (gx + gauge_w, gy + gauge_h), (50, 50, 60), -1)
    fill_w = int(gauge_w * min(1.0, max(0.0, risk_score)))
    bar_color = (0, 220, 100) if risk_score < 0.3 else ((0, 165, 255) if risk_score < 0.6 else (0, 0, 255))
    if fill_w > 0:
        cv2.rectangle(frame, (gx, gy), (gx + fill_w, gy + gauge_h), bar_color, -1)
    cv2.rectangle(frame, (gx, gy), (gx + gauge_w, gy + gauge_h), (200, 200, 200), 1)

    cv2.putText(frame, f"ACCIDENT RISK (P_5s): {risk_score:.2f}", (gx, gy - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)


def main():
    args = parse_args()

    if not os.path.isfile(args.video):
        print(f"Error: Video file not found: {args.video}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    video_stem = os.path.splitext(os.path.basename(args.video))[0]
    out_filename = f"{video_stem}_analyzed_{int(args.duration)}s.mp4"
    out_path = os.path.join(args.output_dir, out_filename)

    print(f"==================================================")
    print(f"  ALGORIX CV VIDEO PROCESSOR & EXPORTER")
    print(f"==================================================")
    print(f"Input video : {args.video}")
    print(f"Output dest : {out_path}")
    print(f"Start time  : {args.start_sec}s, Duration: {args.duration}s")
    print(f"Stride      : {args.stride}, ImgSz: {args.imgsz}")
    print(f"Device      : {settings.device}")
    print(f"--------------------------------------------------")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Failed to open video {args.video}", file=sys.stderr)
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Downscale output resolution if 4K to 1920x1080 for smooth playback and faster encoding
    out_w, out_h = (1920, 1080) if W > 1920 else (W, H)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, fps, (out_w, out_h))

    start_frame = int(args.start_sec * fps)
    max_frames_to_process = int(args.duration * fps) if args.duration > 0 else (total_frames - start_frame)
    end_frame = min(total_frames, start_frame + max_frames_to_process)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Initialize Modules
    scene = Scene.defaults_estimated(W, H)
    geom = None
    if os.path.exists(settings.scene_config_path):
        try:
            geom = Geometry.from_json(settings.scene_config_path, frame_w=W, frame_h=H)
        except Exception as e:
            print(f"Notice: Geometry load skipped: {e}")

    detector = Detector(model_path=settings.weights_path, conf=settings.conf,
                        iou=settings.iou, device=settings.device, imgsz=args.imgsz)
    event_manager = EventManager(scene, settings, width=W, height=H)
    risk_estimator = RiskEstimator()
    risk_estimator.reset({
        "video_id": video_stem,
        "fps": fps,
        "width": W,
        "height": H,
        "n_frames": total_frames
    })

    small_h = max(1, int(args.imgsz * H / W))
    sx, sy = W / args.imgsz, H / small_h

    current_frame_idx = start_frame
    last_dets = []
    last_active_events = set()
    last_risk = 0.0

    t_start_proc = time.time()
    frames_written = 0

    print("Processing video frames...")
    while current_frame_idx < end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        t_sec = current_frame_idx / fps
        is_infer_frame = ((current_frame_idx - start_frame) % args.stride == 0)

        if is_infer_frame:
            small = cv2.resize(frame, (args.imgsz, small_h))
            dets = detector.track(small, persist=True)
            for d in dets:
                x1, y1, x2, y2 = d["xyxy"]
                d["xyxy"] = (x1 * sx, y1 * sy, x2 * sx, y2 * sy)
            last_dets = dets

            # Update event manager
            event_manager.step(dets, t_sec)

            # Identify currently active events
            active = set()
            for lbl, flags in event_manager.flags_map.items():
                if flags and flags[-1]:
                    active.add(lbl)
            last_active_events = active

            # Update causal risk
            last_risk = risk_estimator.step(frame, t_sec)

        # Scale frame to output resolution
        if (W, H) != (out_w, out_h):
            vis_frame = cv2.resize(frame, (out_w, out_h))
            scale_x, scale_y = out_w / W, out_h / H
        else:
            vis_frame = frame.copy()
            scale_x, scale_y = 1.0, 1.0

        # Draw scene road overlay if available
        if geom is not None and geom.road_polygon:
            pts = np.array([(int(p[0] * scale_x), int(p[1] * scale_y)) for p in geom.road_polygon], np.int32)
            cv2.polylines(vis_frame, [pts], isClosed=True, color=(100, 200, 100), thickness=2)

        # Draw detected objects
        for d in last_dets:
            x1, y1, x2, y2 = d["xyxy"]
            px1, py1 = int(x1 * scale_x), int(y1 * scale_y)
            px2, py2 = int(x2 * scale_x), int(y2 * scale_y)
            label = d.get("label", "vehicle")
            tid = d.get("id", -1)

            box_color = (0, 255, 120) if label == "person" else (255, 180, 50)
            cv2.rectangle(vis_frame, (px1, py1), (px2, py2), box_color, 2)
            cv2.putText(vis_frame, f"#{tid} {label}", (px1, max(20, py1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1, cv2.LINE_AA)

        # Draw HUD overlays
        draw_hud(vis_frame, t_sec, fps, last_active_events, last_risk, out_w, out_h)

        writer.write(vis_frame)
        frames_written += 1
        current_frame_idx += 1

        if frames_written % 30 == 0:
            elapsed = time.time() - t_start_proc
            fps_proc = frames_written / elapsed if elapsed > 0 else 0
            pct = (frames_written / (end_frame - start_frame)) * 100
            print(f"  [{pct:5.1f}%] Frame {frames_written}/{end_frame - start_frame} ({fps_proc:4.1f} fps) | t={t_sec:5.1f}s | Events: {list(last_active_events)}")

    cap.release()
    writer.release()
    total_time = time.time() - t_start_proc

    print(f"--------------------------------------------------")
    print(f"Processing complete!")
    print(f"Exported {frames_written} frames in {total_time:.2f}s ({frames_written / total_time:.1f} fps)")
    print(f"Saved video to: {out_path}")
    print(f"File size: {os.path.getsize(out_path) / (1024*1024):.2f} MB")
    print(f"==================================================")


if __name__ == "__main__":
    main()
