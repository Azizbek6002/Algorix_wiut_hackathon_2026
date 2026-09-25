"""Semi-automatic labeling assist: draft events + preview keyframes for human review.

Runs the SAME pipeline loop as src/pipeline.py (GPU inference), but additionally:
  - records per-flag track evidence (track ids) for honest "why",
  - extracts 3 keyframes per event (start/mid/end) with detection boxes drawn,
  - writes a draft JSON (status "unconfirmed") for the human to confirm/correct.

Nothing here modifies run_submission.py / evaluate.py / solution.py / my_labels.json.
Draft answers are NOT ground truth until confirmed by a human.
"""

from __future__ import annotations

import json
import os

import cv2
import numpy as np

from .events import rules
from .detection import Detector
from .postprocessing import clean_events, events_from_flags
from .scene import Scene

STRIDE = int(os.environ.get("TCV_STRIDE", "3"))
IMG_SZ = int(os.environ.get("TCV_IMGSZ", "800"))
CONF = float(os.environ.get("TCV_CONF", "0.25"))
DEVICE = os.environ.get("TCV_DEVICE", "cuda:0")
MAX_FRAMES = int(os.environ.get("TCV_MAX_FRAMES", "0"))
CV_THREADS = int(os.environ.get("TCV_CV_THREADS", "4"))
OMP_THREADS = os.environ.get("TCV_OMP_THREADS", "4")

os.environ.setdefault("OMP_NUM_THREADS", OMP_THREADS)
os.environ.setdefault("MKL_NUM_THREADS", OMP_THREADS)

try:
    import cv2 as _cv2
    _cv2.setNumThreads(CV_THREADS)
except Exception:
    pass

PREVIEW_W = 1280

EXPLAIN = {
    "wrong_way": "машина едет против главного потока",
    "stopped_vehicle": "машина стоит на дороге >=10 c",
    "congestion": "затор: >=60% машин стоят/ползут",
    "jaywalking": "пешеход идёт по дороге вне перехода",
    "failure_to_yield": "машина на переходе, когда там пешеход",
    "solid_line_crossing": "пересечена сплошная линия",
    "illegal_turn": "запрещённый поворот",
    "illegal_u_turn": "запрещённый разворот",
    "red_light": "пересечена стоп-линия на красный",
    "stop_line": "остановка за стоп-линией",
    "accident": "столкновение треков",
    "near_miss": "резкий манёвр/TTC без контакта",
    "road_obstacle": "препятствие на дороге",
    "fire_smoke": "огонь или дым",
}

COLORS = {
    "wrong_way": (0, 0, 255), "stopped_vehicle": (0, 165, 255),
    "congestion": (0, 165, 255), "jaywalking": (255, 0, 255),
    "failure_to_yield": (0, 255, 255), "accident": (0, 0, 255),
    "near_miss": (0, 215, 255), "solid_line_crossing": (128, 0, 128),
    "illegal_turn": (128, 0, 128), "illegal_u_turn": (128, 0, 128),
    "red_light": (0, 0, 255), "stop_line": (0, 0, 255),
    "road_obstacle": (64, 64, 64), "fire_smoke": (64, 64, 64),
}


def _run_loop(video_path: str, scene: Scene, det: Detector):
    """Same loop as pipeline.run_pipeline; returns timestamps, flags, evidence."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    tracks: dict[int, rules.TrackState] = {}
    rules._congestion_hold["on"] = False
    rules._congestion_hold["at"] = 0.0

    timestamps: list[float] = []
    flags_map = {k: [] for k in EXPLAIN}
    evidence: dict[str, list[list]] = {k: [] for k in EXPLAIN}

    idx = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if idx % STRIDE != 0:
            idx += 1
            continue
        if MAX_FRAMES and idx >= MAX_FRAMES:
            break
        small = cv2.resize(fr, (IMG_SZ, max(1, int(IMG_SZ * H / W))))
        sx, sy = W / IMG_SZ, H / small.shape[0]
        dets = det.track(small, persist=True)
        for d in dets:
            x1, y1, x2, y2 = d["xyxy"]
            d["xyxy"] = (x1 * sx, y1 * sy, x2 * sx, y2 * sy)

        t_sec = idx / fps
        rules.update(tracks, dets, t_sec, scene)
        flags = rules.frame_flags(tracks, scene, t_sec)
        timestamps.append(t_sec)
        for k, v in flags.items():
            flags_map[k].append(v)
        ev = _collect_evidence(tracks, scene, flags, t_sec)
        for k, v in ev.items():
            evidence[k].append(v)
        idx += 1
    cap.release()
    return fps, n, W, H, timestamps, flags_map, evidence


def _collect_evidence(tracks, scene, flags, t_sec):
    ev: dict[str, list] = {k: [] for k in EXPLAIN}
    vehicles = [st for st in tracks.values() if st.label != "person"]
    pedestrians = [st for st in tracks.values() if st.label == "person"]
    if flags["stopped_vehicle"]:
        ev["stopped_vehicle"] = [
            (str(id_), st.label, round(st.speed, 1))
            for id_, st in tracks.items()
            if st.label != "person" and st.stationary_at is not None
            and (t_sec - st.stationary_at) >= rules.STATIONARY_SEC]
    if flags["wrong_way"]:
        ev["wrong_way"] = [
            (str(id_), st.label, round(st.heading, 1))
            for id_, st in tracks.items()
            if st.label != "person" and st.speed >= rules.STILL_PX_S
            and min(rules._angle_dev(st.heading, f) for f in scene.dominant_flow_deg)
            >= 180 - rules.WRONG_WAY_DEV]
    if flags["jaywalking"]:
        ev["jaywalking"] = [
            (str(id_), st.label,
             int(scene.crosswalk_band and rules._in_band((st.x, st.y), scene.crosswalk_band)))
            for id_, st in tracks.items()
            if st.label == "person" and scene.road_poly
            and rules._in_poly(scene.road_poly, st.x, st.y)
            and not (scene.crosswalk_band and rules._in_band((st.x, st.y), scene.crosswalk_band))]
    if flags["failure_to_yield"]:
        ev["failure_to_yield"] = {
            "peds": [str(id_) for id_, st in tracks.items() if st.label == "person"
                     and scene.crosswalk_band
                     and rules._in_band((st.x, st.y), scene.crosswalk_band)],
            "vehs": [str(id_) for id_, st in tracks.items() if st.label != "person"
                     and scene.crosswalk_band
                     and rules._in_band((st.x, st.y), scene.crosswalk_band)]}
    if flags["congestion"]:
        frac = None
        if len(vehicles) >= rules.CONGESTION_MIN_VEHICLES:
            frac = sum(1 for st in vehicles if st.speed < rules.STILL_PX_S) / len(vehicles)
        ev["congestion"] = [round(frac or 0.0, 2), len(vehicles)]
    for k in ("solid_line_crossing", "illegal_turn", "illegal_u_turn",
              "red_light", "stop_line", "accident", "near_miss",
              "road_obstacle", "fire_smoke"):
        if flags[k]:
            ev[k] = [str(flags[k])]
    return ev


def _read_frame(cap, frame_idx: int):
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(frame_idx)))
    ok, fr = cap.read()
    return fr if ok else None


def _sample_evidence(ev: list, timestamps: list[float], s: float, e: float,
                     limit: int = 3) -> list:
    """Compress per-frame evidence: keep up to `limit` distinct samples inside [s, e]."""
    seen = set()
    out = []
    for t, sample in zip(timestamps, ev):
        if not (s - 1.0 <= t <= e + 1.0) or not sample:
            continue
        key = json.dumps(sample, sort_keys=True, ensure_ascii=False)
        if key in seen:
            continue
        seen.add(key)
        out.append(sample)
        if len(out) >= limit:
            break
    return out


def _draw_preview(frame_bgr, det, label: str, t_sec: float, frame_idx: int):
    dets = det.detect(frame_bgr)
    color = COLORS.get(label, (0, 255, 0))
    scale = PREVIEW_W / frame_bgr.shape[1]
    small = cv2.resize(frame_bgr, (PREVIEW_W, max(1, int(PREVIEW_W * frame_bgr.shape[0] / frame_bgr.shape[1]))))
    for d in dets:
        x1, y1, x2, y2 = d["xyxy"]
        p1 = (int(x1 * scale), int(y1 * scale))
        p2 = (int(x2 * scale), int(y2 * scale))
        cv2.rectangle(small, p1, p2, (0, 255, 0), 2)
        cv2.putText(small, f"{d['label']} {d['conf']:.2f}", (p1[0], max(18, p1[1] - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)
    line1 = f"{label}  t={t_sec:.2f}s  frame={frame_idx}"
    cv2.rectangle(small, (4, 4), (int(PREVIEW_W * 0.72), 34), (0, 0, 0), -1)
    cv2.putText(small, line1, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    return small


def label_assist(video_path: str, out_dir: str = "labels_preview",
                 draft_dir: str = "labels_draft") -> dict:
    name = os.path.splitext(os.path.basename(video_path))[0]
    preview_dir = os.path.join(out_dir, name)
    os.makedirs(preview_dir, exist_ok=True)
    os.makedirs(draft_dir, exist_ok=True)

    scene = None
    det = Detector(model_path=r"weights/yolo11x.pt", conf=CONF, device=DEVICE, imgsz=IMG_SZ)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = n / fps if fps else 0.0
    cap.release()

    scene = Scene.defaults_estimated(W, H)
    fps_r, n_r, W_r, H_r, timestamps, flags_map, evidence = _run_loop(video_path, scene, det)

    events = events_from_flags(timestamps, flags_map)
    events = clean_events(events, duration)

    draft = []
    for s, e, label in events:
        i_start = max(0, int(s * fps))
        i_mid = min(n - 1, int(round(0.5 * (s + e) * fps)))
        i_end = min(n - 1, int(e * fps))
        highs = [("start", i_start, s), ("mid", i_mid, 0.5 * (s + e)),
                 ("end", i_end, e)]
        previews = []
        cap2 = cv2.VideoCapture(video_path)
        for tag, idx, t in highs:
            fr = _read_frame(cap2, idx)
            if fr is None:
                continue
            img = _draw_preview(fr, det, label, t, idx)
            fn = os.path.join(preview_dir, f"{label}__{s:.2f}-{e:.2f}__{tag}.jpg")
            cv2.imwrite(fn, img, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
            previews.append(fn)
        cap2.release()
        ev = _sample_evidence(evidence[label], timestamps, s, e, 3)
        draft.append({
            "video": name + ".MP4",
            "event": [round(s, 3), round(e, 3), label],
            "why": EXPLAIN.get(label, ""),
            "evidence": ev,
            "status": "unconfirmed",
            "preview": previews,
        })

    draft_path = os.path.join(draft_dir, name + ".draft.json")
    with open(draft_path, "w", encoding="utf-8") as fh:
        json.dump({"video": name + ".MP4", "duration": duration,
                   "fps": fps, "n_frames": n, "draft_events": draft},
                  fh, ensure_ascii=False, indent=2)

    counts: dict[str, int] = {}
    for _, _, label in events:
        counts[label] = counts.get(label, 0) + 1

    summary = {"video": name + ".MP4", "events": len(events),
               "by_class": counts, "draft": draft_path,
               "previews": preview_dir}
    return summary


if __name__ == "__main__":
    import sys
    v = sys.argv[1] if len(sys.argv) > 1 else r"C:\Projects\Traffic Computer Vision\video\C3905.MP4"
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join("..", "..", "labels_preview")
    drf = sys.argv[3] if len(sys.argv) > 3 else os.path.join("..", "..", "labels_draft")
    print(json.dumps(label_assist(v, out, drf), ensure_ascii=False, indent=2))