"""VISUAL validation tool for the near_miss detector (PHASE 12.1).

Runs the EXACT production near_miss pipeline on a chosen video (same inference
stack and stride as the smoke tests) and renders, on every full-res frame:

  * all tracked objects: bbox, track id, class;
  * current bottom-center position + recent trajectory trail;
  * near-miss EVIDENCE pairs: red line + pair id/classes + live metrics
    (NEAR MISS, TTC / MinPred / Dist / Closing / RelSpeed / Heading);
  * COLLISION-GATED pairs: orange line + pair id + MinPred (the non-collision
    gate rejected them);
  * top-N OTHER candidate pairs (yellow lines) so the frame stays readable —
    controlled by TCV_NM_VIS_TOP_N (default 5); raw candidates are NOT all drawn.
  * timestamp `TIME: XX.XXs` + a top status bar `NEAR_MISS: <pair>` / `NO NEAR MISS`.

Usage:
    python debug/near_miss_visual.py [video] [stride] [max_frames]

    TCV_NM_VIS_PROFILE=default|selective   (default=default)
    TCV_NM_VIS_TOP_N=5                     (default 5; other candidates only)
    TCV_DEVICE / TCV_IMGSZ / TCV_CONF      (same as every smoke test)

Profiles map onto the EXISTING NearMissDetector constructor — no second
algorithm:
  default   : max_ttc=3.0 max_pred=100 min_closing=5  min_relative=8
  selective : max_ttc=1.5 max_pred= 60 min_closing=8  min_relative=15

Outputs:
  video  debug/near_miss_visual_<stem>[_<profile>].mp4
  csv    debug/near_miss_candidates_<stem>[_<profile>].csv
and a console summary (PROFILE / VIDEO / raw candidates / evidence frames /
confirmed events / unique pairs / collision gated / event list).

THE MARKED EVENTS ARE CANDIDATES, NOT GROUND TRUTH. Decide their category
(TRUE-LIKE / FALSE-LIKE / TRACKING ARTIFACT / UNCERTAIN) by WATCHING the video,
manually. This tool never auto-labels them.
"""

from __future__ import annotations

import csv
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

from src.detector import Detector
from src.geometry import Geometry
from src.motion import MotionEngine
from src.near_miss import NearMissDetector
from src.trajectory import Detection, TrajectoryEngine

VIDEO_DIR = r"C:\Projects\Traffic Computer Vision\video"
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
SCENE_CFG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "scene_config.json")

IMG_SZ = int(os.environ.get("TCV_IMGSZ", "800"))
CONF = float(os.environ.get("TCV_CONF", "0.25"))
ALLOWED_GAP = 0.6
INF = float("inf")
TRAIL_SEC = 1.2
TRAIL_MAX_PTS = 14
COLLISION_REASONS = frozenset({"collision_predicted", "collision_distance"})

PROFILES = {
    "default": dict(max_ttc_sec=3.0, max_min_predicted_distance_px=100.0,
                    min_closing_speed_px_s=5.0, min_relative_speed_px_s=8.0),
    "selective": dict(max_ttc_sec=1.5, max_min_predicted_distance_px=60.0,
                      min_closing_speed_px_s=8.0, min_relative_speed_px_s=15.0),
}

COL_BBOX = (0, 200, 0)
COL_TRAIL = (255, 220, 100)
COL_EVID = (0, 0, 255)
COL_GATED = (0, 140, 255)
COL_CAND = (0, 220, 255)


def _text(img, text, pos, color, scale=0.6, thick=1, bgr_ref=None):
    if bgr_ref is not None:
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        cv2.rectangle(img, (int(pos[0]) - 2, int(pos[1]) - th - 4),
                      (int(pos[0]) + tw + 2, int(pos[1]) + 4),
                      (0, 0, 0), -1)
    cv2.putText(img, text, (int(pos[0]), int(pos[1])),
                cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def _danger_key(rec):
    """smallest is most dangerous: TTC first, then predicted gap."""
    ttc = rec["ttc_sec"]
    p = rec["min_predicted_distance_px"]
    t = ttc if (ttc is not None and math.isfinite(ttc)) else 1e6
    return (t, p if p is not None else INF)


def _draw_track(img, tr):
    if tr.last is None:
        return
    pts = [p for p in tr.points if p.t >= tr.last.t - TRAIL_SEC][-TRAIL_MAX_PTS:]
    if len(pts) >= 2:
        line = np.array([(p.x, p.bottom_y) for p in pts], dtype=np.int32)
        cv2.polylines(img, [line], False, COL_TRAIL, 1, cv2.LINE_AA)
    for p in pts[-1:]:
        cv2.circle(img, (int(p.x), int(p.bottom_y)), 4, COL_TRAIL, -1)
    x1, y1, x2, y2 = tr.last.xyxy
    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), COL_BBOX, 2)
    _text(img, f"#{tr.track_id} {tr.label}",
          (x1, y1 - 6), COL_BBOX, 0.55, 1, True)


def main() -> int:
    video = sys.argv[1] if len(sys.argv) > 1 else os.path.join(VIDEO_DIR, "C3905.MP4")
    stride = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    max_frames = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    profile = str(os.environ.get("TCV_NM_VIS_PROFILE", "default")).lower()
    if profile not in PROFILES:
        print(f"unknown profile {profile!r}; using default")
        profile = "default"
    top_n = int(os.environ.get("TCV_NM_VIS_TOP_N", "5"))

    cfg = dict(PROFILES[profile], collision_distance_px=15.0,
               min_pair_quality=0.2, pair_expire_sec=2.0,
               min_on_duration=0.8, allowed_gap=0.6, merge_gap=1.2,
               min_duration=0.5)

    stem = os.path.splitext(os.path.basename(video))[0]
    suffix = "" if profile == "default" else f"_{profile}"
    out_video = os.path.join(OUT_DIR, f"near_miss_visual_{stem}{suffix}.mp4")
    out_csv = os.path.join(OUT_DIR, f"near_miss_candidates_{stem}{suffix}.csv")

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        print("cannot open", video)
        return 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"video={os.path.basename(video)} {W}x{H} fps={fps:.2f} frames={n_frames} "
          f"duration={n_frames / fps:.1f}s stride={stride} profile={profile} "
          f"top_n={top_n}")

    writer = cv2.VideoWriter(out_video, cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (W, H))
    wrote_video = writer.isOpened()
    if not wrote_video:
        print("WARNING: VideoWriter failed - video will NOT be written")

    geometry = Geometry.from_json(SCENE_CFG, W, H)
    det = Detector(model_path=r"weights/yolo11x.pt", conf=CONF,
                   device=os.environ.get("TCV_DEVICE", "cuda:0"), imgsz=IMG_SZ)
    traj = TrajectoryEngine(keep_sec=4.0)
    motion = MotionEngine(window_s=0.8, noise_px=4.0, min_points=3)
    nm = NearMissDetector(**cfg)

    agg: dict[str, dict] = {}
    runs: dict[str, dict] = {}
    finished_runs: dict[str, list] = {}
    raw_pairs: set[str] = set()
    collision_gated: set[str] = set()
    n_evidence_frames = 0
    n_frames_seen = 0
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
            if tr.label not in nm.candidate_labels:
                continue
            st = motion.update(tr, t_sec)
            if st is not None:
                states[tr.track_id] = st

        report = nm.update(tracks, states, geometry, t_sec)
        n_frames_seen += 1

        # ---- per-pair aggregation -----------------------------------------
        for key, rec in report["pairs"].items():
            raw_pairs.add(key)
            a = agg.setdefault(key, {
                "pair_id": key, "class_a": rec["class_a"], "class_b": rec["class_b"],
                "first_seen": INF, "last_seen": -INF,
                "min_ttc": INF, "min_pred": INF, "min_dist": INF,
                "max_closing": 0.0, "max_rel": 0.0, "min_hdiff": INF,
                "collision_gated": False, "evidence_frames": 0,
                "ev_start": None, "ev_end": None, "relevant_frames": 0})
            if rec["reason"] not in ("not_relevant", "insufficient_history"):
                a["relevant_frames"] += 1
            a["first_seen"] = min(a["first_seen"], t_sec)
            a["last_seen"] = max(a["last_seen"], t_sec)
            if rec["ttc_sec"] is not None and math.isfinite(rec["ttc_sec"]):
                a["min_ttc"] = min(a["min_ttc"], rec["ttc_sec"])
            a["min_pred"] = min(a["min_pred"], rec["min_predicted_distance_px"])
            a["min_dist"] = min(a["min_dist"], rec["distance_px"])
            a["max_closing"] = max(a["max_closing"], rec["closing_speed_px_s"])
            a["max_rel"] = max(a["max_rel"], rec["relative_speed_px_s"])
            hd = rec["heading_difference_deg"]
            if hd is not None:
                a["min_hdiff"] = min(a["min_hdiff"], hd)
            if rec["collision_gated"]:
                a["collision_gated"] = True
                collision_gated.add(key)

        active = set(report["active_pairs"])
        n_evidence_frames += len(active)
        for key in list(runs.keys()):
            if key not in active and t_sec - runs[key]["last"] >= ALLOWED_GAP:
                closed = runs[key]
                closed["end"] = closed.pop("last")
                finished_runs.setdefault(key, []).append(closed)
                del runs[key]
        for key in active:
            r = runs.get(key)
            if r is None:
                r = {"start": t_sec, "last": t_sec, "frames": 0}
                runs[key] = r
            r["last"] = t_sec
            r["frames"] += 1
        for key, r in runs.items():
            a = agg.get(key)
            if a is None:
                continue
            a["evidence_frames"] = r["frames"]
            if a["ev_start"] is None:
                a["ev_start"] = r["start"]
            a["ev_end"] = r["last"]

        # ---- draw this frame ----------------------------------------------
        for tr in tracks.values():
            _draw_track(fr, tr)

        ev_keys = sorted(active)
        for key in ev_keys:
            rec = report["pairs"][key]
            tr_a, tr_b = tracks.get(rec["id_a"]), tracks.get(rec["id_b"])
            if tr_a is None or tr_a.last is None or tr_b is None or tr_b.last is None:
                continue
            pa = (tr_a.last.x, tr_a.last.bottom_y)
            pb = (tr_b.last.x, tr_b.last.bottom_y)
            cv2.line(fr, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])),
                     COL_EVID, 2, cv2.LINE_AA)
            _text(fr, "NEAR MISS", (pa[0], pa[1] - 30), COL_EVID, 0.6, 2, True)
            ttc = rec["ttc_sec"]
            ttc_s = f"{ttc:.1f}" if ttc is not None and math.isfinite(ttc) else "inf"
            hd = rec["heading_difference_deg"]
            hd_s = f"{hd:.0f}" if hd is not None else "-"
            metrics = [
                f"TTC: {ttc_s}s",
                f"MinPred: {rec['min_predicted_distance_px']:.0f} px",
                f"Dist: {rec['distance_px']:.0f} px",
                f"Closing: {rec['closing_speed_px_s']:.0f} px/s",
                f"RelSpeed: {rec['relative_speed_px_s']:.0f} px/s",
                f"Heading: {hd_s} deg",
            ]
            bx = int(min(pa[0], pb[0]))
            base_y = int(max(pa[1], pb[1])) - 40
            for i, line in enumerate(metrics):
                _text(fr, line, (bx, base_y - (len(metrics) - 1 - i) * 18),
                      COL_EVID, 0.45, 1, True)
            _text(fr, f"{key} {rec['class_a']}-{rec['class_b']}",
                  (bx, base_y + 18), COL_EVID, 0.55, 1, True)

        for key, rec in report["pairs"].items():
            if rec["reason"] not in COLLISION_REASONS:
                continue
            tr_a, tr_b = tracks.get(rec["id_a"]), tracks.get(rec["id_b"])
            if tr_a is None or tr_a.last is None or tr_b is None or tr_b.last is None:
                continue
            pa = (tr_a.last.x, tr_a.last.bottom_y)
            pb = (tr_b.last.x, tr_b.last.bottom_y)
            cv2.line(fr, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])),
                     COL_GATED, 1, cv2.LINE_AA)
            _text(fr, f"{key} {rec['class_a']}-{rec['class_b']}",
                  ((pa[0] + pb[0]) / 2, max(pa[1], pb[1]) - 14), COL_GATED,
                  0.5, 1, True)
            _text(fr, "COLLISION GATED",
                  (pa[0], pa[1] - 30), COL_GATED, 0.55, 1, True)
            _text(fr, f"MinPred: {rec['min_predicted_distance_px']:.0f}px",
                  ((pa[0] + pb[0]) / 2, max(pa[1], pb[1]) - 26), COL_GATED,
                  0.45, 1, True)

        candidates = [(key, rec) for key, rec in report["pairs"].items()
                      if not rec["active"] and rec["reason"] not in COLLISION_REASONS
                      and rec["reason"] != "insufficient_history"
                      and rec["reason"] != "not_relevant"]
        candidates.sort(key=lambda kv: _danger_key(kv[1]))
        for key, rec in candidates[:top_n]:
            tr_a, tr_b = tracks.get(rec["id_a"]), tracks.get(rec["id_b"])
            if tr_a is None or tr_a.last is None or tr_b is None or tr_b.last is None:
                continue
            pa = (tr_a.last.x, tr_a.last.bottom_y)
            pb = (tr_b.last.x, tr_b.last.bottom_y)
            cv2.line(fr, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])),
                     COL_CAND, 1, cv2.LINE_AA)
            _text(fr, f"{key} cand", ((pa[0] + pb[0]) / 2, max(pa[1], pb[1]) - 14),
                  COL_CAND, 0.45, 1, True)

        _text(fr, f"TIME: {t_sec:.2f}s", (12, 30), (255, 255, 255), 0.7, 2, True)
        if ev_keys:
            _text(fr, "NEAR_MISS: " + ",".join(ev_keys[:4]),
                  (12, 56), COL_EVID, 0.7, 2, True)
        else:
            _text(fr, "NO NEAR MISS", (12, 56), (200, 200, 200), 0.7, 2, True)

        if writer.isOpened():
            writer.write(fr)
        idx += 1
        if idx % 600 == 0:
            print(f"  ... {idx} frames, {len(raw_pairs)} raw pairs, "
                  f"{n_evidence_frames} ev frames")
    cap.release()
    if wrote_video:
        writer.release()

    for key, r in list(runs.items()):
        r["end"] = r.pop("last")
        finished_runs.setdefault(key, []).append(r)
    for key, rs in finished_runs.items():
        if key in agg:
            a = agg[key]
            a["ev_start"] = min(a["ev_start"], rs[0]["start"]) if a["ev_start"] is not None \
                else rs[0]["start"]
            a["ev_end"] = max(a["ev_end"], rs[-1]["end"]) if a["ev_end"] is not None \
                else rs[-1]["end"]

    events = nm.finalize()

    # ---- CSV ----------------------------------------------------------------
    rows = []
    for a in agg.values():
        if a["relevant_frames"] == 0:
            continue
        rows.append({
            "pair_id": a["pair_id"],
            "class_a": a["class_a"], "class_b": a["class_b"],
            "first_seen": round(a["first_seen"], 3),
            "last_seen": round(a["last_seen"], 3),
            "evidence_start": f"{a['ev_start']:.3f}" if a["ev_start"] is not None else "",
            "evidence_end": f"{a['ev_end']:.3f}" if a["ev_end"] is not None else "",
            "min_ttc": round(a["min_ttc"], 3) if math.isfinite(a["min_ttc"]) else "",
            "min_predicted_distance": round(a["min_pred"], 1),
            "min_current_distance": round(a["min_dist"], 1),
            "max_closing_speed": round(a["max_closing"], 1),
            "max_relative_speed": round(a["max_rel"], 1),
            "min_heading_difference": round(a["min_hdiff"], 1)
            if math.isfinite(a["min_hdiff"]) else "",
            "collision_gated": int(a["collision_gated"]),
            "evidence_frames": a["evidence_frames"],
        })
    rows.sort(key=lambda r: (r["min_ttc"] if isinstance(r["min_ttc"], float) else 1e6,
                             r["min_predicted_distance"]))
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [
            "pair_id", "class_a", "class_b", "first_seen", "last_seen",
            "evidence_start", "evidence_end", "min_ttc", "min_predicted_distance",
            "min_current_distance", "max_closing_speed", "max_relative_speed",
            "min_heading_difference", "collision_gated", "evidence_frames"])
        w.writeheader()
        w.writerows(rows)
    print(f"csv written: {out_csv} ({len(rows)} pairs)")

    # ---- summary ------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PROFILE:", profile)
    print("VIDEO:", os.path.basename(video))
    print("RAW PAIR CANDIDATES:", len(raw_pairs))
    print("NEAR_MISS EVIDENCE FRAMES:", n_evidence_frames)
    print("CONFIRMED EVENTS:", len(events))
    print("UNIQUE PAIRS:", len(rows))
    print("COLLISION GATED:", len(collision_gated))
    print("CONFIRMED EVENTS (start -> end):")
    for s in events:
        print(f"  {round(s.start, 3)} -> {round(s.end, 3)}  near_miss")
    if not events:
        print("  (none)")
    if wrote_video:
        print(f"video written: {out_video}")
    print("=" * 60)
    print("CANDIDATES ARE NOT GROUND TRUTH - verify each one on the video "
          "and classify manually (TRUE-LIKE / FALSE-LIKE / TRACKING ARTIFACT / UNCERTAIN).")
    return 0


if __name__ == "__main__":
    sys.exit(main())