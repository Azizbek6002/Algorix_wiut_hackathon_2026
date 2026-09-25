"""VISUAL validation tool for the accident detector (PHASE 13).

Runs the EXACT production accident pipeline on a chosen video (same inference
stack and stride as the smoke tests) and renders, on every full-res frame:

  * all tracked objects: bbox, track id, class;
  * current bottom-center position + recent trajectory trail;
  * CONFIRMED ACCIDENT pairs: red line + pair id/classes + live metrics
    (ACCIDENT, TTC / Dist / MinPred / Closing / SpeedDrop / HeadingChange)
    + post-impact signals + contact/decel state;
  * IMPACT CANDIDATES (collision candidate entered, pending confirmation):
    orange line + "ACCIDENT CANDIDATE" + the same metrics;
  * REJECTED collision candidates (phase reset): grey line + the rejection
    reason (no_contact / no_pre_separation / insufficient_impact_evidence);
  * top-N OTHER proximity pairs (yellow) so the frame stays readable —
    TCV_ACC_VIS_TOP_N (default 5).
  * timestamp `TIME: XX.XXs` + a top status bar `ACCIDENT: <pair>` /
    `ACCIDENT CANDIDATE: <pair>` / `NO ACCIDENT`.

Usage:
    python debug/accident_visual.py [video] [stride] [max_frames]

    TCV_ACC_VIS_PROFILE=default|selective   (default=default)
    TCV_ACC_VIS_TOP_N=5                     (default 5; proximity pairs only)
    TCV_DEVICE / TCV_IMGSZ / TCV_CONF      (same as every smoke test)

Profiles map onto the EXISTING AccidentDetector constructor — no second
algorithm:
  default  : collision=15px ttc=1.0s min_closing=8px/s min_rel=10px/s
             drop=30px/s decel=60px/s^2 heading=40deg signals=2 pre_sep=60px
  selective: collision=12px ttc=0.6s min_closing=15px/s min_rel=20px/s
             drop=60px/s decel=120px/s^2 heading=60deg signals=2 pre_sep=120px

Outputs:
  video  debug/accident_visual_<stem>[_<profile>].mp4
  csv    debug/accident_candidates_<stem>[_<profile>].csv
and a console summary (PROFILE / VIDEO / raw pairs / impact candidates /
impact-evidence frames / confirmed events / rejected + reasons / event list).

THE MARKED EVENTS ARE CANDIDATES, NOT GROUND TRUTH. Decide their category
(TRUE-LIKE / FALSE-LIKE / TRACKING ARTIFACT / UNCERTAIN) by WATCHING the
video, manually. This tool never auto-labels them (and never silently computes
accidents out of TTC/distance alone — see src/events/accident.py).
"""

from __future__ import annotations

import csv
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

from src.accident import AccidentDetector
from src.detector import Detector
from src.geometry import Geometry
from src.motion import MotionEngine
from src.trajectory import Detection, TrajectoryEngine

VIDEO_DIR = r"C:\Projects\Traffic Computer Vision\video"
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
SCENE_CFG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "scene_config.json")

IMG_SZ = int(os.environ.get("TCV_IMGSZ", "800"))
CONF = float(os.environ.get("TCV_CONF", "0.25"))
TRAIL_SEC = 1.2
TRAIL_MAX_PTS = 14
PROXIMITY_PX = 400.0
INF = float("inf")
SIGNAL_NAMES = ("speed_drop", "decel", "heading_change", "stop", "collocation")

PROFILES = {
    "default": dict(collision_distance_px=15.0, impact_ttc_sec=1.0,
                    min_closing_speed_px_s=8.0, min_relative_speed_px_s=10.0,
                    min_speed_drop_px_s=30.0, min_deceleration_px_s2=60.0,
                    min_heading_change_deg=40.0, min_impact_signals=2,
                    post_impact_window_sec=2.5, post_impact_stationary_sec=0.6,
                    pre_separation_px=60.0),
    "selective": dict(collision_distance_px=12.0, impact_ttc_sec=0.6,
                      min_closing_speed_px_s=15.0, min_relative_speed_px_s=20.0,
                      min_speed_drop_px_s=60.0, min_deceleration_px_s2=120.0,
                      min_heading_change_deg=60.0, min_impact_signals=2,
                      post_impact_window_sec=2.5, post_impact_stationary_sec=0.6,
                      pre_separation_px=120.0),
}

COL_BBOX = (0, 200, 0)
COL_TRAIL = (255, 220, 100)
COL_CONFIRMED = (0, 0, 255)
COL_CAND = (0, 140, 255)
COL_REJECT = (130, 130, 130)
COL_PROX = (0, 220, 255)


def _text(img, text, pos, color, scale=0.6, thick=1, bgr_ref=None):
    if bgr_ref is not None:
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        cv2.rectangle(img, (int(pos[0]) - 2, int(pos[1]) - th - 4),
                      (int(pos[0]) + tw + 2, int(pos[1]) + 4),
                      (0, 0, 0), -1)
    cv2.putText(img, text, (int(pos[0]), int(pos[1])),
                cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


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


def _metrics(rec):
    life = rec["life"]
    ttc = rec["ttc_sec"]
    return [
        f"TTC: {ttc:.1f}s" if ttc is not None and math.isfinite(ttc)
        else "TTC: inf",
        f"Dist: {rec['distance_px']:.0f} px",
        f"MinPred: {rec['min_predicted_distance_px']:.0f} px",
        f"Closing: {rec['closing_speed_px_s']:.0f} px/s",
        f"SpeedDrop: {life['speed_drop'] and max(life['speed_drop'].values()) or 0:.0f} px/s",
        f"HeadingCh: {life['heading_change'] and max(life['heading_change'].values()) or 0:.0f} deg",
    ]


def _pair_endpoints(rec, tracks):
    tr_a, tr_b = tracks.get(rec["id_a"]), tracks.get(rec["id_b"])
    if tr_a is None or tr_a.last is None or tr_b is None or tr_b.last is None:
        return None, None
    return (tr_a.last.x, tr_a.last.bottom_y), (tr_b.last.x, tr_b.last.bottom_y)


def main() -> int:
    video = sys.argv[1] if len(sys.argv) > 1 else os.path.join(VIDEO_DIR, "C3905.MP4")
    stride = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    max_frames = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    profile = str(os.environ.get("TCV_ACC_VIS_PROFILE", "default")).lower()
    if profile not in PROFILES:
        print(f"unknown profile {profile!r}; using default")
        profile = "default"
    top_n = int(os.environ.get("TCV_ACC_VIS_TOP_N", "5"))

    cfg = dict(PROFILES[profile], min_pair_quality=0.2, pair_expire_sec=3.0,
               min_on_duration=0.5, allowed_gap=0.6, merge_gap=1.2,
               min_duration=0.5)

    stem = os.path.splitext(os.path.basename(video))[0]
    suffix = "" if profile == "default" else f"_{profile}"
    out_video = os.path.join(OUT_DIR, f"accident_visual_{stem}{suffix}.mp4")
    out_csv = os.path.join(OUT_DIR, f"accident_candidates_{stem}{suffix}.csv")

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
    accident = AccidentDetector(**cfg)

    agg: dict[str, dict] = {}
    impact_runs: dict[str, dict] = {}
    raw_pairs: set[str] = set()
    impact_keys: set[str] = set()
    n_impact_frames = 0
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
            if tr.label not in accident.candidate_labels:
                continue
            st = motion.update(tr, t_sec)
            if st is not None:
                states[tr.track_id] = st

        report = accident.update(tracks, states, geometry, t_sec)
        n_frames_seen += 1

        impact = set(report["impact_pairs"])
        confirmed_impact = set(report["confirmed_pairs"])
        n_impact_frames += len(impact)

        # ---- per-pair aggregation -----------------------------------------
        for key, rec in report["pairs"].items():
            raw_pairs.add(key)
            a = agg.setdefault(key, {
                "pair_id": key, "class_a": rec["class_a"], "class_b": rec["class_b"],
                "first_seen": INF, "last_seen": -INF,
                "impact_start": None, "impact_end": None,
                "confirmed": 0, "event_start": None, "event_end": None,
                "min_ttc": INF, "min_pred": INF, "min_dist": INF,
                "max_closing": 0.0, "max_rel": 0.0,
                "max_speed_drop": 0.0, "max_heading_change": 0.0,
                "contact": 0, "max_dist_pre": 0.0,
                "signals": set(), "reject_reason": "", "impact_frames": 0})
            a["first_seen"] = min(a["first_seen"], max(rec["first_t"], 0.0))
            a["last_seen"] = max(a["last_seen"], rec["last_t"])
            life = rec["life"]
            if math.isfinite(life["min_ttc"]):
                a["min_ttc"] = min(a["min_ttc"], life["min_ttc"])
            a["min_pred"] = min(a["min_pred"], life["min_pred"])
            a["min_dist"] = min(a["min_dist"], life["min_dist"])
            a["max_closing"] = max(a["max_closing"], life["max_closing"])
            a["max_rel"] = max(a["max_rel"], life["max_rel"])
            if life["speed_drop"]:
                a["max_speed_drop"] = max(a["max_speed_drop"],
                                          *life["speed_drop"].values())
            if life["heading_change"]:
                a["max_heading_change"] = max(a["max_heading_change"],
                                              *life["heading_change"].values())
            a["contact"] = max(a["contact"], int(rec["contact"]))
            a["max_dist_pre"] = max(a["max_dist_pre"], rec["max_dist_pre"])
            a["signals"] |= set(rec["signals"])
        # impact runs per pair (mirror the detector phase, gap-bounded)
        for key in impact_runs:
            impact_runs[key]["last"] = t_sec
        for key in impact:
            impact_keys.add(key)
            r = impact_runs.get(key)
            if r is None:
                impact_runs[key] = {"start": t_sec, "last": t_sec}
                r = impact_runs[key]
            r["last"] = t_sec
            a = agg.get(key)
            if a is not None:
                a["impact_start"] = r["start"] if a["impact_start"] is None \
                    else a["impact_start"]
                a["impact_end"] = r["last"]
                a["impact_frames"] += 1
        for reason_key, reason in report["rejected"].items():
            if reason_key in agg:
                agg[reason_key]["reject_reason"] = reason

        # ---- draw this frame ----------------------------------------------
        for tr in tracks.values():
            _draw_track(fr, tr)

        def _draw_pair_rec(color, label, rec, key=""):
            pa, pb = _pair_endpoints(rec, tracks)
            if pa is None:
                return
            cv2.line(fr, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])),
                     color, 2, cv2.LINE_AA)
            _text(fr, label, (pa[0], pa[1] - 34), color, 0.6, 2, True)
            metrics = _metrics(rec)
            bx = int(min(pa[0], pb[0]))
            base_y = int(max(pa[1], pb[1])) - 40
            for i, line in enumerate(metrics):
                _text(fr, line, (bx, base_y - (len(metrics) - 1 - i) * 18),
                      color, 0.45, 1, True)
            _text(fr, f"{rec['pair'] or key} {rec['class_a']}-{rec['class_b']}",
                  (bx, base_y + 22), color, 0.55, 1, True)
            sig = ",".join(s for s in SIGNAL_NAMES if s in rec["signals"])
            if sig:
                _text(fr, f"sig: {sig}", (bx, base_y + 40),
                      (255, 255, 255), 0.42, 1, True)

        for key in sorted(confirmed_impact):
            if key in report["pairs"]:
                _draw_pair_rec(COL_CONFIRMED, "ACCIDENT", report["pairs"][key], key)
        for key in sorted(impact - confirmed_impact):
            if key in report["pairs"]:
                _draw_pair_rec(COL_CAND, "ACCIDENT CANDIDATE", report["pairs"][key], key)

        # rejected collision candidates (phase reset): grey marker
        for key, reason in report["rejected"].items():
            rec = report["pairs"].get(key)
            if rec is None or rec["collision_candidate"]:
                continue
            pa, pb = _pair_endpoints(rec, tracks)
            if pa is None:
                continue
            cv2.line(fr, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])),
                     COL_REJECT, 1, cv2.LINE_AA)
            _text(fr, "COLLISION CANDIDATE - REJECTED",
                  (pa[0], pa[1] - 30), COL_REJECT, 0.5, 1, True)
            _text(fr, f"{key} {reason}", ((pa[0] + pb[0]) / 2, max(pa[1], pb[1]) - 14),
                  COL_REJECT, 0.45, 1, True)

        # other proximity pairs (always idle, not rejected, close together)
        prox = [(key, rec) for key, rec in report["pairs"].items()
                if not rec["collision_candidate"] and key not in report["rejected"]
                and rec["distance_px"] <= PROXIMITY_PX]
        prox.sort(key=lambda kv: kv[1]["distance_px"])
        for key, rec in prox[:top_n]:
            pa, pb = _pair_endpoints(rec, tracks)
            if pa is None:
                continue
            cv2.line(fr, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])),
                     COL_PROX, 1, cv2.LINE_AA)
            _text(fr, f"{key} cand", ((pa[0] + pb[0]) / 2, max(pa[1], pb[1]) - 14),
                  COL_PROX, 0.45, 1, True)

        _text(fr, f"TIME: {t_sec:.2f}s", (12, 30), (255, 255, 255), 0.7, 2, True)
        if confirmed_impact:
            _text(fr, "ACCIDENT: " + ",".join(sorted(confirmed_impact)[:4]),
                  (12, 56), COL_CONFIRMED, 0.7, 2, True)
        elif impact:
            _text(fr, "ACCIDENT CANDIDATE: " + ",".join(sorted(impact)[:4]),
                  (12, 56), COL_CAND, 0.7, 2, True)
        else:
            _text(fr, "NO ACCIDENT", (12, 56), (200, 200, 200), 0.7, 2, True)

        if writer.isOpened():
            writer.write(fr)
        idx += 1
        if idx % 600 == 0:
            print(f"  ... {idx} frames, {len(raw_pairs)} raw pairs, "
                  f"{len(impact_keys)} impact candidates")
    cap.release()
    if wrote_video:
        writer.release()

    events = accident.finalize()
    for key in accident.confirmed_keys:
        if key not in agg:
            continue
        agg[key]["confirmed"] = 1
        a = agg[key]
        lo = a["impact_start"] if a["impact_start"] is not None else a["first_seen"]
        hi = a["impact_end"] if a["impact_end"] is not None else a["last_seen"]
        for s in events:
            if s.start <= hi and lo <= s.end:      # segment overlaps this pair's impact
                agg[key]["event_start"] = s.start
                agg[key]["event_end"] = s.end
                break

    # ---- CSV ----------------------------------------------------------------
    rows = []
    for a in agg.values():
        if not (a["impact_frames"] or a["confirmed"] or a["reject_reason"]):
            continue
        rows.append({
            "pair_id": a["pair_id"],
            "class_a": a["class_a"], "class_b": a["class_b"],
            "first_seen": round(a["first_seen"], 3),
            "last_seen": round(a["last_seen"], 3),
            "impact_start": f"{a['impact_start']:.3f}" if a["impact_start"] is not None else "",
            "impact_end": f"{a['impact_end']:.3f}" if a["impact_end"] is not None else "",
            "confirmed": a["confirmed"],
            "event_start": f"{a['event_start']:.3f}" if a["event_start"] is not None else "",
            "event_end": f"{a['event_end']:.3f}" if a["event_end"] is not None else "",
            "min_ttc": round(a["min_ttc"], 3) if math.isfinite(a["min_ttc"]) else "",
            "min_predicted_distance": round(a["min_pred"], 1),
            "min_current_distance": round(a["min_dist"], 1),
            "max_closing_speed": round(a["max_closing"], 1),
            "max_relative_speed": round(a["max_rel"], 1),
            "max_speed_drop": round(a["max_speed_drop"], 1),
            "max_heading_change": round(a["max_heading_change"], 1),
            "contact": a["contact"],
            "max_dist_pre": round(a["max_dist_pre"], 1),
            "signals": "|".join(sorted(a["signals"])),
            "reject_reason": a["reject_reason"],
        })
    rows.sort(key=lambda r: (r["confirmed"], -(r["contact"])))
    flds = ["pair_id", "class_a", "class_b", "first_seen", "last_seen",
            "impact_start", "impact_end", "confirmed", "event_start", "event_end",
            "min_ttc", "min_predicted_distance", "min_current_distance",
            "max_closing_speed", "max_relative_speed", "max_speed_drop",
            "max_heading_change", "contact", "max_dist_pre", "signals",
            "reject_reason"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=flds)
        w.writeheader()
        w.writerows(rows)
    print(f"csv written: {out_csv} ({len(rows)} candidate rows)")

    # ---- summary ------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PROFILE:", profile)
    print("VIDEO:", os.path.basename(video))
    print("RAW PAIR CANDIDATES:", len(raw_pairs))
    print("IMPACT CANDIDATES:", len(impact_keys))
    print("IMPACT-EVIDENCE FRAMES:", n_impact_frames)
    print("CONFIRMED ACCIDENT EVENTS:", len(events))
    print("UNIQUE CANDIDATE PAIRS (csv):", len(rows))
    print("REJECTED CANDIDATES:", sum(1 for r in rows if r["reject_reason"]))
    for reason in ("no_contact", "no_pre_separation", "insufficient_impact_evidence"):
        n = sum(1 for r in rows if r["reject_reason"] == reason)
        print(f"    {reason}: {n}")
    print("CONFIRMED EVENTS (start -> end):")
    for s in events:
        print(f"  {round(s.start, 3)} -> {round(s.end, 3)}  accident")
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