"""VISUAL validation tool for the failure_to_yield detector (PHASE 14).

Runs the EXACT production failure_to_yield pipeline on a chosen video (same
inference stack and stride as the smoke test) and renders, on every full-res
frame:

  * all tracked objects: bbox, track id, class;
  * current bottom-center position + recent trajectory trail;
  * the enabled crosswalk polygons (translucent overlay) — the zone the
    pedestrian must be crossing;
  * FAILURE-TO-YIELD evidence pairs (active): red line + pair ids + live
    metrics (FAILURE TO YIELD: TTC / Dist / MinPred / Closing / VehSpeed /
    PedSpeed / VehAccel / Braking);
  * YIELD / REJECTED pairs: grey line + the frame-level rejection reason
    (pedestrian_not_in_crosswalk / pedestrian_stationary / vehicle_not_moving /
    vehicle_yielding / ...);
  * top-N OTHER proximity vehicle<->pedestrian pairs (yellow) so the frame
    stays readable — TCV_FTY_VIS_TOP_N (default 5).
  * timestamp `TIME: XX.XXs` + a top status bar `FAILURE TO YIELD: <pair>` /
    `NO FAILURE TO YIELD`.

Usage:
    python debug/failure_to_yield_visual.py [video] [stride] [max_frames]

    TCV_FTY_VIS_PROFILE=default|selective   (default=default)
    TCV_FTY_VIS_TOP_N=5                     (default 5; proximity pairs only)
    TCV_DEVICE / TCV_IMGSZ / TCV_CONF      (same as every smoke test)

Profiles map onto the EXISTING FailureToYieldDetector constructor — no second
algorithm:
  default  : min_ped=18 min_veh=20 cw_margin=0 grace=1.5 braking=35
             yield_mem=1.5 max_ttc=3.5 min_closing=5 min_rel=8
             max_interaction=350 max_pred=120 approach_heading=90 ped_away=25
  selective: min_ped=25 min_veh=25 cw_margin=0 grace=1.5 braking=60
             yield_mem=1.0 max_ttc=2.5 min_closing=15 min_rel=20
             max_interaction=300 max_pred=80 approach_heading=75 ped_away=20

Outputs:
  video  debug/failure_to_yield_visual_<stem>[_<profile>].mp4
  csv    debug/failure_to_yield_candidates_<stem>[_<profile>].csv
and a console summary (PROFILE / VIDEO / raw vehicle-ped pairs / crosswalk
candidates / approach candidates / conflict-evidence frames / confirmed events
/ rejected + reasons / event list).

THE MARKED EVENTS ARE CANDIDATES, NOT GROUND TRUTH. Decide their category
(TRUE-LIKE / FALSE-LIKE / TRACKING ARTIFACT / UNCERTAIN) by WATCHING the
video, manually. This tool never auto-labels them.
"""

from __future__ import annotations

import csv
import math
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

from src.events.failure_to_yield import FailureToYieldDetector
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

PROFILES = {
    "default": dict(min_pedestrian_speed_px_s=18.0, min_vehicle_speed_px_s=20.0,
                    crosswalk_margin_px=0.0, stationary_grace_sec=1.5,
                    required_braking_response_px_s2=35.0, yield_memory_sec=1.5,
                    max_ttc_sec=3.5, min_closing_speed_px_s=5.0,
                    min_relative_speed_px_s=8.0, max_interaction_distance_px=350.0,
                    max_predicted_distance_px=120.0, min_pair_quality=0.2,
                    max_approach_heading_deg=90.0, pedestrian_away_angle_deg=25.0,
                    pair_expire_sec=2.0),
    "selective": dict(min_pedestrian_speed_px_s=25.0, min_vehicle_speed_px_s=25.0,
                      crosswalk_margin_px=0.0, stationary_grace_sec=1.5,
                      required_braking_response_px_s2=60.0, yield_memory_sec=1.0,
                      max_ttc_sec=2.5, min_closing_speed_px_s=15.0,
                      min_relative_speed_px_s=20.0, max_interaction_distance_px=300.0,
                      max_predicted_distance_px=80.0, min_pair_quality=0.25,
                      max_approach_heading_deg=75.0, pedestrian_away_angle_deg=20.0,
                      pair_expire_sec=2.0),
}

COL_BBOX = (0, 200, 0)
COL_TRAIL = (255, 220, 100)
COL_CW = (60, 200, 60)
COL_CONFIRMED = (0, 0, 255)
COL_REJECT = (130, 130, 130)
COL_PROX = (0, 220, 255)
COL_TEXT = (255, 255, 255)


def _text(img, text, pos, color, scale=0.6, thick=1, bgr_ref=None):
    if bgr_ref is not None:
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        cv2.rectangle(img, (int(pos[0]) - 2, int(pos[1]) - th - 4),
                      (int(pos[0]) + tw + 2, int(pos[1]) + 4),
                      (0, 0, 0), -1)
    cv2.putText(img, text, (int(pos[0]), int(pos[1])),
                cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def _draw_crosswalks(img, geometry):
    if not geometry.crosswalks:
        return
    overlay = img.copy()
    for poly in geometry.crosswalks:
        pts = [(int(p[0] * geometry.sx), int(p[1] * geometry.sy)) for p in poly]
        cv2.fillPoly(overlay, [np.array(pts, dtype=np.int32)], COL_CW)
        cv2.polylines(overlay, [np.array(pts, dtype=np.int32)], True, (0, 120, 0), 1)
        cx = int(sum(p[0] for p in pts) / len(pts))
        cy = int(sum(p[1] for p in pts) / len(pts))
        cv2.putText(overlay, "CROSSWALK", (cx - 40, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL_TEXT, 1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.18, img, 0.82, 0, img)


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
    ttc = rec["ttc_sec"]
    return [
        f"TTC: {ttc:.1f}s" if ttc is not None and math.isfinite(ttc) else "TTC: inf",
        f"Dist: {rec['distance_px']:.0f}px",
        f"MinPred: {rec['min_predicted_distance_px']:.0f}px",
        f"Closing: {rec['closing_speed_px_s']:.0f}px/s",
        f"VehSpd: {rec['veh_speed']:.0f}px/s" if rec["veh_speed"] is not None else "VehSpd: -",
        f"PedSpd: {rec['ped_speed']:.0f}px/s" if rec["ped_speed"] is not None else "PedSpd: -",
        f"VehAcc: {rec['veh_accel']:.0f}px/s^2" if rec["veh_accel"] is not None else "VehAcc: -",
        f"Braking: {'YES' if rec['veh_yielding'] else 'no'}",
    ]


def _pair_endpoints(rec, tracks):
    va = rec["veh_id"]
    pa = rec["ped_id"]
    ta = tracks.get(va)
    tb = tracks.get(pa)
    if ta is None or ta.last is None or tb is None or tb.last is None:
        return None, None
    return (ta.last.x, ta.last.bottom_y), (tb.last.x, tb.last.bottom_y)


def main() -> int:
    video = sys.argv[1] if len(sys.argv) > 1 else os.path.join(VIDEO_DIR, "C3905.MP4")
    stride = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    max_frames = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    profile = str(os.environ.get("TCV_FTY_VIS_PROFILE", "default")).lower()
    if profile not in PROFILES:
        print(f"unknown profile {profile!r}; using default")
        profile = "default"
    top_n = int(os.environ.get("TCV_FTY_VIS_TOP_N", "5"))

    cfg = dict(PROFILES[profile], min_on_duration=0.6, allowed_gap=0.6,
               merge_gap=1.2, min_duration=0.5)
    fty = FailureToYieldDetector(**cfg)

    stem = os.path.splitext(os.path.basename(video))[0]
    suffix = "" if profile == "default" else f"_{profile}"
    out_video = os.path.join(OUT_DIR, f"failure_to_yield_visual_{stem}{suffix}.mp4")
    out_csv = os.path.join(OUT_DIR, f"failure_to_yield_candidates_{stem}{suffix}.csv")

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

    agg: dict[str, dict] = {}
    raw_pairs: set[str] = set()
    cw_keys: set[str] = set()
    conflict_keys: set[str] = set()
    n_conflict_frames = 0
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
            if tr.label not in fty.candidate_labels:
                continue
            st = motion.update(tr, t_sec)
            if st is not None:
                states[tr.track_id] = st

        report = fty.update(tracks, states, geometry, t_sec)
        n_frames_seen += 1
        n_evidence_frames += int(report["evidence"])
        n_conflict_frames += len(report["active_pairs"])
        for key in report["active_pairs"]:
            conflict_keys.add(key)

        for key, rec in report["pairs"].items():
            raw_pairs.add(key)
            a = agg.setdefault(key, {
                "pair_id": key, "veh_id": rec["veh_id"], "ped_id": rec["ped_id"],
                "veh_class": rec["veh_class"], "ped_class": rec["ped_class"],
                "first_seen": INF, "last_seen": -INF,
                "conflict_first": None, "conflict_last": None,
                "confirmed": 0, "event_start": None, "event_end": None,
                "conflict_frames": 0, "crosswalk_frames": 0,
                "min_ttc": INF, "min_pred": INF, "min_dist": INF,
                "max_closing": 0.0, "max_rel": 0.0,
                "max_veh_speed": 0.0, "max_ped_speed": 0.0,
                "min_veh_accel": INF, "braking": False,
                "reject_reason": ""})
            a["first_seen"] = min(a["first_seen"], rec["first_t"])
            a["last_seen"] = max(a["last_seen"], rec["last_t"])
            if math.isfinite(rec["ttc_sec"]):
                a["min_ttc"] = min(a["min_ttc"], rec["ttc_sec"])
            a["min_pred"] = min(a["min_pred"], rec["min_predicted_distance_px"])
            a["min_dist"] = min(a["min_dist"], rec["distance_px"])
            a["max_closing"] = max(a["max_closing"], rec["closing_speed_px_s"])
            a["max_rel"] = max(a["max_rel"], rec["relative_speed_px_s"])
            if rec["veh_speed"] is not None:
                a["max_veh_speed"] = max(a["max_veh_speed"], rec["veh_speed"])
            if rec["ped_speed"] is not None:
                a["max_ped_speed"] = max(a["max_ped_speed"], rec["ped_speed"])
            if rec["veh_accel"] is not None:
                a["min_veh_accel"] = min(a["min_veh_accel"], rec["veh_accel"])
            a["braking"] = a["braking"] or bool(rec["veh_yielding"])
            if rec["ped_in_crosswalk"]:
                cw_keys.add(key)
                a["crosswalk_frames"] += 1
            if rec["conflict"]:
                a["conflict_frames"] += 1
                if a["conflict_first"] is None:
                    a["conflict_first"] = t_sec
                a["conflict_last"] = t_sec
        for reason_key, reason in report["rejected"].items():
            if reason_key in agg and not agg[reason_key]["reject_reason"]:
                agg[reason_key]["reject_reason"] = reason

        # ---- draw this frame ----------------------------------------------
        _draw_crosswalks(fr, geometry)
        for tr in tracks.values():
            _draw_track(fr, tr)

        def _draw_pair_rec(color, label, rec, metrics=True):
            pa, pb = _pair_endpoints(rec, tracks)
            if pa is None:
                return
            cv2.line(fr, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])),
                     color, 2, cv2.LINE_AA)
            _text(fr, label, (pa[0], pa[1] - 34), color, 0.6, 2, True)
            if metrics:
                m = _metrics(rec)
                bx = int(min(pa[0], pb[0]))
                base_y = int(max(pa[1], pb[1])) - 40
                for i, line in enumerate(m):
                    _text(fr, line, (bx, base_y - (len(m) - 1 - i) * 18),
                          color, 0.45, 1, True)
                _text(fr, f"{rec['pair']} {rec['veh_class']}{rec['veh_id']}->"
                          f"{rec['ped_class']}{rec['ped_id']}",
                      (bx, base_y + 22), color, 0.55, 1, True)

        active = set(report["active_pairs"])
        for key in sorted(active & set(report["pairs"])):
            _draw_pair_rec(COL_CONFIRMED, "FAILURE TO YIELD", report["pairs"][key])

        # YIELD / REJECTED pairs: in-crosswalk (or close) but not active
        for key, rec in report["pairs"].items():
            if key in active:
                continue
            if not (rec["ped_in_crosswalk"] or rec["reason"] is not None):
                continue
            pa, pb = _pair_endpoints(rec, tracks)
            if pa is None:
                continue
            cv2.line(fr, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])),
                     COL_REJECT, 1, cv2.LINE_AA)
            _text(fr, "YIELD / REJECTED", (pa[0], pa[1] - 30), COL_REJECT, 0.5, 1, True)
            reason = rec["reason"] or "?"
            _text(fr, f"{key} {reason}", ((pa[0] + pb[0]) / 2, max(pa[1], pb[1]) - 14),
                  COL_REJECT, 0.45, 1, True)

        # other proximity vehicle<->pedestrian pairs (yellow)
        prox = [(key, rec) for key, rec in report["pairs"].items()
                if key not in active and not rec["ped_in_crosswalk"]
                and rec["reason"] is None and rec["distance_px"] <= PROXIMITY_PX]
        prox.sort(key=lambda kv: kv[1]["distance_px"])
        for key, rec in prox[:top_n]:
            pa, pb = _pair_endpoints(rec, tracks)
            if pa is None:
                continue
            cv2.line(fr, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])),
                     COL_PROX, 1, cv2.LINE_AA)
            _text(fr, f"{key} near", ((pa[0] + pb[0]) / 2, max(pa[1], pb[1]) - 14),
                  COL_PROX, 0.45, 1, True)

        _text(fr, f"TIME: {t_sec:.2f}s", (12, 30), COL_TEXT, 0.7, 2, True)
        if active:
            _text(fr, "FAILURE TO YIELD: " + ",".join(sorted(active)[:4]),
                  (12, 56), COL_CONFIRMED, 0.7, 2, True)
        else:
            _text(fr, "NO FAILURE TO YIELD", (12, 56), (200, 200, 200), 0.7, 2, True)

        if writer.isOpened():
            writer.write(fr)
        idx += 1
        if idx % 600 == 0:
            print(f"  ... {idx} frames, {len(raw_pairs)} raw pairs, "
                  f"{len(conflict_keys)} approach candidates")
    cap.release()
    if wrote_video:
        writer.release()

    events = fty.finalize()
    # attribute confirmed events to pairs overlapping their conflict window
    for key, a in agg.items():
        if not a["conflict_frames"]:
            continue
        lo = a["conflict_first"]
        hi = a["conflict_last"]
        for s in events:
            if s.start <= hi and lo <= s.end:
                a["confirmed"] = 1
                a["event_start"] = s.start
                a["event_end"] = s.end
                break

    # ---- CSV ----------------------------------------------------------------
    rows = []
    for a in agg.values():
        if not (a["crosswalk_frames"] or a["conflict_frames"]):
            continue                  # geometry-relevant candidates only
        rows.append({
            "pair_id": a["pair_id"],
            "veh_id": a["veh_id"], "ped_id": a["ped_id"],
            "veh_class": a["veh_class"], "ped_class": a["ped_class"],
            "first_seen": round(a["first_seen"], 3),
            "last_seen": round(a["last_seen"], 3),
            "confirmed": a["confirmed"],
            "event_start": f"{a['event_start']:.3f}" if a["event_start"] is not None else "",
            "event_end": f"{a['event_end']:.3f}" if a["event_end"] is not None else "",
            "conflict_frames": a["conflict_frames"],
            "crosswalk_frames": a["crosswalk_frames"],
            "min_ttc": round(a["min_ttc"], 3) if math.isfinite(a["min_ttc"]) else "",
            "min_predicted_distance": round(a["min_pred"], 1),
            "min_current_distance": round(a["min_dist"], 1),
            "max_closing_speed": round(a["max_closing"], 1),
            "max_relative_speed": round(a["max_rel"], 1),
            "max_vehicle_speed": round(a["max_veh_speed"], 1),
            "max_pedestrian_speed": round(a["max_ped_speed"], 1),
            "min_vehicle_accel": round(a["min_veh_accel"], 1)
            if math.isfinite(a["min_veh_accel"]) else "",
            "braking_response": "yes" if a["braking"] else "no",
            "reject_reason": a["reject_reason"],
        })
    rows.sort(key=lambda r: (not r["confirmed"], r["first_seen"]))
    flds = ["pair_id", "veh_id", "ped_id", "veh_class", "ped_class",
            "first_seen", "last_seen", "confirmed", "event_start", "event_end",
            "conflict_frames", "crosswalk_frames", "min_ttc",
            "min_predicted_distance", "min_current_distance", "max_closing_speed",
            "max_relative_speed", "max_vehicle_speed", "max_pedestrian_speed",
            "min_vehicle_accel", "braking_response", "reject_reason"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=flds)
        w.writeheader()
        w.writerows(rows)
    print(f"csv written: {out_csv} ({len(rows)} candidate rows)")

    # ---- summary ------------------------------------------------------------
    rejected = Counter(a["reject_reason"] or "no_interaction" for a in agg.values()
                       if not a["conflict_frames"])
    print("\n" + "=" * 60)
    print("PROFILE:", profile)
    print("VIDEO:", os.path.basename(video))
    print("RAW VEHICLE-PEDESTRIAN PAIRS:", len(raw_pairs))
    print("CROSSWALK CANDIDATES (ped inside):", len(cw_keys))
    print("APPROACH CANDIDATES (conflict seen):", len(conflict_keys))
    print("CONFLICT-EVIDENCE FRAMES:", n_conflict_frames)
    print("TEMPORAL-EVIDENCE FRAMES:", n_evidence_frames)
    print("CONFIRMED FAILURE_TO_YIELD EVENTS:", len(events))
    print("UNIQUE CANDIDATE PAIRS (csv):", len(rows))
    print("REJECTED (never-conflict) PAIRS:", sum(1 for a in agg.values()
                                                  if not a["conflict_frames"]))
    for reason, n in rejected.most_common():
        print(f"    {reason}: {n}")
    print("CONFIRMED EVENTS (start -> end):")
    for s in events:
        pairs = ",".join(a["pair_id"] for a in agg.values()
                         if a["confirmed"] and a["event_start"] == s.start)
        print(f"  {round(s.start, 3)} -> {round(s.end, 3)}  failure_to_yield"
              + (f"  (pairs: {pairs})" if pairs else ""))
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