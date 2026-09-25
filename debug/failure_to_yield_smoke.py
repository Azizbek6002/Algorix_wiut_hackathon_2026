"""failure_to_yield smoke test on a real video (YOLO11x + full stack).

    python debug/failure_to_yield_smoke.py [video] [stride] [max_frames]

Defaults: C3905.MP4, stride 3, frame cap 0 (full video).

Prints, per vehicle<->pedestrian interplay:
  - RAW vehicle-pedestrian pairs (pair ids + classes + presence interval);
  - CROSSWALK candidates   (pair whose pedestrian bottom-center is inside a
                            configured crosswalk polygon at least once);
  - APPROACH candidates    (pair reaching `conflict` evidence at least once);
  - evidence frames        (temporal-engine active frames);
  - CONFIRMED events       (finalize() -> "failure_to_yield" segments);
  - REJECTED candidates    (pairs that never produced conflict, with the
                            frame-level rejection reason).

Per confirmed event the report prints the accumulated pair life-extrema:
TTC / current distance / min predicted distance / closing speed / vehicle
speed / pedestrian speed / vehicle acceleration (braking response law y) and
the crosswalk status (was the pedestrian inside the crosswalk?).

CANDIDATES ARE NOT GROUND TRUTH: thresholds are engineering estimates; every
candidate needs visual verification before trusting it
(debug/failure_to_yield_visual.py).

Env knobs (mirror the constructor): TCV_FTY_MIN_PED_SPEED TCV_FTY_MIN_VEH_SPEED
TCV_FTY_CW_MARGIN TCV_FTY_STATIONARY_GRACE TCV_FTY_BRAKING TCV_FTY_YIELD_MEMORY
TCV_FTY_MAX_TTC TCV_FTY_MIN_CLOSING TCV_FTY_MIN_REL TCV_FTY_MAX_INTERACTION
TCV_FTY_MAX_PRED TCV_FTY_QUALITY TCV_FTY_APPROACH_HEADING TCV_FTY_PED_AWAY
TCV_FTY_PAIR_EXPIRE (plus TCV_DEVICE/TCV_IMGSZ/TCV_CONF).
"""

from __future__ import annotations

import math
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2

from src.events.failure_to_yield import FailureToYieldDetector
from src.detector import Detector
from src.geometry import Geometry
from src.motion import MotionEngine
from src.trajectory import Detection, TrajectoryEngine

VIDEO_DIR = r"C:\Projects\Traffic Computer Vision\video"
SCENE_CFG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "scene_config.json")

IMG_SZ = int(os.environ.get("TCV_IMGSZ", "800"))
CONF = float(os.environ.get("TCV_CONF", "0.25"))
INF = float("inf")


def _fmt(v, nd=1):
    if v is None:
        return "-"
    if isinstance(v, float) and math.isinf(v):
        return "inf"
    return f"{v:.{nd}f}"


def _env(name, default):
    return float(os.environ.get(name, str(default)))


def build_detector() -> FailureToYieldDetector:
    return FailureToYieldDetector(
        min_pedestrian_speed_px_s=_env("TCV_FTY_MIN_PED_SPEED", 18.0),
        min_vehicle_speed_px_s=_env("TCV_FTY_MIN_VEH_SPEED", 20.0),
        crosswalk_margin_px=_env("TCV_FTY_CW_MARGIN", 0.0),
        stationary_grace_sec=_env("TCV_FTY_STATIONARY_GRACE", 1.5),
        required_braking_response_px_s2=_env("TCV_FTY_BRAKING", 35.0),
        yield_memory_sec=_env("TCV_FTY_YIELD_MEMORY", 1.5),
        max_ttc_sec=_env("TCV_FTY_MAX_TTC", 3.5),
        min_closing_speed_px_s=_env("TCV_FTY_MIN_CLOSING", 5.0),
        min_relative_speed_px_s=_env("TCV_FTY_MIN_REL", 8.0),
        max_interaction_distance_px=_env("TCV_FTY_MAX_INTERACTION", 350.0),
        max_predicted_distance_px=_env("TCV_FTY_MAX_PRED", 120.0),
        min_pair_quality=_env("TCV_FTY_QUALITY", 0.2),
        max_approach_heading_deg=_env("TCV_FTY_APPROACH_HEADING", 90.0),
        pedestrian_away_angle_deg=_env("TCV_FTY_PED_AWAY", 25.0),
        pair_expire_sec=_env("TCV_FTY_PAIR_EXPIRE", 2.0))


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
    print("failure_to_yield thresholds: min_ped_speed="
          f"{os.environ.get('TCV_FTY_MIN_PED_SPEED', '18.0')}px/s min_veh_speed="
          f"{os.environ.get('TCV_FTY_MIN_VEH_SPEED', '20.0')}px/s cw_margin="
          f"{os.environ.get('TCV_FTY_CW_MARGIN', '0.0')}px grace="
          f"{os.environ.get('TCV_FTY_STATIONARY_GRACE', '1.5')}s braking="
          f"{os.environ.get('TCV_FTY_BRAKING', '35.0')}px/s^2 yield_mem="
          f"{os.environ.get('TCV_FTY_YIELD_MEMORY', '1.5')}s max_ttc="
          f"{os.environ.get('TCV_FTY_MAX_TTC', '3.5')}s min_closing="
          f"{os.environ.get('TCV_FTY_MIN_CLOSING', '5.0')}px/s min_rel="
          f"{os.environ.get('TCV_FTY_MIN_REL', '8.0')}px/s max_interaction="
          f"{os.environ.get('TCV_FTY_MAX_INTERACTION', '350.0')}px max_pred="
          f"{os.environ.get('TCV_FTY_MAX_PRED', '120.0')}px min_quality="
          f"{os.environ.get('TCV_FTY_QUALITY', '0.2')} approach_heading="
          f"{os.environ.get('TCV_FTY_APPROACH_HEADING', '90.0')}deg ped_away="
          f"{os.environ.get('TCV_FTY_PED_AWAY', '25.0')}deg")

    geometry = Geometry.from_json(SCENE_CFG, W, H)
    print(f"crosswalks enabled: {len(geometry.crosswalks)}")

    det = Detector(model_path=r"weights/yolo11x.pt", conf=CONF,
                   device=os.environ.get("TCV_DEVICE", "cuda:0"), imgsz=IMG_SZ)
    traj = TrajectoryEngine(keep_sec=4.0)
    motion = MotionEngine(window_s=0.8, noise_px=4.0, min_points=3)
    fty = build_detector()

    agg: dict[str, dict] = {}          # pair_key -> per-pair stats
    conflict_keys: set[str] = set()
    cw_keys: set[str] = set()
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
            a = agg.setdefault(key, {
                "pair": key, "veh_id": rec["veh_id"], "ped_id": rec["ped_id"],
                "veh_class": rec["veh_class"], "ped_class": rec["ped_class"],
                "first_t": rec["first_t"], "last_t": rec["last_t"],
                "min_ttc": INF, "min_pred": INF, "min_dist": INF,
                "max_closing": 0.0, "max_rel": 0.0,
                "max_veh_speed": 0.0, "max_ped_speed": 0.0,
                "min_veh_accel": INF, "braking": False,
                "ped_in_crosswalk": 0, "conflict": 0,
                "reason": None})
            a["first_t"] = min(a["first_t"], rec["first_t"])
            a["last_t"] = max(a["last_t"], rec["last_t"])
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
                a["ped_in_crosswalk"] += 1
            a["conflict"] += int(rec["conflict"])
            if rec["reason"] is not None:
                a["reason"] = rec["reason"]
        idx += 1
        if idx % 600 == 0:
            print(f"  ... {idx} frames, {len(agg)} raw pairs, "
                  f"{len(conflict_keys)} approach candidates")
    cap.release()

    events = fty.finalize()

    # per-pair conflict windows (active_pairs frames, allowed-gap 0.6s closing)
    runs: dict[str, list] = {}
    # (re-run the active_pairs stream from report history: we did not store it,
    #  so rebuild from the aggregated conflict flag + t)
    # simplest: attribute each event segment to pairs overlapping its span
    event_pairs: dict[int, list] = {}
    for s in events:
        if not agg:
            break
        hit = []
        for key, a in agg.items():
            if a["conflict"] and s.start <= a["last_t"] and a["first_t"] <= s.end:
                hit.append(key)
        event_pairs.setdefault(id(s), hit)

    rejected = Counter()
    rejected_list = [a for a in agg.values() if not a["conflict"]]
    for a in rejected_list:
        rejected[a["reason"] or "no_interaction"] += 1

    print(f"\nframes processed: {n_frames_seen}")
    print(f"raw vehicle-pedestrian pairs: {len(agg)}")
    print(f"crosswalk candidates (ped inside): {len(cw_keys)}")
    print(f"approach candidates (conflict seen): {len(conflict_keys)}")
    print(f"conflict-evidence frames: {n_conflict_frames}")
    print(f"temporal-evidence frames: {n_evidence_frames}")
    print(f"temporally confirmed failure_to_yield events: {len(events)}")
    for s in events:
        pairs = ", ".join(event_pairs.get(id(s), [])) or "(pair attribution n/a)"
        print(f"  [{round(s.start, 3)}, {round(s.end, 3)}] failure_to_yield "
              f"(pairs: {pairs})")
    if not events:
        print("  (none)")
    print(f"rejected candidates (never conflict): {len(rejected_list)}")
    for reason, n in rejected.most_common():
        print(f"    {reason}: {n}")

    print("\nconfirmed failure_to_yield candidates (pair life extrema):")
    if not events:
        print("  (none)")
    for s in events:
        for key in event_pairs.get(id(s), []):
            a = agg[key]
            print(f"  pair={a['pair']} vehicle=#{a['veh_id']} {a['veh_class']} "
                  f"pedestrian=#{a['ped_id']} {a['ped_class']} "
                  f"event=[{round(s.start, 3)}, {round(s.end, 3)}]s")
            print(f"      min_ttc={_fmt(a['min_ttc'])}s min_dist={_fmt(a['min_dist'])}px "
                  f"min_pred={_fmt(a['min_pred'])}px max_closing={_fmt(a['max_closing'])} px/s "
                  f"max_rel={_fmt(a['max_rel'])} px/s")
            print(f"      veh_speed_max={_fmt(a['max_veh_speed'])}px/s ped_speed_max="
                  f"{_fmt(a['max_ped_speed'])}px/s veh_accel_min={_fmt(a['min_veh_accel'], 0)}px/s^2 "
                  f"yield_response={'yes' if a['braking'] else 'no'} "
                  f"ped_in_crosswalk_frames={a['ped_in_crosswalk']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())