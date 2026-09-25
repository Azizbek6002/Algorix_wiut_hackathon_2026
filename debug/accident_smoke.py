"""accident smoke test on a real video (YOLO11x + ByteTrack + full stack).

    python debug/accident_smoke.py [video] [stride] [max_frames]

Defaults: C3905.MP4, stride 3, frame cap 0 (full video).

Prints, per impact candidate (raw -> impact -> rejected/confirmed):
  - unordered pair ids + classes, pair presence interval;
  - impact evidence: min TTC, min current distance, min predicted distance,
    max closing speed, max relative speed (life extrema);
  - post-impact evidence signals reached (speed_drop / decel / heading_change /
    stop / collocation) + max speed drop + max heading change in degrees;
  - rejection reason for non-confirmed candidates.
And aggregates: raw pair candidates, impact candidates (entered impact),
impact-evidence frames, confirmed accident events, unique pairs,
rejected collision candidates (with reason histogram).

CANDIDATES ARE NOT GROUND TRUTH: thresholds are engineering estimates; every
candidate needs visual verification before trusting it (debug/accident_visual.py).

Env knobs: TCV_ACC_COLLISION TCV_ACC_TTC TCV_ACC_MIN_CLOSING TCV_ACC_MIN_REL
TCV_ACC_DROP TCV_ACC_DECEL TCV_ACC_HEADING TCV_ACC_SIGNALS TCV_ACC_POST_WINDOW
TCV_ACC_POST_STATIONARY TCV_ACC_PRE_SEP TCV_ACC_QUALITY (plus TCV_DEVICE/TCV_IMGSZ).
"""

from __future__ import annotations

import math
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2

from src.accident import AccidentDetector, pair_key
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
    print("accident thresholds: collision="
          f"{os.environ.get('TCV_ACC_COLLISION', '15.0')}px impact_ttc="
          f"{os.environ.get('TCV_ACC_TTC', '1.0')}s min_closing="
          f"{os.environ.get('TCV_ACC_MIN_CLOSING', '8.0')}px/s min_rel="
          f"{os.environ.get('TCV_ACC_MIN_REL', '10.0')}px/s min_drop="
          f"{os.environ.get('TCV_ACC_DROP', '30.0')}px/s min_decel="
          f"{os.environ.get('TCV_ACC_DECEL', '60.0')}px/s^2 min_heading="
          f"{os.environ.get('TCV_ACC_HEADING', '40.0')}deg signals="
          f"{os.environ.get('TCV_ACC_SIGNALS', '2')} pre_sep="
          f"{os.environ.get('TCV_ACC_PRE_SEP', '60.0')}px")

    geometry = Geometry.from_json(SCENE_CFG, W, H)

    det = Detector(model_path=r"weights/yolo11x.pt", conf=CONF,
                   device=os.environ.get("TCV_DEVICE", "cuda:0"), imgsz=IMG_SZ)
    traj = TrajectoryEngine(keep_sec=4.0)
    motion = MotionEngine(window_s=0.8, noise_px=4.0, min_points=3)
    accident = AccidentDetector(
        collision_distance_px=float(os.environ.get("TCV_ACC_COLLISION", "15.0")),
        impact_ttc_sec=float(os.environ.get("TCV_ACC_TTC", "1.0")),
        min_closing_speed_px_s=float(os.environ.get("TCV_ACC_MIN_CLOSING", "8.0")),
        min_relative_speed_px_s=float(os.environ.get("TCV_ACC_MIN_REL", "10.0")),
        min_speed_drop_px_s=float(os.environ.get("TCV_ACC_DROP", "30.0")),
        min_deceleration_px_s2=float(os.environ.get("TCV_ACC_DECEL", "60.0")),
        min_heading_change_deg=float(os.environ.get("TCV_ACC_HEADING", "40.0")),
        min_impact_signals=int(os.environ.get("TCV_ACC_SIGNALS", "2")),
        post_impact_window_sec=float(os.environ.get("TCV_ACC_POST_WINDOW", "2.5")),
        post_impact_stationary_sec=float(os.environ.get("TCV_ACC_POST_STATIONARY", "0.6")),
        pre_separation_px=float(os.environ.get("TCV_ACC_PRE_SEP", "60.0")),
        min_pair_quality=float(os.environ.get("TCV_ACC_QUALITY", "0.2")))

    agg: dict[str, dict] = {}          # pair_key -> accumulated per-pair stats
    impact_keys: set[str] = set()
    n_impact_frames = 0
    n_confirmed_frames = 0
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
        n_impact_frames += len(report["impact_pairs"])
        n_confirmed_frames += int(report["evidence"])
        for key in report["impact_pairs"]:
            impact_keys.add(key)

        for key, rec in report["pairs"].items():
            life = rec["life"]
            a = agg.setdefault(key, {
                "key": key, "id_a": rec["id_a"], "id_b": rec["id_b"],
                "class_a": rec["class_a"], "class_b": rec["class_b"],
                "first_t": rec["first_t"], "last_t": rec["last_t"],
                "min_ttc": INF, "min_pred": INF, "min_dist": INF,
                "max_closing": 0.0, "max_rel": 0.0,
                "speed_drop": 0.0, "heading_change": 0.0,
                "signals": set(), "max_dist_pre": 0.0, "contact": False})
            a["first_t"] = min(a["first_t"], rec["first_t"])
            a["last_t"] = max(a["last_t"], rec["last_t"])
            a["min_ttc"] = min(a["min_ttc"], life["min_ttc"])
            a["min_pred"] = min(a["min_pred"], life["min_pred"])
            a["min_dist"] = min(a["min_dist"], life["min_dist"])
            a["max_closing"] = max(a["max_closing"], life["max_closing"])
            a["max_rel"] = max(a["max_rel"], life["max_rel"])
            if life["speed_drop"]:
                a["speed_drop"] = max(a["speed_drop"], *life["speed_drop"].values())
            if life["heading_change"]:
                a["heading_change"] = max(a["heading_change"], *life["heading_change"].values())
            a["signals"] |= set(rec["signals"])
            a["max_dist_pre"] = max(a["max_dist_pre"], rec["max_dist_pre"])
            a["contact"] = a["contact"] or rec["contact"]
        idx += 1
        if idx % 600 == 0:
            print(f"  ... {idx} frames, {len(agg)} raw pairs, "
                  f"{len(impact_keys)} impact candidates")
    cap.release()

    events = accident.finalize()
    seg_by_key = {}
    for key in accident.confirmed_keys:
        if len(accident.confirmed_keys) == 1:
            seg_by_key[key] = [(s.start, s.end) for s in events]
        else:
            seg_by_key[key] = []

    rejected = Counter(accident.rejected_reasons.values())
    print(f"\nframes processed: {n_frames_seen}")
    print(f"raw pair candidates: {len(agg)}")
    print(f"impact candidates: {len(impact_keys)}")
    print(f"impact-evidence frames: {n_impact_frames}")
    print(f"confirmed-evidence frames: {n_confirmed_frames}")
    print(f"temporally confirmed accident events: {len(events)}")
    for s in events:
        print(f"  [{round(s.start, 3)}, {round(s.end, 3)}] accident")
    if not events:
        print("  (none)")
    print(f"unique pairs (all-time): {len(agg)}")
    print(f"rejected collision candidates: {len(impact_keys) - len(accident.confirmed_keys)}")
    for reason, n in rejected.most_common():
        print(f"    {reason}: {n}")

    confirmed = [a for a in agg.values() if a["key"] in accident.confirmed_keys]
    print("\nconfirmed accident candidates:")
    if not confirmed:
        print("  (none)")
    for a in sorted(confirmed, key=lambda a: a["first_t"]):
        spans = seg_by_key.get(a["key"], [])
        spans_str = ", ".join(f"[{s0:.2f}, {s1:.2f}]" for s0, s1 in spans) or "(see events list)"
        print(f"  pair={a['key']:<9} classes={a['class_a']}<>{a['class_b']:<9} "
              f"present=[{a['first_t']:.2f}, {a['last_t']:.2f}] event={spans_str}")
        print(f"      min_ttc={_fmt(a['min_ttc'])}s min_dist={_fmt(a['min_dist'])}px "
              f"min_pred={_fmt(a['min_pred'])}px max_closing={_fmt(a['max_closing'])} px/s "
              f"max_rel={_fmt(a['max_rel'])} px/s")
        print(f"      speed_drop={_fmt(a['speed_drop'])}px/s "
              f"heading_change={_fmt(a['heading_change'])}deg "
              f"max_dist_pre={_fmt(a['max_dist_pre'])}px contact={a['contact']} "
              f"signals={sorted(a['signals'])}")

    rejected_list = [a for a in agg.values()
                     if a["key"] in impact_keys and a["key"] not in accident.confirmed_keys]
    if rejected_list:
        print(f"\nrejected collision candidates ({len(rejected_list)}):")
        for a in sorted(rejected_list, key=lambda a: a["first_t"]):
            reason = accident.rejected_reasons.get(a["key"], "?")
            print(f"  pair={a['key']:<9} classes={a['class_a']}<>{a['class_b']} "
                  f"present=[{a['first_t']:.2f}, {a['last_t']:.2f}] "
                  f"contact={a['contact']} max_dist_pre={_fmt(a['max_dist_pre'])}px "
                  f"signals={sorted(a['signals'])} reject={reason}")
    return 0


if __name__ == "__main__":
    sys.exit(main())