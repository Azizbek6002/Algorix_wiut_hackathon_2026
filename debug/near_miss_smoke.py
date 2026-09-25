"""near_miss smoke test on a real video (YOLO11x + ByteTrack + full stack).

    python debug/near_miss_smoke.py [video] [stride] [max_frames]

Defaults: C3905.MP4, stride 3, frame cap 0 (full video).

Prints, separately for every DANGEROUS or COLLISION-GATED pair:
  - unordered pair ids + classes;
  - pair presence interval (first/last frame seen);
  - near-miss evidence runs (start/end);
  - life minimum TTC, minimum predicted distance, minimum current distance,
    max closing speed + closing / heading-difference at the min-TTC moment;
  - whether the collision/overlap gate ever rejected this pair.
And aggregates: raw pair candidates, near_miss evidence frames, temporally
confirmed events, collision-gated pairs.

CANDIDATES ARE NOT GROUND TRUTH: the thresholds are engineering estimates and
every candidate needs visual verification before trusting it.

Note: GPU inference dominates the runtime (see TCV_DEVICE/TCV_IMGSZ envs).
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2

from src.detector import Detector
from src.geometry import Geometry
from src.motion import MotionEngine
from src.near_miss import NearMissDetector, pair_key
from src.trajectory import Detection, TrajectoryEngine

VIDEO_DIR = r"C:\Projects\Traffic Computer Vision\video"
SCENE_CFG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "scene_config.json")

IMG_SZ = int(os.environ.get("TCV_IMGSZ", "800"))
CONF = float(os.environ.get("TCV_CONF", "0.25"))
ALLOWED_GAP = 0.6
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
    print("near_miss thresholds: max_ttc="
          f"{os.environ.get('TCV_NM_MAX_TTC', '3.0')}s max_min_pred="
          f"{os.environ.get('TCV_NM_MAX_PRED', '100.0')}px min_closing="
          f"{os.environ.get('TCV_NM_MIN_CLOSING', '5.0')}px/s min_rel="
          f"{os.environ.get('TCV_NM_MIN_REL', '8.0')}px/s collision="
          f"{os.environ.get('TCV_NM_COLLISION', '15.0')}px")

    geometry = Geometry.from_json(SCENE_CFG, W, H)

    det = Detector(model_path=r"weights/yolo11x.pt", conf=CONF,
                   device=os.environ.get("TCV_DEVICE", "cuda:0"), imgsz=IMG_SZ)
    traj = TrajectoryEngine(keep_sec=4.0)
    motion = MotionEngine(window_s=0.8, noise_px=4.0, min_points=3)
    nm = NearMissDetector(
        max_ttc_sec=float(os.environ.get("TCV_NM_MAX_TTC", "3.0")),
        max_min_predicted_distance_px=float(os.environ.get("TCV_NM_MAX_PRED", "100.0")),
        min_closing_speed_px_s=float(os.environ.get("TCV_NM_MIN_CLOSING", "5.0")),
        min_relative_speed_px_s=float(os.environ.get("TCV_NM_MIN_REL", "8.0")),
        collision_distance_px=float(os.environ.get("TCV_NM_COLLISION", "15.0")),
        min_pair_quality=float(os.environ.get("TCV_NM_QUALITY", "0.2")))

    raw_pairs: set[str] = set()
    agg: dict[str, dict] = {}          # pair_key -> accumulated per-pair stats
    runs: dict[str, dict] = {}         # pair_key -> open evidence run
    finished_runs: dict[str, list] = {}  # pair_key -> closed runs
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

        # accumulate per-pair life stats (survives prunes via min/max over life)
        for key, rec in report["pairs"].items():
            raw_pairs.add(key)
            life = rec["life"]
            a = agg.setdefault(key, {
                "key": key, "id_a": rec["id_a"], "id_b": rec["id_b"],
                "class_a": rec["class_a"], "class_b": rec["class_b"],
                "first_t": life["first_t"], "last_t": life["last_t"],
                "min_ttc": INF, "min_pred": INF, "min_dist": INF,
                "max_closing": 0.0, "ttc_closing": None, "ttc_heading_diff": None,
                "evidence_runs": [], "collision_gated": False, "raw_reasons": {}})
            a["first_t"] = min(a["first_t"], life["first_t"])
            a["last_t"] = max(a["last_t"], life["last_t"])
            a["min_ttc"] = min(a["min_ttc"], life["min_ttc"])
            a["min_pred"] = min(a["min_pred"], life["min_predicted_distance_px"])
            a["min_dist"] = min(a["min_dist"], life["min_distance_px"])
            a["max_closing"] = max(a["max_closing"], life["max_closing_speed_px_s"])
            if life["min_ttc_closing_speed_px_s"] is not None and \
               life["min_ttc"] <= a["min_ttc"] + 1e-9:
                a["ttc_closing"] = life["min_ttc_closing_speed_px_s"]
                a["ttc_heading_diff"] = life["min_ttc_heading_diff_deg"]
            if rec["collision_gated"]:
                a["collision_gated"] = True
                collision_gated.add(key)
            a["raw_reasons"][rec["reason"]] = a["raw_reasons"].get(rec["reason"], 0) + 1

        # evidence runs per pair (gap-closed, mirror of the temporal engine)
        active = set(report["active_pairs"])
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
        n_evidence_frames += len(report["active_pairs"])
        idx += 1
        if idx % 600 == 0:
            print(f"  ... {idx} frames, {len(raw_pairs)} raw pairs, "
                  f"{n_evidence_frames} evidence frames")
    cap.release()

    for key, r in list(runs.items()):
        r["end"] = r.pop("last")
        finished_runs.setdefault(key, []).append(r)
    events = nm.finalize()

    # attach closed runs
    for key, runs_list in finished_runs.items():
        if key in agg:
            agg[key]["evidence_runs"] = runs_list

    interesting = [a for a in agg.values()
                   if a["evidence_runs"] or a["collision_gated"]]
    benign = [a for a in agg.values()
              if not a["evidence_runs"] and not a["collision_gated"]]

    print(f"\nframes processed: {n_frames_seen}")
    print(f"raw pair candidates: {len(raw_pairs)}")
    print(f"near_miss evidence frames: {n_evidence_frames}")
    print(f"temporally confirmed events: {len(events)}")
    for s in events:
        print(f"  [{round(s.start, 3)}, {round(s.end, 3)}] near_miss")
    if not events:
        print("  (none)")
    print(f"collision-gated pairs: {len(collision_gated)}")

    print("\nnear-miss candidates (evidence runs and/or collision-gated):")
    if not interesting:
        print("  (none)")
    for a in sorted(interesting, key=lambda a: a["first_t"]):
        runs_str = ", ".join(f"[{r['start']:.2f}, {r['end']:.2f}]"
                             for r in a["evidence_runs"]) or "(no run)"
        print(f"  pair={a['key']:<9} classes={a['class_a']}<>{a['class_b']:<9} "
              f"present=[{a['first_t']:.2f}, {a['last_t']:.2f}]")
        print(f"      evidence_runs: {runs_str}")
        print(f"      min_ttc={_fmt(a['min_ttc'])}s (closing "
              f"{_fmt(a['ttc_closing'])} px/s, heading_diff "
              f"{_fmt(a['ttc_heading_diff'])} deg) "
              f"min_pred={_fmt(a['min_pred'])}px "
              f"min_dist={_fmt(a['min_dist'])}px "
              f"max_closing={_fmt(a['max_closing'])} px/s "
              f"collision_gated={a['collision_gated']}")

    if benign:
        print(f"\nbenign raw pairs (no evidence, never collision-gated): "
              f"{len(benign)}")
        for a in sorted(benign, key=lambda a: a["first_t"]):
            reasons = sorted(a["raw_reasons"].items(),
                             key=lambda kv: -kv[1])
            top = ", ".join(f"{k}:{v}" for k, v in reasons[:3])
            print(f"  pair={a['key']:<9} classes={a['class_a']}<>{a['class_b']} "
                  f"present=[{a['first_t']:.2f}, {a['last_t']:.2f}] top: {top}")
    return 0


if __name__ == "__main__":
    sys.exit(main())