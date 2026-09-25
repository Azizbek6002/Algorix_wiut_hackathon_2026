"""illegal_turn smoke test on a real video (YOLO11x + ByteTrack + full stack).

    python debug/illegal_turn_smoke.py [video] [stride] [max_frames]

Defaults: C3905.MP4, stride 3, frame cap 0. Prints, separately:
  * turn candidates (tier-1 motion facts inside the configured intersection)
  * illegal_turn candidates (tier-2, only when a configured rule forbids the
    performed 'left'/'right'/'straight' manoeuvre)
with track id, start/end, initial/final heading, heading change, intersection
membership and allowed/forbidden/unknown status.

Optional env TCV_ALLOWED_TURNS: JSON dict, e.g. '{"0": ["right"]}'. Without a
configured regulation every turn candidate stays UNKNOWN and there can be NO
illegal-turn candidate BY DESIGN (missing rules never become a false event).

CANDIDATES ARE NOT CONFIRMED GROUND TRUTH: visual verification on the video is
required before trusting them.

Note: GPU inference dominates the runtime (see TCV_DEVICE/TCV_IMGSZ envs).
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2

from src.detector import Detector
from src.geometry import Geometry
from src.illegal_turn import IllegalTurnDetector
from src.motion import MotionEngine
from src.trajectory import Detection, TrajectoryEngine
from src.wrong_way import DEFAULT_VEHICLE_LABELS  # noqa: E402

VIDEO_DIR = r"C:\Projects\Traffic Computer Vision\video"
SCENE_CFG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "scene_config.json")

IMG_SZ = int(os.environ.get("TCV_IMGSZ", "800"))
CONF = float(os.environ.get("TCV_CONF", "0.25"))
ALLOWED_GAP = 0.6


def _close(run, out):
    if run and run["frames"]:
        run["end"] = run["last"]
        out.append(run)
        return {}
    return run


def main() -> int:
    video = sys.argv[1] if len(sys.argv) > 1 else os.path.join(VIDEO_DIR, "C3905.MP4")
    stride = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    max_frames = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    allowed = json.loads(os.environ.get("TCV_ALLOWED_TURNS", "null"))

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        print("cannot open", video)
        return 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"video={os.path.basename(video)} {W}x{H} fps={fps:.2f} frames={n_frames} "
          f"duration={n_frames / fps:.1f}s stride={stride} allowed_turns={allowed}")

    geometry = Geometry.from_json(SCENE_CFG, W, H)
    print(f"intersection zones: {len(geometry.intersections)} enabled")
    for z in geometry.intersections:
        xs = [p[0] for p in z]
        ys = [p[1] for p in z]
        print(f"  zone bbox (ref-space): x {min(xs):.0f}-{max(xs):.0f} "
              f"y {min(ys):.0f}-{max(ys):.0f} ({len(z)} pts)")

    det = Detector(model_path=r"weights/yolo11x.pt", conf=CONF,
                   device=os.environ.get("TCV_DEVICE", "cuda:0"), imgsz=IMG_SZ)
    traj = TrajectoryEngine(keep_sec=4.0)
    motion = MotionEngine(window_s=0.8, noise_px=4.0, min_points=3)
    turn = IllegalTurnDetector(allowed_turns=allowed)

    reasons: dict[str, int] = {}
    turn_runs: list[dict] = []
    illegal_runs: list[dict] = []
    cand_run: dict = {}
    ill_run: dict = {}
    n_turn_frames = 0
    n_ill_frames = 0
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
            if tr.label not in DEFAULT_VEHICLE_LABELS:
                continue
            st = motion.update(tr, t_sec)
            if st is not None:
                states[tr.track_id] = st

        report = turn.update(tracks, states, geometry, t_sec)
        for rec in report["tracks"].values():
            reasons[rec["reason"]] = reasons.get(rec["reason"], 0) + 1

        for tid in list(cand_run.keys()):
            c = cand_run[tid]
            if tid not in report["turn_candidates"] and t_sec - c["last"] >= ALLOWED_GAP:
                _close(c, turn_runs)
                cand_run.pop(tid)
        for tid in report["turn_candidates"]:
            rec = report["tracks"][tid]
            c = cand_run.setdefault(tid, {"tid": tid, "start": None, "last": None,
                                          "frames": [], "label": tracks[tid].label})
            if c["start"] is None:
                c["start"] = t_sec
            c["last"] = t_sec
            c["frames"].append({"t": t_sec, "entry": rec["entry_heading_deg"],
                                "exit": rec["exit_heading_deg"],
                                "change": rec["signed_change_deg"],
                                "arc": rec["arc_px"], "in_iz": rec["in_intersection"],
                                "zone": rec["zone_index"], "kind": rec["turn_kind"],
                                "status": rec["allowed_status"]})
        n_turn_frames += len(report["turn_candidates"])

        for tid in list(ill_run.keys()):
            c = ill_run[tid]
            if tid not in report["active_tracks"] and t_sec - c["last"] >= ALLOWED_GAP:
                _close(c, illegal_runs)
                ill_run.pop(tid)
        for tid in report["active_tracks"]:
            rec = report["tracks"][tid]
            c = ill_run.setdefault(tid, {"tid": tid, "start": None, "last": None,
                                         "frames": [], "label": tracks[tid].label})
            if c["start"] is None:
                c["start"] = t_sec
            c["last"] = t_sec
            c["frames"].append({"t": t_sec, "entry": rec["entry_heading_deg"],
                                "exit": rec["exit_heading_deg"],
                                "change": rec["signed_change_deg"], "arc": rec["arc_px"],
                                "in_iz": rec["in_intersection"], "zone": rec["zone_index"],
                                "kind": rec["turn_kind"], "status": rec["allowed_status"]})
        n_ill_frames += len(report["active_tracks"])
        idx += 1
        if idx % 300 == 0:
            print(f"  ... {idx} frames, {n_turn_frames} turn-candidate frames")
    cap.release()

    for tid in list(cand_run.keys()):
        _close(cand_run[tid], turn_runs)
    for tid in list(ill_run.keys()):
        _close(ill_run[tid], illegal_runs)

    segs = turn.finalize()
    turn_runs.sort(key=lambda c: c["start"])
    illegal_runs.sort(key=lambda c: c["start"])

    print(f"\nper-track evaluation reasons: {reasons}")
    print(f"turn-candidate frames: {n_turn_frames} | active illegal frames: {n_ill_frames}")
    print("illegal_turn events:", [[round(s.start, 3), round(s.end, 3), s.label]
                                   for s in segs] or "(none)")

    def summarize(run):
        fr = run["frames"]
        peak = max(fr, key=lambda f: abs(f["change"] or 0.0))
        mid = fr[len(fr) // 2]
        return (f"  tid={run['tid']:<4} label={run['label']:<9} "
                f"[{run['start']:.2f}, {run['end']:.2f}] "
                f"dur={run['end'] - run['start']:.2f}s frames={len(fr)}\n"
                f"      initial_h={peak['entry']:.0f} final_h={peak['exit']:.0f} "
                f"change={peak['change']:.0f} arc={peak['arc']:.0f}px "
                f"kind={peak['kind']} in_intersection={mid['in_iz']} "
                f"zone={peak['zone']} allowed={peak['status']}")

    print("\nturn candidates:")
    for run in turn_runs[:30]:
        print(summarize(run))
    if not turn_runs:
        print("  (none)")

    print("\nillegal-turn candidates:")
    for run in illegal_runs:
        print(summarize(run))
    if not illegal_runs:
        print("  (none)  <- no configured forbidden rule (allowed_turns=None), "
              "so no candidate may be declared illegal; run with "
              "TCV_ALLOWED_TURNS=... to test a hypothesis")
    return 0


if __name__ == "__main__":
    sys.exit(main())