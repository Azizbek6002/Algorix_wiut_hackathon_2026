"""failure_to_yield event detector (PHASE 14).

Semantics (strict): a PEDESTRIAN is actively crossing a configured crosswalk
while a VEHICLE approaches that crossing and keeps driving without a sufficient
yielding/braking response, with an objectively dangerous pairwise state.

NOT deducible from "person in crosswalk AND car in crosswalk": the detector
combines, per frame and per (vehicle, pedestrian) pair:

  * pedestrian crossing state        - in crosswalk (+ optional buffer margin)
                                      AND in-motion (or a bounded stationary
                                      grace while still inside the crosswalk);
  * vehicle approach state           - vehicle is moving, pedestrian is ahead
                                      of the vehicle along its heading, pair
                                      is converging on the crosswalk region;
  * pairwise danger                  - ONLY from PairwiseInteractionEngine:
                                      approaching, closing speed, relative
                                      speed, TTC, distance, predicted gap;
  * yielding response                - vehicle accelerating/braking from
                                      MotionEngine: a hard braking response
                                      (`accel <= -required_braking_response`)
                                      or a recent full stop suppresses evidence.

All thresholds live in the constructor (none hard-coded, all changeable).
Crosswalk geometry comes ONLY from the existing Geometry API
(`is_in_crosswalk` on bottom-center points); `crosswalk_margin_px` only
widens the interaction area around the SAME crosswalk polygons (scene_config
is never touched). Pairwise magnitudes never recomputed here. Temporal
confirmation reuses the shared TemporalEventEngine (label "failure_to_yield").
Strictly causal: only t <= current.
"""

from __future__ import annotations

import math

from ..tracking.interaction import PairwiseInteractionEngine
from ..tracking.motion import MotionState
from .temporal import EventSegment, TemporalEventEngine

LABEL = "failure_to_yield"
INF = float("inf")
EPS = 1e-9
DEFAULT_HEAVY_LABELS = frozenset({"car", "truck", "bus"})
DEFAULT_PEDESTRIAN_LABEL = "person"

# canonical ordered rejection reasons (CSV / debug reports order-friendly)
REASONS = ("pedestrian_not_in_crosswalk", "pedestrian_stationary",
           "vehicle_not_moving", "vehicle_yielding",
           "low_pair_quality", "too_far", "moving_apart", "low_closing",
           "low_relative_speed", "large_ttc", "predicted_gap_too_large",
           "vehicle_past", "pedestrian_moving_away")


def pair_key(a, b) -> str:
    return f"{min(a, b)}-{max(a, b)}"


def _angle_between_deg(dx: float, dy: float, ux: float, uy: float) -> float | None:
    """Small unsigned angle between two vectors, in degrees (None if 0-length)."""
    n1 = math.hypot(dx, dy)
    n2 = math.hypot(ux, uy)
    if n1 <= EPS or n2 <= EPS:
        return None
    c = (dx * ux + dy * uy) / (n1 * n2)
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def _point_seg_dist_px(px: float, py: float, ax: float, ay: float,
                       bx: float, by: float) -> float:
    abx, aby = bx - ax, by - ay
    L2 = abx * abx + aby * aby
    if L2 <= EPS:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * abx + (py - ay) * aby) / L2
    t = max(0.0, min(1.0, t))
    return math.hypot(px - (ax + t * abx), py - (ay + t * aby))


def crosswalk_distance_px(geometry, point) -> float | None:
    """Distance (px, full-res) from `point` to the nearest authorized crosswalk
    polygon; 0 when inside one. None when no crosswalk is configured."""
    if not geometry.crosswalks:
        return None
    rp = geometry.to_ref(point)
    best = INF
    for poly in geometry.crosswalks:
        # translate ref polygon -> full-res pixels
        pts = [(float(p[0]) * geometry.sx, float(p[1]) * geometry.sy) for p in poly]
        inside = False
        j = len(pts) - 1
        x, y = rp[0] * geometry.sx, rp[1] * geometry.sy
        for i in range(len(pts)):
            xi, yi = pts[i]
            xj, yj = pts[j]
            if ((yi > y) != (yj > y)) and \
               (x < (xj - xi) * (y - yi) / (yj - yi if abs(yj - yi) > EPS else EPS) + xi):
                inside = not inside
            dseg = _point_seg_dist_px(x, y, xi, yi, xj, yj)
            best = min(best, dseg)
            j = i
        if inside:
            return 0.0
    return best


class FailureToYieldDetector:
    """Per-frame failure_to_yield evidence for (vehicle, pedestrian) pairs.

    Signature mirrors the other detectors: `update(tracks, motion_states,
    geometry, t_sec)` -> report; `finalize()` -> confirmed segments; `reset()`.
    Geometry is REQUIRED (crosswalk queries).
    """

    def __init__(
        self,
        min_pedestrian_speed_px_s: float = 18.0,
        min_vehicle_speed_px_s: float = 20.0,
        crosswalk_margin_px: float = 0.0,
        stationary_grace_sec: float = 1.5,
        required_braking_response_px_s2: float = 35.0,
        yield_memory_sec: float = 1.5,
        max_ttc_sec: float = 3.5,
        min_closing_speed_px_s: float = 5.0,
        min_relative_speed_px_s: float = 8.0,
        max_interaction_distance_px: float = 350.0,
        max_predicted_distance_px: float = 120.0,
        min_pair_quality: float = 0.2,
        max_approach_heading_deg: float = 90.0,
        pedestrian_away_angle_deg: float = 25.0,
        pair_expire_sec: float = 2.0,
        heavy_labels=frozenset(DEFAULT_HEAVY_LABELS),
        pedestrian_label: str = DEFAULT_PEDESTRIAN_LABEL,
        pairwise: PairwiseInteractionEngine | None = None,
        temporal: TemporalEventEngine | None = None,
        min_on_duration: float = 0.6,
        allowed_gap: float = 0.6,
        merge_gap: float = 1.2,
        min_duration: float = 0.5,
    ) -> None:
        self.min_ped_speed = float(min_pedestrian_speed_px_s)
        self.min_veh_speed = float(min_vehicle_speed_px_s)
        self.cw_margin = float(crosswalk_margin_px)
        self.stationary_grace = float(stationary_grace_sec)
        self.braking_resp = float(required_braking_response_px_s2)
        self.yield_memory = float(yield_memory_sec)
        self.max_ttc = float(max_ttc_sec)
        self.min_closing = float(min_closing_speed_px_s)
        self.min_rel = float(min_relative_speed_px_s)
        self.max_interaction_dist = float(max_interaction_distance_px)
        self.max_pred_dist = float(max_predicted_distance_px)
        self.min_quality = float(min_pair_quality)
        self.max_approach_heading = float(max_approach_heading_deg)
        self.ped_away_angle = float(pedestrian_away_angle_deg)
        self.pair_expire = float(pair_expire_sec)
        self.heavy_labels = frozenset(heavy_labels)
        self.ped_label = pedestrian_label
        self.candidate_labels = frozenset(heavy_labels) | {pedestrian_label}
        self.pairwise = pairwise if pairwise is not None else PairwiseInteractionEngine()
        self.temporal = temporal if temporal is not None else TemporalEventEngine(
            min_on_duration=min_on_duration, allowed_gap=allowed_gap,
            merge_gap=merge_gap, min_duration=min_duration)
        # per-pair history (bounded, pruned)
        self._hist: dict[str, dict] = {}
        # per-track causal memory
        self._ped_last_moved_cw: dict[int, float] = {}
        self._veh_yielding_at: dict[int, float] = {}

    # ------------------------------------------------------------------ API
    def update(self, tracks, motion_states, geometry=None, t_sec: float = 0.0) -> dict:
        items = []
        for tid, tr in tracks.items():
            if tr.last is None or tr.label not in self.candidate_labels:
                continue
            items.append((tid, tr.label, (tr.last.x, tr.last.bottom_y),
                          motion_states.get(tid)))
        interactions = self.pairwise.pairs(items, t_sec) if len(items) >= 2 else []

        records: dict[str, dict] = {}
        active: list[str] = []
        for inter in interactions:
            key = pair_key(inter.track_id_a, inter.track_id_b)
            rec = self._evaluate(inter, motion_states, geometry, t_sec, key)
            if rec is None:                      # not a vehicle<>pedestrian pair
                continue
            records[key] = rec
            if rec["conflict"]:
                active.append(key)

        self._prune(t_sec)
        active.sort()
        evidence = bool(active)
        self.temporal.update(LABEL, t_sec, evidence=evidence)
        return {"t_sec": t_sec, "evidence": evidence,
                "active_pairs": active, "pairs": records,
                "rejected": {k: h["reason"] for k, h in self._hist.items()}}

    def finalize(self) -> list[EventSegment]:
        return self.temporal.finalize()

    def reset(self) -> None:
        self.temporal.reset()
        self._hist.clear()
        self._ped_last_moved_cw.clear()
        self._veh_yielding_at.clear()

    # ------------------------------------------------------------- evaluate
    def _evaluate(self, inter, motion_states, geometry, t_sec, key):
        if geometry is None or not geometry.crosswalks:
            return None                      # detector needs crosswalk geometry
        # assign vehicle / pedestrian roles by class
        cls_a, cls_b = inter.class_a, inter.class_b
        if cls_a in self.heavy_labels and cls_b == self.ped_label:
            veh_id, ped_id = inter.track_id_a, inter.track_id_b
            veh_cls, ped_cls, veh_pos, ped_pos = cls_a, cls_b, inter.pos_a, inter.pos_b
        elif cls_b in self.heavy_labels and cls_a == self.ped_label:
            veh_id, ped_id = inter.track_id_b, inter.track_id_a
            veh_cls, ped_cls, veh_pos, ped_pos = cls_b, cls_a, inter.pos_b, inter.pos_a
        else:
            return None                          # heavy-heavy / irrelevant

        st_veh = motion_states.get(veh_id)
        st_ped = motion_states.get(ped_id)

        # ---- history -------------------------------------------------------
        h = self._hist.get(key)
        if h is None:
            h = {"first_t": t_sec, "last_t": t_sec, "reason": None}
            self._hist[key] = h
        h["last_t"] = t_sec

        conflict = True
        reason = None
        crossing = False
        veh_yielding = False

        # ---- 1. pedestrian crossing a crosswalk ----------------------------
        cw_only = geometry.is_in_crosswalk(ped_pos)
        if self.cw_margin > 0.0:
            d_cw = crosswalk_distance_px(geometry, ped_pos)
            ped_in_cw = cw_only or (d_cw is not None and d_cw <= self.cw_margin)
        else:
            ped_in_cw = cw_only
        if not ped_in_cw:
            conflict, reason = False, "pedestrian_not_in_crosswalk"
        else:
            ped_moving = st_ped is not None and st_ped.speed >= self.min_ped_speed
            if ped_moving:
                self._ped_last_moved_cw[ped_id] = t_sec
            last_moved = self._ped_last_moved_cw.get(ped_id)
            crossing = ped_moving or (
                last_moved is not None and t_sec - last_moved <= self.stationary_grace)
            if not crossing:
                conflict, reason = False, "pedestrian_stationary"

        # ---- 2. vehicle moving / yielding ----------------------------------
        # always compute the yielding response for reporting + the brake memory
        # (a car braking hard in front of a crosswalk is yielding regardless of
        #  the pedestrian gate that rejected the pair at this frame)
        if st_veh is not None:
            braking = st_veh.accel is not None \
                and st_veh.accel <= -self.braking_resp
            if braking:
                self._veh_yielding_at[veh_id] = t_sec
            last_yield = self._veh_yielding_at.get(veh_id, -INF)
            veh_yielding = braking or (t_sec - last_yield <= self.yield_memory)
        veh_moving = st_veh is not None and st_veh.speed >= self.min_veh_speed \
            and not st_veh.stationary
        if conflict and not veh_moving:
            conflict, reason = False, "vehicle_not_moving"
        elif conflict and veh_yielding:
            conflict, reason = False, "vehicle_yielding"

        # ---- 3. pairwise danger (only PairwiseInteractionEngine values) ----
        if conflict:
            q = min((st_veh.quality if st_veh is not None else 0.0),
                    (st_ped.quality if st_ped is not None else 0.0))
            if q < self.min_quality:
                conflict, reason = False, "low_pair_quality"
        if conflict and inter.distance_px > self.max_interaction_dist:
            conflict, reason = False, "too_far"
        if conflict and not inter.approaching:
            conflict, reason = False, "moving_apart"
        if conflict and inter.closing_speed_px_s < self.min_closing:
            conflict, reason = False, "low_closing"
        if conflict and inter.relative_speed_px_s < self.min_rel:
            conflict, reason = False, "low_relative_speed"
        if conflict and (not math.isfinite(inter.ttc_sec) or inter.ttc_sec > self.max_ttc):
            conflict, reason = False, "large_ttc"
        if conflict and inter.min_predicted_distance_px > self.max_pred_dist:
            conflict, reason = False, "predicted_gap_too_large"

        # ---- 4. conflict-geometry heading gates ----------------------------
        if conflict and st_veh is not None and st_veh.heading_deg is not None:
            ang = _angle_between_deg(
                ped_pos[0] - veh_pos[0], ped_pos[1] - veh_pos[1],
                st_veh.vx, -st_veh.vy)          # velocity dir (image coords)
            if ang is not None and ang > self.max_approach_heading:
                conflict, reason = False, "vehicle_past"
        if conflict and st_ped is not None and st_ped.heading_deg is not None:
            ang = _angle_between_deg(
                ped_pos[0] - veh_pos[0], ped_pos[1] - veh_pos[1],
                st_ped.vx, -st_ped.vy)
            if ang is not None and ang <= self.ped_away_angle:
                conflict, reason = False, "pedestrian_moving_away"

        if not conflict and reason is not None:
            h["reason"] = reason

        return self._rec(inter, t_sec, conflict, reason, st_veh, st_ped,
                         veh_id, ped_id, veh_cls, ped_cls,
                         ped_in_cw, crossing, veh_yielding, h)

    @staticmethod
    def _rec(inter, t_sec, conflict, reason, st_veh, st_ped,
             veh_id=None, ped_id=None, veh_cls=None, ped_cls=None,
             ped_in_cw=False, crossing=False, veh_yielding=False, h=None):
        return {
            "pair": pair_key(inter.track_id_a, inter.track_id_b),
            "id_a": inter.track_id_a, "id_b": inter.track_id_b,
            "veh_id": veh_id, "ped_id": ped_id,
            "veh_class": veh_cls, "ped_class": ped_cls,
            "distance_px": inter.distance_px,
            "closing_speed_px_s": inter.closing_speed_px_s,
            "relative_speed_px_s": inter.relative_speed_px_s,
            "ttc_sec": inter.ttc_sec,
            "min_predicted_distance_px": inter.min_predicted_distance_px,
            "heading_difference_deg": inter.heading_difference_deg,
            "approaching": inter.approaching,
            "veh_speed": st_veh.speed if st_veh is not None else None,
            "veh_accel": st_veh.accel if st_veh is not None else None,
            "ped_speed": st_ped.speed if st_ped is not None else None,
            "ped_heading_deg": st_ped.heading_deg if st_ped is not None else None,
            "veh_heading_deg": st_veh.heading_deg if st_veh is not None else None,
            "ped_in_crosswalk": bool(ped_in_cw),
            "crossing": bool(crossing),
            "veh_yielding": bool(veh_yielding),
            "conflict": bool(conflict),
            "reason": reason,
            "first_t": (h["first_t"] if h is not None else t_sec),
            "last_t": (h["last_t"] if h is not None else t_sec),
        }

    # ---------------------------------------------------------------- prune
    def _prune(self, t_sec: float) -> None:
        if t_sec < 0.0:
            return
        stale = [k for k, h in self._hist.items()
                 if t_sec - h["last_t"] > self.pair_expire]
        for k in stale:
            del self._hist[k]
        keep = t_sec - max(self.stationary_grace + 0.5,
                           self.yield_memory + 0.5)
        self._ped_last_moved_cw = {k: v for k, v in self._ped_last_moved_cw.items()
                                   if v >= keep}
        self._veh_yielding_at = {k: v for k, v in self._veh_yielding_at.items()
                                 if v >= keep - 0.5}