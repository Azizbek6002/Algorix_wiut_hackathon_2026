"""Illegal-turn event detector (PHASE 11) — intersection-anchored turn shape
analysis with an explicit UNKNOWN-aware allowed-turn policy.

The detector does NOT report "a 90deg turn happened" as illegal_turn. It first
establishes (deterministically, from trajectory + MotionEngine) that the vehicle
actually performed a TURN and only then consults the configured turn
regulations:

  tier 1  TURN CANDIDATE (motion fact, always computed):
    sufficient history (moving samples over min_history_duration with
    min_history_points entries) -> stable entry heading (circular mean of the
    oldest quarter of the window) and stable exit heading (circular mean of the
    newest quarter) -> substantial NET direction change
    |signed_change| in [min_heading_change_deg, max_heading_change_for_non_u_turn]
    with matching cumulative rotation (trajectory change, not a single heading
    jump) and travelled arc >= min_turn_distance, while the maneuver is tied to
    a configured intersection zone (inside it, or with at most pin_rot_deg of
    extra rotation after the last in-zone sample).

  tier 2  ILLEGAL-TURN EVIDENCE:
    only when the allowed-turn configuration classifies the PERFORMED kind
    ("left" / "right" / "straight") as forbidden for that intersection.
    When the configuration is absent or the zone entry is UNKNOWN -> the record
    keeps turn_candidate=True but allowed_status="UNKNOWN" and NO evidence is
    produced. Missing regulations never become a false illegal_turn.

Classification ("what", not "allowed or not"):
    kind "left"     -> signed change > 0 (counter-clockwise on the image),
    kind "right"    -> signed change < 0 (clockwise on the image),
    kind "straight" -> |signed change| < straight_below_deg,
    kind "u_turn"   -> |signed change| > max_heading_change_for_non_u_turn
                       (deferred to src/illegal_u_turn, never reported here),
    anything else   -> "other" (also never reported as illegal_turn).

    left/right here describe the observed motion in the MotionEngine heading
    convention (0 = +x image right, 90 = image top, math angles); they are a
    description of WHAT happened, NOT a claim that the move is legal or not.

Allowed-turn configuration (constructor `allowed_turns`):
    None                     -> every zone is UNKNOWN (default; scene_config has
                                no turn table and none may be invented),
    {"0": ["left", ...]}     -> rules per intersection-zone index (order of
                                `geometry.intersections`; "any" = all zones),
    {"0": "UNKNOWN"}         -> explicit unknown for a zone.
    Missing key / value "UNKNOWN" -> UNKNOWN -> never illegal.

Temporal confirmation is delegated to the existing src/temporal.
TemporalEventEngine (label "illegal_turn"); turn candidates themselves are
diagnostic and are never fed to the engine. This module has no temporal logic.

Deterministic: pure function of the window of moving samples; reset() clears
all per-track state.
"""

from __future__ import annotations

import math
from collections import deque

from ..scene.geometry import Geometry, point_in_polygon

from ..tracking.interaction import heading_difference_deg

from ..tracking.motion import MotionState

from .temporal import EventSegment, TemporalEventEngine

DEFAULT_VEHICLE_LABELS = frozenset({"car", "truck", "bus", "motorcycle"})

LABEL = "illegal_turn"


def _wrap_deg(a: float) -> float:
    """Wrap angle difference to [-180, 180)."""
    a = (a + 180.0) % 360.0
    if a < 0:
        a += 360.0
    return a - 180.0


def _circ_mean_deg(values) -> float:
    """Circular mean of headings in [0, 360) (math convention)."""
    x = sum(math.cos(math.radians(v)) for v in values)
    y = sum(math.sin(math.radians(v)) for v in values)
    m = math.hypot(x, y)
    if m < 1e-9:
        return 0.0
    return math.degrees(math.atan2(y, x)) % 360.0


class IllegalTurnDetector:
    """Frame-level turn / illegal_turn evidence for intersection zones.

    Tier 1 (turn_candidate) is a pure motion+geometry fact and is always
    computed when the turn shape holds; tier 2 (illegal_turn, active) requires
    a configured forbidden regulation.
    """

    def __init__(self, min_speed_px_s: float = 6.0, min_quality: float = 0.3,
                 min_history_duration: float = 1.0, min_history_points: int = 10,
                 window_sec: float = 3.5, min_heading_change_deg: float = 45.0,
                 max_heading_change_for_non_u_turn: float = 150.0,
                 straight_below_deg: float = 30.0, min_turn_distance: float = 150.0,
                 pin_rot_deg: float = 60.0, require_intersection: bool = True,
                 allowed_turns: dict | None = None,
                 vehicle_labels=DEFAULT_VEHICLE_LABELS,
                 temporal: TemporalEventEngine | None = None,
                 min_on_duration: float = 0.8, allowed_gap: float = 0.6,
                 merge_gap: float = 1.2, min_duration: float = 0.5):
        self.min_speed = float(min_speed_px_s)
        self.min_quality = float(min_quality)
        self.min_history_duration = float(min_history_duration)
        self.min_history_points = int(min_history_points)
        self.window_sec = float(window_sec)
        self.min_heading_change = float(min_heading_change_deg)
        self.max_heading_change = float(max_heading_change_for_non_u_turn)
        self.straight_below = float(straight_below_deg)
        self.min_turn_distance = float(min_turn_distance)
        self.pin_rot = float(pin_rot_deg)
        self.require_intersection = bool(require_intersection)
        self.allowed_turns = allowed_turns          # None => UNKNOWN everywhere
        self.vehicle_labels = frozenset(vehicle_labels)
        if temporal is not None:
            self.temporal = temporal                # shared engine (caller-owned)
        else:
            self.temporal = TemporalEventEngine(
                min_on_duration=min_on_duration, allowed_gap=allowed_gap,
                merge_gap=merge_gap, min_duration=min_duration,
                threshold_on=None, threshold_off=None)
        self.temporal.per_label.setdefault(LABEL, {})
        for key, val in (("min_on_duration", min_on_duration),
                         ("allowed_gap", allowed_gap),
                         ("merge_gap", merge_gap),
                         ("min_duration", min_duration)):
            self.temporal.per_label[LABEL].setdefault(key, val)
        self._win: dict[int, dict] = {}

    # ------------------------------------------------------------------ API
    def update(self, tracks, motion_states, geometry: Geometry,
               t_sec: float) -> dict:
        """Evaluate turn / illegal_turn evidence at time t_sec.

        Returns:
            {"t_sec", "evidence", "active_tracks", "turn_candidates",
             "tracks": {tid: record}} and feeds ("illegal_turn", evidence)
            into the TemporalEventEngine.
        """
        details: dict[int, dict] = {}
        for tid, state in motion_states.items():
            rec = self._evaluate_track(tid, state, tracks.get(tid),
                                       geometry, t_sec)
            details[tid] = rec
        active = sorted(tid for tid, rec in details.items() if rec["active"])
        candidates = sorted(tid for tid, rec in details.items()
                            if rec["turn_candidate"])
        self.temporal.update(LABEL, t_sec, evidence=bool(active))
        return {"t_sec": t_sec, "evidence": bool(active),
                "active_tracks": active, "turn_candidates": candidates,
                "tracks": details}

    def finalize(self) -> list[EventSegment]:
        """Return confirmed, non-overlapping illegal_turn segments."""
        return self.temporal.finalize()

    def reset(self) -> None:
        self.temporal.reset()
        self._win.clear()

    # ------------------------------------------------------------------ rules
    def _allowed_status(self, zone_key: str, kind: str) -> str:
        """'allowed' | 'forbidden' | 'UNKNOWN' for a performed manoeuvre kind."""
        if self.allowed_turns is None:
            return "UNKNOWN"
        rule = None
        if zone_key is not None:
            rule = self.allowed_turns.get(zone_key)
        if rule is None:
            rule = self.allowed_turns.get("any")
        if rule is None or rule == "UNKNOWN":
            return "UNKNOWN"
        allowed = set(rule)
        return "allowed" if kind in allowed else "forbidden"

    # --------------------------------------------------------------- reasons
    def _evaluate_track(self, tid: int, state: MotionState, tr,
                        geometry: Geometry, t_sec: float) -> dict:
        rec: dict = {"active": False, "reason": "ok", "turn_candidate": False,
                     "turn_kind": None, "allowed_status": None,
                     "in_intersection": False, "zone_index": None,
                     "heading_deg": state.heading_deg if state else None,
                     "speed_px_s": state.speed if state else None,
                     "stationary": state.stationary if state else None,
                     "quality": state.quality if state else None,
                     "entry_heading_deg": None, "exit_heading_deg": None,
                     "signed_change_deg": None, "cum_turn_deg": None,
                     "arc_px": None, "samples": 0, "window_span_s": None,
                     "zone_visited": False}
        if tr is None or tr.last is None:
            rec["reason"] = "no_trajectory"
            return rec
        if tr.label not in self.vehicle_labels:
            rec["reason"] = "not_vehicle"
            return rec
        if state is None or state.heading_deg is None:
            rec["reason"] = "no_heading"
            return rec
        if state.quality < self.min_quality:
            rec["reason"] = "low_quality"
            return rec
        if state.stationary:
            rec["reason"] = "stationary"
            return rec
        if state.speed < self.min_speed:
            rec["reason"] = "below_min_speed"
            return rec

        pos = (tr.last.x, tr.last.bottom_y)
        in_inter = geometry.is_in_intersection(pos)
        rec["in_intersection"] = in_inter
        rec["heading_deg"] = state.heading_deg
        rec["speed_px_s"] = state.speed
        rec["stationary"] = state.stationary
        rec["quality"] = state.quality

        win = self._win.setdefault(tid, {"samples": deque()})
        win["samples"].append((t_sec, state.heading_deg, pos, in_inter))
        cutoff = t_sec - self.window_sec
        while win["samples"] and win["samples"][0][0] < cutoff:
            win["samples"].popleft()

        samples = list(win["samples"])
        rec["samples"] = len(samples)
        rec["window_span_s"] = samples[-1][0] - samples[0][0]
        if len(samples) < self.min_history_points:
            rec["reason"] = "insufficient_history"
            return rec
        if rec["window_span_s"] < self.min_history_duration:
            rec["reason"] = "insufficient_history"
            return rec

        # ---- window scan: cumulative rotation, arc, intersection timing ----
        n = len(samples)
        cum = 0.0
        arc = 0.0
        prev_h = samples[0][1]
        prev_pos = samples[0][2]
        cum_before_inter = None
        cum_at_last_inter = None
        zone_idx = None
        zone_visited = False
        for i in range(1, n):
            t_i, h_i, p_i, iz_i = samples[i]
            cum += _wrap_deg(h_i - prev_h)
            arc += math.hypot(p_i[0] - prev_pos[0], p_i[1] - prev_pos[1])
            prev_h, prev_pos = h_i, p_i
            if iz_i:
                if not zone_visited:
                    zone_idx = _intersection_index(geometry, p_i)
                    cum_before_inter = cum     # rotation up to first in-zone sample
                zone_visited = True
                cum_at_last_inter = cum
        rec["zone_visited"] = zone_visited
        rec["zone_index"] = zone_idx
        rec["cum_turn_deg"] = cum
        rec["arc_px"] = arc

        # ---- stable entry / exit headings (quarter means) ------------------
        q = max(1, n // 4)
        entry = _circ_mean_deg([s[1] for s in samples[:q]])
        exit_ = _circ_mean_deg([s[1] for s in samples[-q:]])
        signed = _wrap_deg(exit_ - entry)
        reversal = heading_difference_deg(entry, exit_)
        rec["entry_heading_deg"] = entry
        rec["exit_heading_deg"] = exit_
        rec["signed_change_deg"] = signed
        rec["reversal_deg"] = reversal

        # ---- turn shape: substantial direction change, not noise ----------
        if abs(signed) < self.min_heading_change:
            rec["reason"] = "change_too_small"   # straight / lane change / noise
            return rec
        if abs(signed) > self.max_heading_change:
            rec["reason"] = "u_turn"             # deferred to illegal_u_turn
            return rec
        if abs(cum) < self.min_heading_change:
            rec["reason"] = "insufficient_curvature"
            return rec
        if arc < self.min_turn_distance:
            rec["reason"] = "arc_too_short"
            return rec

        # ---- intersection anchoring ---------------------------------------
        if self.require_intersection and not zone_visited:
            rec["reason"] = "outside_intersection"
            return rec
        if zone_visited:
            pinned = in_inter or abs(cum - (cum_at_last_inter or 0.0)) <= self.pin_rot
            if not pinned:
                rec["reason"] = "outside_intersection"
                return rec
            if (cum_before_inter is not None
                    and abs(cum_before_inter) >= self.min_heading_change):
                rec["reason"] = "outside_intersection"   # turn finished before entry
                return rec

        # ---- tier 1: turn candidate (motion fact) --------------------------
        kind = ("straight" if abs(signed) < self.straight_below
                else ("left" if signed > 0 else "right"))
        rec["turn_kind"] = kind
        rec["turn_candidate"] = True

        # ---- tier 2: configured regulation decides evidence ----------------
        status = self._allowed_status(str(zone_idx) if zone_idx is not None
                                      else None, kind)
        rec["allowed_status"] = status
        if status == "forbidden":
            rec["active"] = True
            rec["reason"] = "ok"
        elif status == "allowed":
            rec["reason"] = "allowed_turn"
        else:                                       # UNKNOWN: never illegal
            rec["reason"] = "turn_rule_unknown"
        return rec


def _intersection_index(geometry: Geometry, point) -> int | None:
    """Index of the configured intersection polygon containing `point`
    (first match, deterministic), scaled to frame space like other queries."""
    rp = geometry.to_ref(point)
    for i, poly in enumerate(geometry.intersections):
        if point_in_polygon(rp, poly):
            return i
    return None


def segments_to_events(segments: list[EventSegment]) -> list[list]:
    """EventSegments -> harness events [[start, end, label]]. (Part A glue.)"""
    return [s.to_list() for s in segments]