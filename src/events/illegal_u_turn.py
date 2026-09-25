"""Illegal U-turn event detector (PHASE 10B) — trajectory-shape, zone-anchored.

A U-turn is treated as a persistent REVERSAL of the motion direction inside (or
through) a calibrated u-turn zone: the vehicle enters the window travelling on
one heading, rotates ~180deg, and exits travelling the opposite way. The
detector deliberately does NOT apply the naive rule `heading_change > 120 ->
u-turn`, which confuses ordinary 90deg turns, lane changes, heading noise and
complex intersection movements.

Detector per vehicle track (frame-level evidence, strictly causal):

  1. history — a rolling window of recent *moving* samples (t, smoothed heading,
     bottom-center position, in-zone flag) of min_history_points samples
     spanning >= min_history_duration; nothing is decided on first frames.
  2. initial heading — circular mean of the OLDEST quarter of the window
     (robust: a single noisy measurement cannot move a multi-sample mean).
  3. possible reversal — cumulative SIGNED rotation across the window must
     reach |min_heading_change_deg| (curvature / shape gate: a 90deg turn
     leaves cum ~90 < 120 and never qualifies), and the trajectory must have
     travelled > min_turn_arc_px (minimum movement distance).
  4. reversal requirement — the NEWEST quarter's circular-mean heading must lie
     ~180deg (within target_reverse_heading_tolerance_deg) of the initial one.
  5. zone anchoring — at least one window sample inside a configured u-turn
     zone, the current position in a zone, or no more than pin_rot_deg of extra
     rotation after the last zone sample; a rotation already >= the turn
     threshold BEFORE the first zone sample disqualifies the maneuver.
  6. movement — evidence frames require speed >= min_speed_px_s and a
     non-stationary state (a parked/pivoting object is not a U-turn).
  7. noise — heading comes from the already-smoothed MotionState.heading_deg,
     quarter-means average it further, and TemporalEventEngine (min_on_duration
     + gap semantics) filters isolated blips. No temporal logic here.

Rationale for window-based (vs per-track latched baseline) design: the entry
heading self-updates as old samples age out, so consecutive U-turns by the same
vehicle stay detectable, restart-cost is zero, and there is no latched-baseline
drift to corrupt a later event.

Output: frame-level evidence fed with label "illegal_u_turn" into the existing
src/temporal.TemporalEventEngine; `finalize()` returns the confirmed event
segments. This module contains no temporal implementation of its own.

All thresholds live in the constructor / config; the decision path contains no
hard-coded constants.
"""

from __future__ import annotations

import math
from collections import deque

from ..scene.geometry import Geometry

from ..tracking.interaction import heading_difference_deg

from ..tracking.motion import MotionState

from .temporal import EventSegment, TemporalEventEngine

# YOLO classes that can plausibly perform a U-turn
DEFAULT_VEHICLE_LABELS = frozenset({"car", "truck", "bus", "motorcycle"})

LABEL = "illegal_u_turn"


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


class IllegalUTurnDetector:
    """Frame-level illegal_u_turn evidence, temporally confirmed by the
    existing TemporalEventEngine."""

    def __init__(self, min_speed_px_s: float = 6.0, min_quality: float = 0.3,
                 min_history_duration: float = 1.0, min_history_points: int = 10,
                 window_sec: float = 3.0, min_heading_change_deg: float = 120.0,
                 target_reverse_heading_tolerance_deg: float = 45.0,
                 min_turn_arc_px: float = 200.0, pin_rot_deg: float = 60.0,
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
        self.reverse_tol = float(target_reverse_heading_tolerance_deg)
        self.min_turn_arc = float(min_turn_arc_px)
        self.pin_rot = float(pin_rot_deg)
        self.vehicle_labels = frozenset(vehicle_labels)
        if temporal is not None:
            self.temporal = temporal          # shared engine (caller-owned)
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
        """Evaluate illegal_u_turn evidence at time t_sec.

        Args:
            tracks:       dict {track_id -> TrackTrajectory} (position + label).
            motion_states: dict {track_id -> MotionState} (smoothed motion).
            geometry:     shared camera-level Geometry (u-turn zones).
            t_sec:        current time (causal).
        Returns per-frame report:
            {"t_sec", "evidence", "active_tracks", "tracks": {tid: record}}
        and feeds ("illegal_u_turn", evidence) into the TemporalEventEngine.
        """
        details: dict[int, dict] = {}
        for tid, state in motion_states.items():
            rec = self._evaluate_track(tid, state, tracks.get(tid),
                                       geometry, t_sec)
            details[tid] = rec
        active = sorted(tid for tid, rec in details.items() if rec["active"])
        self.temporal.update(LABEL, t_sec, evidence=bool(active))
        return {"t_sec": t_sec, "evidence": bool(active),
                "active_tracks": active, "tracks": details}

    def finalize(self) -> list[EventSegment]:
        """Return confirmed, non-overlapping illegal_u_turn segments."""
        return self.temporal.finalize()

    def reset(self) -> None:
        self.temporal.reset()
        self._win.clear()

    # --------------------------------------------------------------- reasons
    def _evaluate_track(self, tid: int, state: MotionState, tr,
                        geometry: Geometry, t_sec: float) -> dict:
        rec: dict = {"active": False, "reason": "ok", "in_zone": False,
                     "heading_deg": state.heading_deg if state else None,
                     "speed_px_s": state.speed if state else None,
                     "stationary": state.stationary if state else None,
                     "quality": state.quality if state else None,
                     "entry_heading_deg": None, "exit_heading_deg": None,
                     "reversal_deg": None, "cum_turn_deg": None,
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
        in_zone = geometry.is_in_u_turn_zone(pos)
        rec["in_zone"] = in_zone
        rec["heading_deg"] = state.heading_deg
        rec["speed_px_s"] = state.speed
        rec["stationary"] = state.stationary
        rec["quality"] = state.quality

        win = self._win.setdefault(tid, {"samples": deque()})
        win["samples"].append((t_sec, state.heading_deg, pos, in_zone))
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

        # ---- scan window: cumulative rotation, travelled arc, zone timing ---
        n = len(samples)
        cum = 0.0
        arc = 0.0
        prev_h = samples[0][1]
        prev_pos = samples[0][2]
        cum_before_zone = None
        cum_at_last_zone = None
        last_zone_i = -1
        zone_visited = False
        for i in range(1, n):
            t_i, h_i, p_i, iz_i = samples[i]
            cum += _wrap_deg(h_i - prev_h)
            arc += math.hypot(p_i[0] - prev_pos[0], p_i[1] - prev_pos[1])
            prev_h, prev_pos = h_i, p_i
            if iz_i:
                zone_visited = True
                if cum_before_zone is None:
                    cum_before_zone = cum     # rotation up to 1st zone sample
                cum_at_last_zone = cum
                last_zone_i = i
        rec["zone_visited"] = zone_visited
        rec["cum_turn_deg"] = cum
        rec["arc_px"] = arc

        if not zone_visited:
            rec["reason"] = "outside_u_turn_zone"
            return rec

        # ---- shape / curvature gates ---------------------------------------
        if abs(cum) < self.min_heading_change:
            rec["reason"] = "turn_too_small"
            return rec
        if arc < self.min_turn_arc:
            rec["reason"] = "arc_too_short"
            return rec

        # ---- zone anchoring -------------------------------------------------
        in_zone_now = in_zone
        post_zone_rot = abs(cum - (cum_at_last_zone or 0.0))
        pinned = in_zone_now or (zone_visited and post_zone_rot <= self.pin_rot)
        if not pinned:
            rec["reason"] = "outside_u_turn_zone"
            return rec
        # the big rotation must happen inside the zone, not before reaching it
        if cum_before_zone is not None and abs(cum_before_zone) >= self.min_heading_change:
            rec["reason"] = "outside_u_turn_zone"
            return rec

        # ---- reversal of direction ~180deg --------------------------------
        q = max(1, n // 4)
        entry = _circ_mean_deg([s[1] for s in samples[:q]])
        exit_ = _circ_mean_deg([s[1] for s in samples[-q:]])
        reversal = heading_difference_deg(entry, exit_)
        rec["entry_heading_deg"] = entry
        rec["exit_heading_deg"] = exit_
        rec["reversal_deg"] = reversal
        low = 180.0 - self.reverse_tol
        high = 180.0 + self.reverse_tol
        if reversal < low or reversal > high:
            rec["reason"] = "reversal_too_small"
            return rec

        rec["active"] = True
        return rec


def segments_to_events(segments: list[EventSegment]) -> list[list]:
    """EventSegments -> harness events [[start, end, label]]. (Part A glue.)"""
    return [s.to_list() for s in segments]