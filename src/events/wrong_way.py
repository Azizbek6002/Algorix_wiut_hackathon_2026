"""Wrong-way event detector (PHASE 9) — lane-specific, over the existing stack.

Pipeline per vehicle track per frame (raw evidence):

    position (bottom-center) -> lane id via Geometry.get_lane
      -> expected direction of THAT lane (Geometry.get_lane_direction)
      -> smoothed heading from src/motion.MotionState
      -> angular deviation between actual heading and expected lane direction
      -> gates: vehicle class, moving (speed >= min_speed, not stationary),
                heading reliability (quality), known lane, deviation gate,
                optional u-turn-zone suppression
      -> per-frame wrong_way evidence

The detector NEVER uses a global dominant-flow rule: direction is per-lane, and
objects whose lane is unknown are ignored entirely.

Temporal confirmation is delegated to the existing src/temporal.
TemporalEventEngine (label "wrong_way"): short jitter / single-frame blips stay
below min_on_duration and never become events, and an end-of-run is confirmed
through allowed_gap. The detector itself contains NO temporal logic.

HEADING CONVENTION (documented, do not "fix" silently):
  * src/motion.heading_deg         — math convention: 0 = +x (image right),
                                    90 = +y image TOP, range [0, 360).
  * scene_config lane
    expected_direction             — IMAGE convention as drawn by
                                    debug/scene_geometry.py (y-down):
                                    90 points DOWN the image. The scene was
                                    calibrated visually in that convention.
  The detector converts the lane direction into the motion convention
  (math = (360 - image) % 360) before comparing. A vehicle driving toward the
  camera (down the lane, math heading ~270 with lane direction 90) is compliant.

U-turn handling: while an object is inside a calibrated u-turn zone, wrong_way
evidence is suppressed (a permitted reversal must not be reported as a full
wrong_way event); coupled with min_on_duration, transient reversal headings
during a turn do not produce events. The independent U-turn detector will not
be duplicated here.
"""

from __future__ import annotations

from ..scene.geometry import Geometry
from ..tracking.interaction import heading_difference_deg
from ..tracking.motion import MotionState
from .temporal import EventSegment, TemporalEventEngine

# YOLO classes that can plausibly drive against traffic
DEFAULT_VEHICLE_LABELS = frozenset({"car", "truck", "bus", "motorcycle"})

LABEL = "wrong_way"


def lane_direction_to_motion_deg(image_deg: float) -> float:
    """scene_config lane direction (image/y-down, as drawn by debug overlays)
    -> src/motion math convention (0 = +x, 90 = +y up)."""
    return (360.0 - (float(image_deg) % 360.0)) % 360.0


class WrongWayDetector:
    """Frame-level wrong_way evidence, temporally confirmed by TemporalEventEngine."""

    def __init__(self, min_speed_px_s: float = 4.0, angle_threshold: float = 120.0,
                 min_quality: float = 0.3,
                 vehicle_labels=DEFAULT_VEHICLE_LABELS,
                 suppress_in_u_turn_zone: bool = True,
                 temporal: TemporalEventEngine | None = None,
                 min_on_duration: float = 0.8, allowed_gap: float = 0.6,
                 merge_gap: float = 1.2, min_duration: float = 0.5):
        self.min_speed = float(min_speed_px_s)
        self.angle_threshold = float(angle_threshold)
        self.min_quality = float(min_quality)
        self.vehicle_labels = frozenset(vehicle_labels)
        self.suppress_u_turn = suppress_in_u_turn_zone
        if temporal is not None:
            self.temporal = temporal          # shared engine (caller-owned)
        else:
            self.temporal = TemporalEventEngine(
                min_on_duration=min_on_duration, allowed_gap=allowed_gap,
                merge_gap=merge_gap, min_duration=min_duration,
                threshold_on=None, threshold_off=None)
        # make the temporal params explicit for the label even when a shared
        # engine was supplied (per-label override, empty keys keep defaults)
        self.temporal.per_label.setdefault(LABEL, {})
        for key, val in (("min_on_duration", min_on_duration),
                         ("allowed_gap", allowed_gap),
                         ("merge_gap", merge_gap),
                         ("min_duration", min_duration)):
            self.temporal.per_label[LABEL].setdefault(key, val)

    # ------------------------------------------------------------------ API
    def update(self, tracks, motion_states, geometry: Geometry,
               t_sec: float) -> dict:
        """Evaluate wrong_way evidence at time t_sec.

        Args:
            tracks:       dict {track_id -> TrackTrajectory} (position + label).
            motion_states: dict {track_id -> MotionState} (smoothed motion).
            geometry:     shared camera-level Geometry (lane membership).
            t_sec:        current time (causal).
        Returns per-frame report:
            {"t_sec", "evidence", "active_tracks", "tracks": {tid: record}}
        and feeds ("wrong_way", evidence) into the TemporalEventEngine.
        """
        details: dict[int, dict] = {}
        for tid, state in motion_states.items():
            rec = self._evaluate_track(tid, state, tracks, geometry)
            details[tid] = rec
        active = sorted(tid for tid, rec in details.items() if rec["active"])
        self.temporal.update(LABEL, t_sec, evidence=bool(active))
        return {"t_sec": t_sec, "evidence": bool(active),
                "active_tracks": active, "tracks": details}

    def finalize(self) -> list[EventSegment]:
        """Return confirmed, non-overlapping wrong_way segments."""
        return self.temporal.finalize()

    def reset(self) -> None:
        self.temporal.reset()

    # --------------------------------------------------------------- reasons
    def _evaluate_track(self, tid: int, state: MotionState, tracks,
                        geometry: Geometry) -> dict:
        rec: dict = {"active": False, "reason": "ok", "lane": None,
                     "expected_deg": None, "heading_deg": None,
                     "deviation_deg": None, "speed_px_s": None,
                     "stationary": None, "quality": None, "in_u_turn_zone": False}
        tr = tracks.get(tid)
        if tr is None or tr.last is None:
            rec["reason"] = "no_trajectory"
            return rec
        if tr.label not in self.vehicle_labels:
            rec["reason"] = "not_vehicle"
            return rec
        rec["heading_deg"] = state.heading_deg
        rec["speed_px_s"] = state.speed
        rec["stationary"] = state.stationary
        rec["quality"] = state.quality

        if state.stationary:
            rec["reason"] = "stationary"
            return rec
        if state.speed < self.min_speed:
            rec["reason"] = "below_min_speed"
            return rec
        if state.quality < self.min_quality:
            rec["reason"] = "low_quality"
            return rec
        if state.heading_deg is None:
            rec["reason"] = "no_heading"
            return rec

        pos = (tr.last.x, tr.last.bottom_y)
        lane = geometry.get_lane(pos)
        rec["lane"] = lane
        if lane is None:
            rec["reason"] = "unknown_lane"
            return rec
        expected = geometry.get_lane_direction(lane)
        rec["expected_deg"] = expected
        if expected is None:
            rec["reason"] = "lane_no_direction"
            return rec

        expected_motion = lane_direction_to_motion_deg(expected)
        dev = heading_difference_deg(state.heading_deg, expected_motion)
        rec["deviation_deg"] = dev
        if dev is None or dev < self.angle_threshold:
            rec["reason"] = "below_threshold"
            return rec

        if self.suppress_u_turn and geometry.is_in_u_turn_zone(pos):
            rec["in_u_turn_zone"] = True
            rec["reason"] = "u_turn_zone"
            return rec

        rec["active"] = True
        return rec


def segments_to_events(segments: list[EventSegment]) -> list[list]:
    """EventSegments -> harness events [[start, end, label]]. (Part A glue.)"""
    return [s.to_list() for s in segments]