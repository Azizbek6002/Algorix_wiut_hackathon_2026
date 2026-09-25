"""Near-miss event detector (PHASE 12) — pairwise danger over the existing stack.

Definition: a near miss is a DANGEROUS approach of two road users whose
extrapolated trajectories pass very close to each other WITHOUT an actual
collision. The detector builds on the existing pairwise engine
(src/interaction.PairwiseInteractionEngine) as the SINGLE source of all pairwise
motion quantities (relative velocity / closing speed / TTC / closest approach /
heading difference) — nothing here is recomputed.

Pipeline per frame (strictly causal, t <= current_time):
    active tracks (positions + smoothed MotionStates)
      -> unordered candidate pairs (vehicle-vehicle, vehicle-person,
         vehicle-bicycle) via PairwiseInteractionEngine
      -> combination gates (NOT a single condition):
           dangerous approach  : approaching & closing_speed >= min_closing
                                & relative_speed >= min_relative (real motion)
           small TTC           : ttc <= max_ttc
           small predicted gap : collision_distance <
                                    min_predicted_distance <= max_min_predicted
           non-collision       : current distance > collision_distance AND
                                 predicted minimum distance > collision_distance
      -> frame-level near_miss evidence
      -> existing src/temporal.TemporalEventEngine (label "near_miss")

Collision / overlap exclusion (the pair life CUT-OFF, not an accident detector):
  * min_predicted_distance <= collision_distance  -> NOT near_miss
    (extrapolated paths actually meet => this is a potential collision)
  * current distance       <= collision_distance  -> NOT near_miss
    (already collocated / overlapping => post-contact, not a near miss)
  * once the pair keeps collocated/overlapping the evidence simply stops, so the
    temporal engine closes the run and no continuing near_miss is generated.

Pair identity is unordered: (A,B) == (B,A). One logical interaction per pair per
appearance: per-pair running state (life minima / collision-gated flag) is keyed
by the sorted track-id pair, is pruned when the pair is gone for pair_expire_sec,
so a NEW appearance always starts from a fresh state (no state bleed between
different pair lifetimes or between different pairs).

Deterministic: pure function of the fed frames + injected engines.
"""

from __future__ import annotations

import math
from ..tracking.interaction import PairInteraction, PairwiseInteractionEngine

from .temporal import EventSegment, TemporalEventEngine

# labels that can be part of a near miss
DEFAULT_VEHICLE_LABELS = frozenset({"car", "truck", "bus", "motorcycle"})
DEFAULT_VULNERABLE_LABELS = frozenset({"person", "bicycle"})

LABEL = "near_miss"

INF = float("inf")


def pair_key(a: int, b: int) -> str:
    """Unordered identity of a pair: (5,2) == (2,5) -> '2-5'."""
    lo, hi = (a, b) if a < b else (b, a)
    return f"{lo}-{hi}"


class NearMissDetector:
    """Frame-level near_miss evidence, temporally confirmed by TemporalEventEngine.

    All thresholds are configurable, none is hard-coded behind the API.
    """

    def __init__(self, max_ttc_sec: float = 3.0,
                 max_min_predicted_distance_px: float = 100.0,
                 collision_distance_px: float = 15.0,
                 min_closing_speed_px_s: float = 5.0,
                 min_relative_speed_px_s: float = 8.0,
                 min_pair_quality: float = 0.2,
                 vehicle_labels=DEFAULT_VEHICLE_LABELS,
                 vulnerable_labels=DEFAULT_VULNERABLE_LABELS,
                 pair_expire_sec: float = 2.0,
                 pairwise: PairwiseInteractionEngine | None = None,
                 temporal: TemporalEventEngine | None = None,
                 min_on_duration: float = 0.8, allowed_gap: float = 0.6,
                 merge_gap: float = 1.2, min_duration: float = 0.5):
        assert max_ttc_sec > 0.0
        assert max_min_predicted_distance_px > collision_distance_px >= 0.0
        assert min_closing_speed_px_s >= 0.0
        assert min_relative_speed_px_s >= 0.0
        self.max_ttc = float(max_ttc_sec)
        self.max_min_pred = float(max_min_predicted_distance_px)
        self.collision_dist = float(collision_distance_px)
        self.min_closing = float(min_closing_speed_px_s)
        self.min_relative = float(min_relative_speed_px_s)
        self.min_quality = float(min_pair_quality)
        self.vehicle_labels = frozenset(vehicle_labels)
        self.vulnerable_labels = frozenset(vulnerable_labels)
        self.candidate_labels = self.vehicle_labels | self.vulnerable_labels
        self.pair_expire_sec = float(pair_expire_sec)
        self.pairwise = pairwise if pairwise is not None else PairwiseInteractionEngine()
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
        self._pairs: dict[str, dict] = {}

    # ------------------------------------------------------------------ API
    def update(self, tracks, motion_states, geometry=None, t_sec: float = 0.0) -> dict:
        """Evaluate near_miss evidence for every relevant pair at time t_sec.

        Args:
            tracks:        dict {track_id -> TrackTrajectory} (position + label).
            motion_states: dict {track_id -> MotionState} (smoothed motion).
            geometry:      UNUSED (near miss is pairwise, scene-agnostic); kept
                           for a uniform detector signature.
            t_sec:         current time (causal).
        Returns per-frame report:
            {"t_sec", "evidence", "active_pairs", "pairs": {key: record}}
        and feeds ("near_miss", evidence) into the TemporalEventEngine.
        """
        items = self._build_items(tracks, motion_states)
        interactions = self.pairwise.pairs(items, t_sec) if len(items) >= 2 else []

        records: dict[str, dict] = {}
        active: list[str] = []
        for inter in interactions:
            rec = self._evaluate(inter, motion_states)
            key = pair_key(inter.track_id_a, inter.track_id_b)
            history = self._history(key, inter, t_sec)
            self._update_history(history, inter)
            history["collision_gated"] = history["collision_gated"] or rec["collision_flag"]
            rec["collision_gated"] = history["collision_gated"]
            rec["life"] = {"min_ttc": history["min_ttc"],
                           "min_predicted_distance_px": history["min_pred"],
                           "min_distance_px": history["min_distance"],
                           "max_closing_speed_px_s": history["max_closing"],
                           "min_ttc_closing_speed_px_s": history["ttc_closing"],
                           "min_ttc_heading_diff_deg": history["ttc_heading_diff"],
                           "min_ttc_time_sec": history["ttc_time"],
                           "first_t": history["first_t"], "last_t": history["last_t"]}
            records[key] = rec
            if rec["active"]:
                active.append(key)

        self._prune(t_sec)
        active.sort()
        self.temporal.update(LABEL, t_sec, evidence=bool(active))
        return {"t_sec": t_sec, "evidence": bool(active),
                "active_pairs": active, "pairs": records}

    def finalize(self) -> list[EventSegment]:
        """Return confirmed, non-overlapping near_miss segments."""
        return self.temporal.finalize()

    def reset(self) -> None:
        self.temporal.reset()
        self._pairs.clear()

    # ------------------------------------------------------------- candidate
    def _build_items(self, tracks, motion_states) -> list[tuple]:
        items = []
        for tid, tr in tracks.items():
            if tr.last is None:
                continue
            if tr.label not in self.candidate_labels:
                continue
            items.append((tid, tr.label, (tr.last.x, tr.last.bottom_y),
                          motion_states.get(tid)))
        return items

    def _relevant(self, label_a, label_b) -> bool:
        a = label_a in self.vehicle_labels
        b = label_b in self.vehicle_labels
        if a and b:
            return True
        if (a and label_b in self.vulnerable_labels) or \
           (b and label_a in self.vulnerable_labels):
            return True
        return False

    # ------------------------------------------------------------------ gates
    def _evaluate(self, inter: PairInteraction, motion_states) -> dict:
        """Combination gates -> per-pair record. Returns evidence + reason."""
        rec = {
            "pair": pair_key(inter.track_id_a, inter.track_id_b),
            "id_a": inter.track_id_a, "id_b": inter.track_id_b,
            "class_a": inter.class_a, "class_b": inter.class_b,
            "distance_px": inter.distance_px,
            "closing_speed_px_s": inter.closing_speed_px_s,
            "relative_speed_px_s": inter.relative_speed_px_s,
            "ttc_sec": inter.ttc_sec,
            "time_to_closest_approach_sec": inter.time_to_closest_approach_sec,
            "min_predicted_distance_px": inter.min_predicted_distance_px,
            "heading_difference_deg": inter.heading_difference_deg,
            "approaching": inter.approaching,
            "active": False, "reason": "ok", "collision_flag": False,
        }
        st_a = motion_states.get(inter.track_id_a)
        st_b = motion_states.get(inter.track_id_b)
        if st_a is None or st_b is None:
            rec["reason"] = "insufficient_history"
            return rec
        if not self._relevant(inter.class_a, inter.class_b):
            rec["reason"] = "not_relevant"
            return rec
        pair_q = min(st_a.quality, st_b.quality)
        if pair_q < self.min_quality:
            rec["reason"] = "low_quality"
            return rec
        if st_a.stationary and st_b.stationary:
            rec["reason"] = "stationary"
            return rec
        if not inter.approaching:
            rec["reason"] = "not_approaching"
            return rec
        if inter.closing_speed_px_s < self.min_closing:
            rec["reason"] = "below_min_closing"
            return rec
        if inter.relative_speed_px_s < self.min_relative:
            rec["reason"] = "below_min_relative"
            return rec
        if not math.isfinite(inter.ttc_sec) or inter.ttc_sec > self.max_ttc:
            rec["reason"] = "ttc_too_large"
            return rec
        if inter.min_predicted_distance_px > self.max_min_pred:
            rec["reason"] = "predicted_too_large"
            return rec
        # collision / overlap exclusion (the non-collision requirement)
        if inter.min_predicted_distance_px <= self.collision_dist:
            rec["reason"] = "collision_predicted"
            rec["collision_flag"] = True
            return rec
        if inter.distance_px <= self.collision_dist:
            rec["reason"] = "collision_distance"
            rec["collision_flag"] = True
            return rec
        rec["active"] = True
        return rec

    # ----------------------------------------------------------- pair history
    def _history(self, key: str, inter: PairInteraction, t_sec: float) -> dict:
        h = self._pairs.get(key)
        if h is None:
            h = {"min_ttc": INF, "min_pred": INF, "min_distance": INF,
                 "max_closing": 0.0, "ttc_closing": None, "ttc_heading_diff": None,
                 "ttc_time": None, "collision_gated": False,
                 "first_t": t_sec, "last_t": t_sec}
            self._pairs[key] = h
        return h

    def _update_history(self, h: dict, inter: PairInteraction) -> None:
        t = inter.t_sec
        h["last_t"] = t
        h["min_distance"] = min(h["min_distance"], inter.distance_px)
        if inter.closing_speed_px_s > 0.0:
            h["max_closing"] = max(h["max_closing"], inter.closing_speed_px_s)
            if inter.ttc_sec < h["min_ttc"]:
                h["min_ttc"] = inter.ttc_sec
                h["ttc_closing"] = inter.closing_speed_px_s
                h["ttc_heading_diff"] = inter.heading_difference_deg
                h["ttc_time"] = t
        if inter.min_predicted_distance_px < h["min_pred"]:
            h["min_pred"] = inter.min_predicted_distance_px

    def _prune(self, t_sec: float) -> None:
        if t_sec < 0.0:
            return
        stale = [k for k, h in self._pairs.items()
                 if t_sec - h["last_t"] > self.pair_expire_sec]
        for k in stale:
            del self._pairs[k]


def segments_to_events(segments: list[EventSegment]) -> list[list]:
    """EventSegments -> harness events [[start, end, label]]. (Part A glue.)"""
    return [s.to_list() for s in segments]