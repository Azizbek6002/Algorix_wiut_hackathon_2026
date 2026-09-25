"""Accident event detector (PHASE 13) — confirmed collision + post-impact evidence.

Definition: a near miss is a dangerous approach WITHOUT a confirmed collision;
an accident is a pair of road users that shows signs of an ACTUAL impact and
post-impact behavior. No single cue (TTC, distance, min_predicted_distance,
one overlapping frame, one abrupt speed change) is ever enough on its own —
confirmation needs a COMBINATION of independent signals.

Single source of all pairwise quantities is the existing
PairwiseInteractionEngine (relative velocity / closing speed / TTC / closest
approach / predicted minimum distance / heading difference); nothing here is
recomputed. Speed / acceleration / heading / stationary come from the existing
MotionEngine states.

                                PAIR CANDIDATE
                                      |
                          impact candidate (contact frame)
                                      |
        impact evidence (accumulated in a SHORT causal window):
          1. very small distance            (contact: d <= collision_distance)
          2. strong closing / convergence   (closing, TTC, min_pred)
          3. sudden deceleration            (MotionEngine accel)
          4. sudden heading change          (MotionEngine heading, wrap-aware)
          5. post-impact stop               (stationary for a while)
          6. persistent co-location
                                      |
                  >= min_impact_signals AND contact AND pre-separation
                                      |
                          TemporalEventEngine ("accident")

Tracking-overlap protection (the ID-2/ID-266 risk: two track IDs on ONE
physical object give distance~0 / min_pred~0 BUT it is not an accident):
  * a pair must have been SEPARATED before the contact (max distance seen
    before the impact onset >= pre_separation_px), otherwise it is treated as
    a duplicated/split track of one object;
  * both tracks must not be stationary (a static overlapping pair is not a
    crash);
  * at least min_impact_signals (>= 2) of {decel, speed drop, heading change,
    stop, sustained co-location} must accumulate inside the causal window;
  * an actual contact frame (distance <= collision_distance) is required —
    predicted-only convergence without contact stays a near miss.

Pair identity is unordered (A,B) == (B,A) via the shared sorted pair key; each
pair keeps only a SHORT bounded history (pre-impact snapshot + window minima),
and encounters are pruned after pair_expire_sec so a new appearance starts
fresh. The temporal layer is the existing TemporalEventEngine (label
"accident"); segments are non-overlapping, start < end, and causal (no future
frames). Deterministic — pure function of the fed frames + injected engines.
"""

from __future__ import annotations

import math

from ..tracking.interaction import PairInteraction, PairwiseInteractionEngine
from .near_miss import pair_key
from .temporal import EventSegment, TemporalEventEngine

# labels: an accident pair must contain a HEAVY vehicle; the other participant
# may be heavy, light (motorcycle) or a vulnerable user. Excluded by design:
# person-person, bicycle-bicycle, motorcycle-motorcycle, person-motorcycle ...
DEFAULT_HEAVY_LABELS = frozenset({"car", "truck", "bus"})
DEFAULT_LIGHT_LABELS = frozenset({"motorcycle"})
DEFAULT_VULNERABLE_LABELS = frozenset({"person", "bicycle"})

LABEL = "accident"

INF = float("inf")

COLLOCATION_FACTOR = 2.0   # "persistent co-location" distance (x collision dist)


def _wrap_delta_deg(a: float | None, b: float | None) -> float | None:
    """Signed smallest heading change |a-b| in [-180,180), wrap-aware
    (359 -> 1 is 2 degrees, NOT 358)."""
    if a is None or b is None:
        return None
    return (a - b + 180.0) % 360.0 - 180.0


def _new_state() -> dict:
    return {"phase": "idle", "onset_t": None, "pre_speed": {}, "pre_heading": {},
            "post": None, "max_dist_pre": 0.0, "contact": False, "signals": set(),
            "confirmed": False, "seen": False, "first_t": 0.0, "last_t": 0.0,
            "life": {"min_ttc": INF, "min_pred": INF, "min_dist": INF,
                     "max_closing": 0.0, "max_rel": 0.0,
                     "speed_drop": {}, "heading_change": {}}}


class AccidentDetector:
    """Confirmed-accident detector: impact evidence gates + TemporalEventEngine.

    All thresholds are configurable in the constructor; none are hard-coded
    inside the logic. `pairwise` and `temporal` engines are injectable.
    """

    def __init__(self,
                 collision_distance_px: float = 15.0,
                 impact_ttc_sec: float = 1.0,
                 min_closing_speed_px_s: float = 8.0,
                 min_relative_speed_px_s: float = 10.0,
                 min_pair_quality: float = 0.2,
                 min_speed_drop_px_s: float = 30.0,
                 min_deceleration_px_s2: float = 60.0,
                 min_heading_change_deg: float = 40.0,
                 post_impact_window_sec: float = 2.5,
                 post_impact_stationary_sec: float = 0.6,
                 min_impact_signals: int = 2,
                 pre_separation_px: float = 60.0,
                 max_impact_duration_sec: float = 12.0,
                 heavy_labels=DEFAULT_HEAVY_LABELS,
                 light_labels=DEFAULT_LIGHT_LABELS,
                 vulnerable_labels=DEFAULT_VULNERABLE_LABELS,
                 pair_expire_sec: float = 3.0,
                 pairwise: PairwiseInteractionEngine | None = None,
                 temporal: TemporalEventEngine | None = None,
                 min_on_duration: float = 0.5, allowed_gap: float = 0.6,
                 merge_gap: float = 1.2, min_duration: float = 0.5):
        assert min_impact_signals >= 1
        assert collision_distance_px >= 0.0
        assert pre_separation_px > collision_distance_px, \
            "pre-impact separation must exceed the collision radius"
        self.collision_dist = float(collision_distance_px)
        self.impact_ttc = float(impact_ttc_sec)
        self.min_closing = float(min_closing_speed_px_s)
        self.min_relative = float(min_relative_speed_px_s)
        self.min_quality = float(min_pair_quality)
        self.min_speed_drop = float(min_speed_drop_px_s)
        self.min_decel = float(min_deceleration_px_s2)
        self.min_heading_change = float(min_heading_change_deg)
        self.post_window = float(post_impact_window_sec)
        self.post_stationary = float(post_impact_stationary_sec)
        self.min_signals = int(min_impact_signals)
        self.pre_separation = float(pre_separation_px)
        self.max_impact_dur = float(max_impact_duration_sec)
        self.heavy_labels = frozenset(heavy_labels)
        self.light_labels = frozenset(light_labels)
        self.vulnerable_labels = frozenset(vulnerable_labels)
        self.candidate_labels = (self.heavy_labels | self.light_labels |
                                 self.vulnerable_labels)
        self.pair_expire_sec = float(pair_expire_sec)
        self.pairwise = pairwise if pairwise is not None else PairwiseInteractionEngine()
        if temporal is not None:
            self.temporal = temporal
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
        self.impact_pairs_count = 0       # unique pairs that entered impact
        self.rejected_reasons: dict[str, str] = {}
        self.confirmed_keys: set[str] = set()

    # ------------------------------------------------------------------ API
    def update(self, tracks, motion_states, geometry=None, t_sec: float = 0.0) -> dict:
        """Evaluate accident impact evidence for every relevant pair at t_sec.

        Args: see near_miss.NearMissDetector.update (geometry is unused here —
        accident is pairwise and scene-agnostic; kept for a uniform signature).
        Returns per-frame report:
            {"t_sec", "evidence", "impact_pairs", "confirmed_pairs",
             "pairs": {key: record}, "rejected": {key: reason}}
        and feeds ("accident", evidence) into the TemporalEventEngine.
        """
        items = self._build_items(tracks, motion_states)
        interactions = self.pairwise.pairs(items, t_sec) if len(items) >= 2 else []

        records: dict[str, dict] = {}
        impact_now: list[str] = []
        confirmed_active: list[str] = []
        for inter in interactions:
            key = pair_key(inter.track_id_a, inter.track_id_b)
            state = self._pairs.get(key)
            if state is None:
                state = _new_state()
                self._pairs[key] = state
            if not state["seen"]:
                state["seen"] = True
                state["first_t"] = t_sec
            state["last_t"] = t_sec

            rec = self._evaluate(inter, motion_states, state)

            # pre-impact separation: remembered while the pair is still idle
            if state["phase"] == "idle":
                state["max_dist_pre"] = max(state["max_dist_pre"],
                                            inter.distance_px)

            self._update_impact_state(state, inter, motion_states, t_sec, key)
            rec["phase"] = state["phase"]
            rec["confirmed"] = state["confirmed"]
            rec["signals"] = sorted(state["signals"])
            rec["life"] = dict(state["life"])
            rec["first_t"] = state["first_t"]
            rec["last_t"] = state["last_t"]
            rec["contact"] = state["contact"]
            rec["max_dist_pre"] = state["max_dist_pre"]
            rec["collision_candidate"] = state["phase"] != "idle"
            if state["phase"] != "idle":
                impact_now.append(key)
            if state["confirmed"]:
                confirmed_active.append(key)
                self.confirmed_keys.add(key)
            records[key] = rec

        self._prune(t_sec)
        impact_now.sort()
        confirmed_active.sort()
        evidence = bool(confirmed_active)
        self.temporal.update(LABEL, t_sec, evidence=evidence)
        return {"t_sec": t_sec, "evidence": evidence,
                "impact_pairs": impact_now, "confirmed_pairs": confirmed_active,
                "pairs": records, "rejected": dict(self.rejected_reasons)}

    def finalize(self) -> list[EventSegment]:
        """Return confirmed, non-overlapping accident segments."""
        return self.temporal.finalize()

    def reset(self) -> None:
        self.temporal.reset()
        self._pairs.clear()
        self.impact_pairs_count = 0
        self.rejected_reasons.clear()
        self.confirmed_keys.clear()

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
        """>= 1 heavy vehicle (car/truck/bus); the other heavy / motorcycle /
        vulnerable. Excludes person-person, bicycle-bicycle,
        motorcycle-motorcycle, person-motorcycle, ..."""
        a_heavy = label_a in self.heavy_labels
        b_heavy = label_b in self.heavy_labels
        if a_heavy and b_heavy:
            return True
        if a_heavy or b_heavy:
            other = label_b if a_heavy else label_a
            return other in self.light_labels or other in self.vulnerable_labels
        return False

    def _impact_now(self, inter: PairInteraction, st_a, st_b) -> bool:
        """Frame-level collision/impact candidate (necessary, NOT sufficient)."""
        if st_a is None or st_b is None:
            return False
        if not self._relevant(inter.class_a, inter.class_b):
            return False
        if min(st_a.quality, st_b.quality) < self.min_quality:
            return False
        if st_a.stationary and st_b.stationary:
            return False
        close_ok = inter.distance_px <= self.collision_dist
        approach_ok = (inter.approaching
                       and inter.closing_speed_px_s >= self.min_closing
                       and inter.relative_speed_px_s >= self.min_relative
                       and math.isfinite(inter.ttc_sec)
                       and inter.ttc_sec <= self.impact_ttc
                       and inter.min_predicted_distance_px <= self.collision_dist)
        return close_ok or approach_ok

    # ------------------------------------------------------------------ eval
    def _evaluate(self, inter: PairInteraction, motion_states, state: dict) -> dict:
        st_a = motion_states.get(inter.track_id_a)
        st_b = motion_states.get(inter.track_id_b)
        life = state["life"]
        life["min_ttc"] = min(life["min_ttc"],
                              inter.ttc_sec if math.isfinite(inter.ttc_sec) else INF)
        life["min_dist"] = min(life["min_dist"], inter.distance_px)
        life["min_pred"] = min(life["min_pred"], inter.min_predicted_distance_px)
        life["max_closing"] = max(life["max_closing"], inter.closing_speed_px_s)
        life["max_rel"] = max(life["max_rel"], inter.relative_speed_px_s)
        return {
            "pair": pair_key(inter.track_id_a, inter.track_id_b),
            "id_a": inter.track_id_a, "id_b": inter.track_id_b,
            "class_a": inter.class_a, "class_b": inter.class_b,
            "distance_px": inter.distance_px,
            "closing_speed_px_s": inter.closing_speed_px_s,
            "relative_speed_px_s": inter.relative_speed_px_s,
            "ttc_sec": inter.ttc_sec,
            "min_predicted_distance_px": inter.min_predicted_distance_px,
            "heading_difference_deg": inter.heading_difference_deg,
            "approaching": inter.approaching,
            "impact_ok": self._impact_now(inter, st_a, st_b),
            "phase": "idle", "confirmed": False, "collision_candidate": False,
            "signals": [], "life": None,
        }

    def _update_impact_state(self, state, inter, motion_states, t_sec, key) -> None:
        """Causal per-pair impact-phase machine (bounded history only)."""
        st_a = motion_states.get(inter.track_id_a)
        st_b = motion_states.get(inter.track_id_b)
        hit = self._impact_now(inter, st_a, st_b)

        if state["phase"] == "idle":
            if hit:
                self._enter_impact(state, inter, st_a, st_b, t_sec)
                self.impact_pairs_count += 1
            return
        # ---- impact phase ------------------------------------------------
        if inter.distance_px <= self.collision_dist:
            state["contact"] = True
        # post-impact signals (within the bounded causal window)
        self._post_signals(state, inter, st_a, st_b, t_sec)
        # confirmation: combination of independent cues, all causal
        if (not state["confirmed"] and state["contact"]
                and state["max_dist_pre"] >= self.pre_separation
                and len(state["signals"]) >= self.min_signals):
            state["confirmed"] = True
        # phase exit / persistence
        in_window = t_sec <= state["onset_t"] + self.post_window
        collocated = inter.distance_px <= self.collision_dist * COLLOCATION_FACTOR
        over_cap = t_sec - state["onset_t"] > self.max_impact_dur
        keep = (in_window or (state["confirmed"] and collocated)) and not over_cap
        if not keep:
            self._exit_impact(state, key)

    def _enter_impact(self, state, inter, st_a, st_b, t_sec) -> None:
        state["phase"] = "impact"
        state["onset_t"] = t_sec
        state["pre_speed"] = {inter.track_id_a: (st_a.speed if st_a else None),
                              inter.track_id_b: (st_b.speed if st_b else None)}
        state["pre_heading"] = {inter.track_id_a: (st_a.heading_deg if st_a else None),
                                inter.track_id_b: (st_b.heading_deg if st_b else None)}
        state["post"] = {
            "min_speed": {inter.track_id_a: INF, inter.track_id_b: INF},
            "max_decel": {inter.track_id_a: 0.0, inter.track_id_b: 0.0},
            "max_heading_change": {inter.track_id_a: 0.0, inter.track_id_b: 0.0},
            "stationary_from": {inter.track_id_a: None, inter.track_id_b: None},
            "colloc_from": None,
        }
        state["contact"] = False
        state["confirmed"] = False
        state["signals"] = set()

    def _post_signals(self, state, inter, st_a, st_b, t_sec) -> None:
        post = state["post"]
        for st, tid in ((st_a, inter.track_id_a), (st_b, inter.track_id_b)):
            if st is None:
                continue
            prev_speed = state["pre_speed"].get(tid)
            if prev_speed is not None:
                drop = prev_speed - st.speed
                state["life"]["speed_drop"][tid] = max(
                    state["life"]["speed_drop"].get(tid, 0.0), drop)
                post["min_speed"][tid] = min(post["min_speed"][tid], st.speed)
                if drop >= self.min_speed_drop:
                    state["signals"].add("speed_drop")
            if st.accel is not None and st.accel <= -self.min_decel:
                state["signals"].add("decel")
                post["max_decel"][tid] = max(post["max_decel"][tid], -st.accel)
            pre_h = state["pre_heading"].get(tid)
            if pre_h is not None and st.heading_deg is not None:
                d = abs(_wrap_delta_deg(st.heading_deg, pre_h) or 0.0)
                post["max_heading_change"][tid] = max(
                    post["max_heading_change"][tid], d)
                state["life"]["heading_change"][tid] = max(
                    state["life"]["heading_change"].get(tid, 0.0), d)
                if d >= self.min_heading_change:
                    state["signals"].add("heading_change")
            if st.stationary:
                post["stationary_from"][tid] = post["stationary_from"][tid] or t_sec
                if t_sec - post["stationary_from"][tid] >= self.post_stationary:
                    state["signals"].add("stop")
        if inter.distance_px <= self.collision_dist * COLLOCATION_FACTOR:
            post["colloc_from"] = post["colloc_from"] or t_sec
            if t_sec - post["colloc_from"] >= self.post_stationary:
                state["signals"].add("collocation")
        else:
            post["colloc_from"] = None

    def _exit_impact(self, state, key) -> None:
        if not state["confirmed"]:
            if not state["contact"]:
                reason = "no_contact"
            elif state["max_dist_pre"] < self.pre_separation:
                reason = "no_pre_separation"
            else:
                reason = "insufficient_impact_evidence"
            self.rejected_reasons[key] = reason
        state["phase"] = "idle"
        state["onset_t"] = None
        state["signals"] = set()
        state["confirmed"] = False

    def _prune(self, t_sec: float) -> None:
        if t_sec < 0.0:
            return
        stale = [k for k, s in self._pairs.items()
                 if t_sec - s["last_t"] > self.pair_expire_sec]
        for k in stale:
            del self._pairs[k]


def segments_to_events(segments: list[EventSegment]) -> list[list]:
    """EventSegments -> harness events [[start, end, label]]. (Part A glue.)"""
    return [s.to_list() for s in segments]