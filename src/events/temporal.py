"""Temporal Event Engine (PHASE 8).

Turns frame-level evidence from (future) event detectors into stable,
non-overlapping event segments `[start_sec, end_sec, label]` — the exact
format the harness (run_submission.py / evaluate.py) expects.

Online + strictly causal: at time t only evidence with timestamp <= t is used.
Per label there is at most ONE active run; a run:

  * starts on the first active frame (`run_start` = true event start),
  * survives absent frames as long as the gap since the last active frame
    stays within `allowed_gap` (end confirmation / min_off_duration),
  * is CONFIRMED only after `min_on_duration` of cumulative active time
    (so a single random frame never yields an event), and then becomes an
    EventSegment with start = the real first active frame,
  * closes when the gap is exceeded; the closed segment is immediately merged
    into the previous one if the gap is <= `merge_gap`.

Score-based detectors can disable near-threshold flapping with hysteresis:
inside an active run the score must drop below `threshold_off` before the run
ends, while a new run still needs `threshold_on` (threshold_on > threshold_off).

`finalize()` flushes the still-active runs (as of their last active frame),
applies a per-label safety merge, drops segments shorter than `min_duration`
and zero-length segments (the harness needs start < end), and returns the
final [start, end, label] triples, sorted by (label, start).

Deterministic: pure function of the sequence of updates. `reset()` starts a new
video.

Every parameter is globally configurable at construction and can be
overridden per label via `per_label`; every parameter can be disabled with
None meaning:

  min_on_duration=None -> confirm immediately (no start confirmation)
  allowed_gap=None     -> a run ends the moment evidence is absent
  merge_gap=None       -> no merging
  min_duration=None    -> nothing is dropped
  threshold_on=None    -> score/hysteresis path off; use `evidence`/`active`
  threshold_off=None   -> same as threshold_on (plain threshold, no hysteresis)

API:
    engine = TemporalEventEngine(min_on_duration=0.4, allowed_gap=0.5,
                                 merge_gap=1.0, min_duration=0.0,
                                 threshold_on=None, threshold_off=None,
                                 per_label=None)
    engine.update(label="near_miss", t_sec=12.3, evidence=True)    # bool evidence
    engine.update(label="accident",  t_sec=12.5, score=0.93)       # score (+hysteresis)
    engine.update(label="accident",  t_sec=12.6, active=False)     # explicit absence
    engine.finalize()  # -> [EventSegment(label, start, end), ...] == [start, end, label]

`update` timestamps must be non-decreasing per label (causality guard raises
ValueError otherwise). Equal timestamps (batched updates) are allowed.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EventSegment:
    """One temporal event segment. Unpackable as (start, end, label)."""

    label: str
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start

    def __iter__(self):
        yield self.start
        yield self.end
        yield self.label

    def to_list(self) -> list:
        return [round(self.start, 3), round(self.end, 3), self.label]


_GLOBAL_PARAMS = (
    "min_on_duration", "allowed_gap", "merge_gap",
    "min_duration", "threshold_on", "threshold_off",
)


class _LabelState:
    __slots__ = ("active", "run_start", "last_active", "last_t",
                 "active_total", "confirmed", "closed")

    def __init__(self):
        self.active = False
        self.run_start = 0.0
        self.last_active = 0.0
        self.last_t = -1.0
        self.active_total = 0.0
        self.confirmed = False
        self.closed: list[EventSegment] = []


class TemporalEventEngine:
    """Configurable, causal, deterministic evidence -> event-segments layer."""

    def __init__(self, min_on_duration: float = 0.4, allowed_gap: float = 0.5,
                 merge_gap: float = 1.0, min_duration: float = 0.0,
                 threshold_on: float | None = None,
                 threshold_off: float | None = None,
                 per_label: dict | None = None):
        self._global = {
            "min_on_duration": min_on_duration,
            "allowed_gap": allowed_gap,
            "merge_gap": merge_gap,
            "min_duration": min_duration,
            "threshold_on": threshold_on,
            "threshold_off": threshold_off,
        }
        self.per_label = dict(per_label or {})
        self._states: dict[str, _LabelState] = {}
        self._labels: list[str] = []

    # ------------------------------------------------------------------ util
    def _p(self, label: str, name: str):
        """Resolve a parameter for `label` (per-label override wins)."""
        over = self.per_label.get(label)
        if over is not None and name in over:
            return over[name]
        return self._global[name]

    def _eff(self, label: str, name: str, default: float) -> float:
        """Resolved parameter with None (=disabled) mapped to `default` value."""
        v = self._p(label, name)
        return default if v is None else float(v)

    def _state(self, label: str) -> _LabelState:
        st = self._states.get(label)
        if st is None:
            st = _LabelState()
            self._states[label] = st
            self._labels.append(label)
        return st

    # ------------------------------------------------------------------ main
    def reset(self) -> None:
        """Clear all per-label state so the engine can start a new video."""
        self._states.clear()
        self._labels.clear()

    def update(self, label: str, t_sec: float, evidence: bool | None = None,
               score: float | None = None, active: bool | None = None) -> None:
        """Ingest one frame-level observation for `label` at time t_sec.

        Provide exactly one topic:
          * `score`     -> active iff score >= threshold (hysteresis-aware);
          * `evidence`  -> plain boolean evidence;
          * `active`    -> explicit boolean;
          * none        -> treated as "no evidence at this time" (absence).
        """
        st = self._state(label)
        if t_sec < st.last_t - 1e-9:
            raise ValueError(
                f"non-monotonic time for {label!r}: {t_sec} < {st.last_t}")
        st.last_t = max(st.last_t, t_sec)

        if score is not None:
            on = self._p(label, "threshold_on")
            if on is None:
                ev = False          # score path disabled -> treat as absent
            else:
                off = self._p(label, "threshold_off")
                if off is None:
                    off = on
                thr = off if st.active else on          # hysteresis
                ev = float(score) >= float(thr)
        elif evidence is not None:
            ev = bool(evidence)
        elif active is not None:
            ev = bool(active)
        else:
            ev = False

        if ev:
            if not st.active:                       # new run starts
                st.active = True
                st.run_start = t_sec
                st.last_active = t_sec
                st.active_total = 0.0
                st.confirmed = False
            else:                                   # run continues
                st.active_total += max(0.0, t_sec - st.last_active)
                st.last_active = t_sec
            if not st.confirmed:
                on_dur = self._eff(label, "min_on_duration", 0.0)
                if on_dur <= 0.0 or st.active_total >= on_dur:
                    st.confirmed = True             # start = run_start (real start)
        else:
            gap = t_sec - st.last_active
            allowed = self._eff(label, "allowed_gap", 0.0)
            if st.active and gap >= allowed:
                self._close(label, st)

    # ------------------------------------------------------------------ close
    def _close(self, label: str, st: _LabelState) -> None:
        st.active = False
        if st.confirmed:
            seg = EventSegment(label, st.run_start, st.last_active)
            merge = self._p(label, "merge_gap")
            if merge is not None and st.closed:
                prev = st.closed[-1]
                if seg.start - prev.end <= float(merge):
                    st.closed[-1] = EventSegment(
                        label, prev.start, max(prev.end, seg.end))
                    st.run_start = st.last_active = 0.0
                    st.active_total = 0.0
                    st.confirmed = False
                    return
            st.closed.append(seg)
        st.run_start = st.last_active = 0.0
        st.active_total = 0.0
        st.confirmed = False

    # ------------------------------------------------------------------ result
    def finalize(self) -> list[EventSegment]:
        """Close still-active runs and return the final segments.

        Idempotent: calling finalize() twice yields the same list. The flush
        uses only evidence already seen (a run ends at its last active frame),
        so no future information leaks in.
        """
        for label in list(self._labels):
            st = self._states[label]
            if st.active:
                self._close(label, st)

        labels = sorted(self._labels)
        out: list[EventSegment] = []
        for label in labels:
            st = self._states[label]
            segs = sorted(st.closed, key=lambda s: (s.start, s.end))
            merge = self._p(label, "merge_gap")
            if merge is not None:
                merged: list[EventSegment] = []
                for s in segs:
                    if merged and s.start - merged[-1].end <= float(merge):
                        merged[-1] = EventSegment(label, merged[-1].start,
                                                  max(merged[-1].end, s.end))
                    else:
                        merged.append(s)
                segs = merged
            min_dur = self._eff(label, "min_duration", 0.0)
            if min_dur > 0.0:
                segs = [s for s in segs if s.duration >= min_dur]
            # the harness requires start < end; zero-length segments cannot be
            # scored and are dropped here so the output is always valid
            segs = [s for s in segs if s.end > s.start]
            out.extend(segs)
        out.sort(key=lambda s: (s.label, s.start))
        return out