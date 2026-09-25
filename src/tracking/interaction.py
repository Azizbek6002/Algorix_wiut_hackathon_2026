"""Pairwise interaction + Time-To-Collision engine (PHASE 7).

For a pair of tracked objects at the CURRENT time this computes relative
motion features and a constant-velocity Time-To-Collision estimate:

  - relative position / relative velocity
  - distance, closing speed
  - TTC (only when the relative motion is actually converging)
  - predicted closest approach (time + minimum distance)

The engine is STRICTLY CAUSAL: it consumes only current positions and the
smoothed velocity vectors already produced by src/motion.MotionEngine (which
itself is causal over trajectory history). No future frames, no ground truth.
Its current call takes two MotionState objects + two current positions; all
the speed / heading / acceleration smoothing stays in MotionEngine — this
module does not duplicate it (it reuses MotionState.vx/.vy/.heading_deg).

Linear-extrapolation assumptions (documented, cheap, deterministic):
  r(t) = pos_rel + t * (v_b - v_a),  t >= 0.
The closest-approach time is the minimizer of |r(t)| (t_ca = clamp of
-project(r,v_rel)/|v_rel|^2 to [0, inf)); TTC is distance / closing speed,
with TTC = inf whenever closing_speed <= 0 (separating, parallel or zero
relative velocity). Very small current distances are handled without any
division by zero (TTC -> 0 for a closing pair, inf otherwise).

All geometry conventions match the rest of the codebase: positions are
bottom-centers in FULL-resolution pixels; velocities in px/s (as stored in
MotionState); heading is the math convention used by MotionEngine
(0 = +x image right, 90 = +y image top, [0, 360)).

Deterministic: pure function of its inputs (frozen dataclass output).
"""

from __future__ import annotations

import math

EPS = 1e-9
TINY_DISTANCE_PX = 1e-3   # below this the objects are effectively collocated
INF = float("inf")


def heading_difference_deg(a_deg: float | None, b_deg: float | None) -> float | None:
    """Signed smallest angular distance in degrees, wrap-aware (359 vs 1 -> 2).
    Returns None when either heading is unknown."""
    if a_deg is None or b_deg is None:
        return None
    d = abs(a_deg - b_deg) % 360.0
    return min(d, 360.0 - d)


class PairInteraction:
    """Immutable snapshot of one pair's interaction at a single moment."""

    __slots__ = ("track_id_a", "track_id_b", "t_sec", "pos_a", "pos_b",
                 "class_a", "class_b", "same_class", "distance_px",
                 "relative_velocity_px_s", "relative_speed_px_s",
                 "closing_speed_px_s", "ttc_sec",
                 "time_to_closest_approach_sec", "min_predicted_distance_px",
                 "heading_difference_deg", "approaching")

    def __init__(self, track_id_a: int, track_id_b: int, t_sec: float,
                 pos_a: tuple[float, float], pos_b: tuple[float, float],
                 class_a: str | None, class_b: str | None, same_class: bool,
                 distance_px: float,
                 relative_velocity_px_s: tuple[float, float],
                 relative_speed_px_s: float, closing_speed_px_s: float,
                 ttc_sec: float, time_to_closest_approach_sec: float,
                 min_predicted_distance_px: float,
                 heading_difference_deg: float | None, approaching: bool):
        self.track_id_a = track_id_a
        self.track_id_b = track_id_b
        self.t_sec = t_sec
        self.pos_a = pos_a
        self.pos_b = pos_b
        self.class_a = class_a
        self.class_b = class_b
        self.same_class = same_class
        self.distance_px = distance_px
        self.relative_velocity_px_s = relative_velocity_px_s
        self.relative_speed_px_s = relative_speed_px_s
        self.closing_speed_px_s = closing_speed_px_s
        self.ttc_sec = ttc_sec
        self.time_to_closest_approach_sec = time_to_closest_approach_sec
        self.min_predicted_distance_px = min_predicted_distance_px
        self.heading_difference_deg = heading_difference_deg
        self.approaching = approaching

    def __eq__(self, other) -> bool:
        if not isinstance(other, PairInteraction):
            return NotImplemented
        return all(getattr(self, s) == getattr(other, s)
                   for s in self.__slots__)

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        cls = type(self)
        fields = " ".join(f"{s}={getattr(self, s)!r}" for s in self.__slots__)
        return f"{cls.__name__}({fields})"


class PairwiseInteractionEngine:
    """Computes PairInteraction features from current positions + MotionStates.

    Args:
        tiny_distance_px: distance floor treated as "already collocated".
        eps: numerical threshold for zero-ish velocities / closing speeds.
    """

    def __init__(self, tiny_distance_px: float = TINY_DISTANCE_PX,
                 eps: float = EPS):
        assert tiny_distance_px >= 0.0
        self.tiny_distance_px = float(tiny_distance_px)
        self.eps = float(eps)

    def compute(self,
                track_id_a: int, track_id_b: int, t_sec: float,
                pos_a: tuple[float, float], pos_b: tuple[float, float],
                state_a=None, state_b=None,
                class_a: str | None = None, class_b: str | None = None,
                ) -> PairInteraction:
        """Full pairwise features at time t_sec.

        state_a/state_b: MotionState from src.motion.MotionEngine (None when a
        track does not yet have enough history -> treated as motion-unknown;
        positional distance is still reported).
        """
        va = _velocity(state_a)          # (vx, vy) px/s, (0,0) when unknown
        vb = _velocity(state_b)
        vxr = vb[0] - va[0]
        vyr = vb[1] - va[1]
        rel_speed = math.hypot(vxr, vyr)

        dx = pos_b[0] - pos_a[0]
        dy = pos_b[1] - pos_a[1]
        d = math.hypot(dx, dy)

        if d > self.eps:
            closing = -(dx * vxr + dy * vyr) / d
        else:
            closing = 0.0

        # ---- TTC -----------------------------------------------------------
        if d < self.tiny_distance_px:
            ttc = 0.0 if closing > self.eps else INF
        elif closing > self.eps:
            ttc = d / max(closing, self.eps)     # eps guard: never / by zero
        else:
            ttc = INF

        # ---- predicted closest approach ------------------------------------
        if rel_speed > self.eps:
            t_ca = max(0.0, -(dx * vxr + dy * vyr) / (rel_speed * rel_speed))
        else:
            t_ca = 0.0
        min_pred = math.hypot(dx + vxr * t_ca, dy + vyr * t_ca)

        # ---- headings ------------------------------------------------------
        h_a = state_a.heading_deg if state_a is not None else None
        h_b = state_b.heading_deg if state_b is not None else None
        h_diff = heading_difference_deg(h_a, h_b)

        return PairInteraction(
            track_id_a=track_id_a, track_id_b=track_id_b, t_sec=t_sec,
            pos_a=pos_a, pos_b=pos_b,
            class_a=class_a, class_b=class_b,
            same_class=(class_a is not None and class_a == class_b),
            distance_px=d,
            relative_velocity_px_s=(vxr, vyr),
            relative_speed_px_s=rel_speed,
            closing_speed_px_s=closing,
            ttc_sec=ttc,
            time_to_closest_approach_sec=t_ca,
            min_predicted_distance_px=min_pred,
            heading_difference_deg=h_diff,
            approaching=(closing > self.eps),
        )

    def pairs(self, items, t_sec: float) -> list[PairInteraction]:
        """All unique unordered active pairs.

        items: iterable of (track_id, class_label, pos, state) 4-tuples.
        Returns one PairInteraction per unordered pair (id_a < id_b),
        deterministically ordered by (id_a, id_b).
        """
        items = sorted(items, key=lambda it: it[0])
        out: list[PairInteraction] = []
        n = len(items)
        for i in range(n):
            id_a, cls_a, pos_a, st_a = items[i]
            for j in range(i + 1, n):
                id_b, cls_b, pos_b, st_b = items[j]
                out.append(self.compute(id_a, id_b, t_sec, pos_a, pos_b,
                                        st_a, st_b, cls_a, cls_b))
        return out


def _velocity(state) -> tuple[float, float]:
    """Velocity vector in px/s; motion-unknown states count as zero."""
    if state is None:
        return (0.0, 0.0)
    return (float(state.vx), float(state.vy))