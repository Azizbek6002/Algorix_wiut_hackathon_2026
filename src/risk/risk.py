"""Part B — causal risk. Returns P(accident starts within 5 s) using only
past frames. Signals: min TTC between tracks, hard braking, wrong-way /
red-light candidates, pedestrian near a vehicle. Nominally 0.0.

Caller (RiskEstimator.step) may skip internal frames and re-serve the last score.
"""

from __future__ import annotations

import numpy as np

RISK_HORIZON_SEC = 5.0
ALARM_THETA = 0.5


class RiskEstimator:
    """Matches the harness interface: reset(meta) + step(frame, t_sec) -> float."""

    def reset(self, meta: dict) -> None:
        self.meta = meta
        self._last = 0.0

    def step(self, frame: np.ndarray, t_sec: float) -> float:
        risk = 0.0
        # TODO: min TTC between tracks + braking + scene context (Day 2).
        self._last = float(np.clip(risk, 0.0, 1.0))
        return self._last