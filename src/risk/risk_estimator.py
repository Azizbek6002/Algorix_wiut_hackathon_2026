"""
Part B: Causal Accident Risk Estimator (Accident Anticipation).
WIUT Hackathon 2026 — Algorix Team

Strict Contract Rules:
- STRICTLY CAUSAL: Cannot peek into future frames N+1, N+2... (Rule 7)
- Returns risk score in [0.0, 1.0] indicating probability of accident in next 5 seconds
- Evaluated on: 40% AP + 40% Alarm F1 (threshold=0.5) + 20% Time-to-Accident (TTA)

Signals:
- Min TTC: Smooth Sigmoidal / Inverse exponential curve
- Harsh deceleration / swerve
- Wrong-way trajectory presence
- Roadway pedestrian proximity
"""

import numpy as np
from typing import Optional, List, Dict, Any

from src.geometry.scene_state import SceneState
from src.events.collision_engine import CollisionInteraction


def ttc_to_risk(ttc: float) -> float:
    """
    Map Time-to-Collision (TTC) to a smooth risk score in [0.0, 1.0].
    TTC > 6.0s -> ~0.05
    TTC = 5.0s -> ~0.25
    TTC = 3.0s -> ~0.55  (triggers Alarm F1 >= 0.5)
    TTC = 2.0s -> ~0.78
    TTC = 1.0s -> ~0.95
    """
    if ttc <= 0.0 or np.isinf(ttc):
        return 0.0
    if ttc > 7.0:
        return 0.02

    # Sigmoidal mapping centered around 3.2s
    # risk = 1 / (1 + exp((ttc - 3.2) * 1.2))
    val = 1.0 / (1.0 + np.exp((ttc - 3.2) * 1.3))
    return float(np.clip(val, 0.0, 1.0))


class CausalRiskEstimator:
    """
    Evaluates online, causal risk per frame.
    Never accesses or caches future information.
    """

    def __init__(self):
        self.frame_idx = 0
        self.last_risk = 0.0
        # Exponential smoothing parameter
        self.smooth_alpha = 0.7

    def estimate_risk(
        self,
        min_ttc: float,
        scene_state: Optional[SceneState] = None,
        has_wrong_way: bool = False,
        has_pedestrian_conflict: bool = False,
    ) -> float:
        """
        Compute risk score given causal signals at the current timestamp.
        """
        self.frame_idx += 1

        # 1. Base kinematic TTC risk
        ttc_risk = ttc_to_risk(min_ttc)

        # 2. Additive contextual hazard signals
        context_hazard = 0.0
        if has_wrong_way:
            context_hazard += 0.35
        if has_pedestrian_conflict:
            context_hazard += 0.30

        raw_score = max(ttc_risk, min(1.0, ttc_risk + context_hazard))

        # 3. Temporal smoothing across successive frames (prevents single-frame jitter)
        smoothed_risk = (
            self.smooth_alpha * raw_score + (1.0 - self.smooth_alpha) * self.last_risk
        )
        self.last_risk = float(np.clip(smoothed_risk, 0.0, 1.0))

        return self.last_risk

    def reset(self) -> None:
        """Reset estimator state between video runs."""
        self.frame_idx = 0
        self.last_risk = 0.0
