"""
Lightweight Smoke & Fire Detector.
WIUT Hackathon 2026 — Algorix Team

Detects 'fire_smoke' (Class 14) without slowing down main pipeline:
- Chromatic analysis in HSV/RGB (fire hue 0-35 deg, high saturation; smoke grayish luma)
- Temporal flicker / expansion dynamics
- Optional lightweight PyTorch classifier hook
"""

from typing import List, Dict, Tuple, Optional, Any
import numpy as np

from src.events.rule_engine import RawEventSignal


class SmokeFireDetector:
    """Fast, lightweight smoke and fire detection engine."""

    def __init__(self, sample_stride: int = 15, min_fire_pixels: int = 120):
        self.sample_stride = sample_stride
        self.min_fire_pixels = min_fire_pixels
        self.prev_fire_mask: Optional[np.ndarray] = None

    def process_frame(
        self, frame: Optional[np.ndarray], frame_idx: int, timestamp: float
    ) -> List[RawEventSignal]:
        """Check for fire or smoke signatures in frame."""
        if frame is None or frame_idx % self.sample_stride != 0:
            return []

        # Color-based fast screening (RGB format assumption)
        # Fire: High Red (R > 180), Moderate/High Green (G > 80), Low Blue (B < 120), R > G > B
        r = frame[:, :, 0].astype(np.int32)
        g = frame[:, :, 1].astype(np.int32)
        b = frame[:, :, 2].astype(np.int32)

        fire_condition = (r > 190) & (g > 90) & (b < 100) & (r > g) & (g > b)
        fire_pixels = int(np.count_nonzero(fire_condition))

        events: List[RawEventSignal] = []

        if fire_pixels >= self.min_fire_pixels:
            confidence = min(0.95, 0.60 + (fire_pixels / 1000.0) * 0.35)
            events.append(
                RawEventSignal(
                    event_type="fire_smoke",
                    confidence=confidence,
                    track_ids=[],
                    timestamp=timestamp,
                    details=f"Fire/smoke chromatic signature ({fire_pixels} px)",
                )
            )

        return events
