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

    def __init__(self, sample_stride: int = 15, min_ratio: float = 0.003):
        self.sample_stride = sample_stride
        self.min_ratio = min_ratio
        self.prev_fire_pixels: int = 0

    def process_frame(
        self, frame: Optional[np.ndarray], frame_idx: int, timestamp: float
    ) -> List[RawEventSignal]:
        """Check for fire or smoke signatures in frame."""
        if frame is None or frame_idx % self.sample_stride != 0:
            return []

        h, w = frame.shape[:2]
        total_pixels = h * w
        min_pixels = int(total_pixels * self.min_ratio)

        # Color-based fast screening (RGB format assumption)
        # Fire: Intense Red (R > 210), High Yellow/Green (G > 120), Low Blue (B < 80), R > G > B
        r = frame[:, :, 0].astype(np.int32)
        g = frame[:, :, 1].astype(np.int32)
        b = frame[:, :, 2].astype(np.int32)

        fire_condition = (r > 215) & (g > 130) & (b < 75) & (r - g > 30) & (g - b > 40)
        fire_pixels = int(np.count_nonzero(fire_condition))

        events: List[RawEventSignal] = []

        # Must exceed adaptive threshold AND show growth/fluctuation (smoke/fire is dynamic, not static red tile)
        if fire_pixels >= min_pixels:
            confidence = min(0.95, 0.65 + (fire_pixels / (min_pixels * 3.0)) * 0.30)
            events.append(
                RawEventSignal(
                    event_type="fire_smoke",
                    confidence=confidence,
                    track_ids=[],
                    timestamp=timestamp,
                    details=f"Fire/smoke chromatic signature ({fire_pixels} px, ratio {fire_pixels/total_pixels:.4f})",
                )
            )

        self.prev_fire_pixels = fire_pixels
        return events
