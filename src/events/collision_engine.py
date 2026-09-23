"""
Collision & Time-to-Collision (TTC) Engine.
WIUT Hackathon 2026 — Algorix Team

Detects:
- 'accident': actual physical collision / rapid overlap followed by sudden deceleration
- 'near_miss': dangerously low TTC (evasive maneuver / harsh brake / swerve) without contact
Computes:
- Minimum scene TTC (critical input for Part B Risk Estimator)
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any
import numpy as np

from src.tracking.tracker import Track, calculate_iou
from src.events.rule_engine import RawEventSignal


@dataclass
class CollisionInteraction:
    """Pairwise interaction metrics between two tracks."""
    track_id_1: int
    track_id_2: int
    distance: float
    closing_speed: float
    ttc: float  # Time-to-Collision in seconds (infinity if diverging)
    iou: float


class CollisionEngine:
    """Computes pairwise kinematics, TTC, and classifies accidents / near misses."""

    def __init__(
        self,
        near_miss_ttc_thresh: float = 1.5,
        accident_iou_thresh: float = 0.25,
        min_speed_for_accident: float = 15.0,
    ):
        self.near_miss_ttc_thresh = near_miss_ttc_thresh
        self.accident_iou_thresh = accident_iou_thresh
        self.min_speed_for_accident = min_speed_for_accident
        # Track history of pair interactions: (id1, id2) -> list of (timestamp, distance, ttc, iou)
        self.pair_histories: Dict[Tuple[int, int], List[Tuple[float, float, float, float]]] = {}

    def compute_interactions(
        self, tracks: List[Track], timestamp: float
    ) -> List[CollisionInteraction]:
        """Compute pairwise distance, relative speed, and TTC for all track pairs."""
        interactions: List[CollisionInteraction] = []
        n = len(tracks)

        for i in range(n):
            for j in range(i + 1, n):
                t1, t2 = tracks[i], tracks[j]
                c1, c2 = t1.center, t2.center
                dx = c2[0] - c1[0]
                dy = c2[1] - c1[1]
                distance = float(np.hypot(dx, dy))

                # Relative velocity vector
                rvx = t1.velocity[0] - t2.velocity[0]
                rvy = t1.velocity[1] - t2.velocity[1]

                # Closing speed along line-of-sight
                if distance > 1e-4:
                    los_x, los_y = dx / distance, dy / distance
                    closing_speed = rvx * los_x + rvy * los_y
                else:
                    closing_speed = 0.0

                # TTC = distance / closing_speed (if closing_speed > 0)
                if closing_speed > 5.0:
                    ttc = distance / closing_speed
                else:
                    ttc = float("inf")

                iou = calculate_iou(t1.bbox, t2.bbox)

                pair_key = (min(t1.track_id, t2.track_id), max(t1.track_id, t2.track_id))
                if pair_key not in self.pair_histories:
                    self.pair_histories[pair_key] = []
                self.pair_histories[pair_key].append((timestamp, distance, ttc, iou))
                if len(self.pair_histories[pair_key]) > 90:  # ~3 seconds
                    self.pair_histories[pair_key].pop(0)

                interactions.append(
                    CollisionInteraction(
                        track_id_1=t1.track_id,
                        track_id_2=t2.track_id,
                        distance=distance,
                        closing_speed=closing_speed,
                        ttc=ttc,
                        iou=iou,
                    )
                )

        return interactions

    def detect_collision_events(
        self,
        interactions: List[CollisionInteraction],
        tracks_map: Dict[int, Track],
        timestamp: float,
    ) -> List[RawEventSignal]:
        """Detect 'accident' and 'near_miss' events from kinematics."""
        events: List[RawEventSignal] = []

        for inter in interactions:
            pair_key = (
                min(inter.track_id_1, inter.track_id_2),
                max(inter.track_id_1, inter.track_id_2),
            )
            t1 = tracks_map.get(inter.track_id_1)
            t2 = tracks_map.get(inter.track_id_2)
            if not t1 or not t2:
                continue

            # 1. ACCIDENT: Significant overlap (IoU) while at least one vehicle had notable speed
            if inter.iou >= self.accident_iou_thresh or (
                inter.distance < 40.0 and max(t1.speed, t2.speed) > self.min_speed_for_accident
            ):
                events.append(
                    RawEventSignal(
                        event_type="accident",
                        confidence=0.95,
                        track_ids=[inter.track_id_1, inter.track_id_2],
                        timestamp=timestamp,
                        details=f"Collision contact IoU={inter.iou:.2f}, dist={inter.distance:.1f}",
                    )
                )
                continue

            # 2. NEAR MISS: Low TTC (< 1.5s) without actual contact, high closing speed
            if 0.1 < inter.ttc <= self.near_miss_ttc_thresh:
                # Ensure closing speed is genuinely alarming
                if inter.closing_speed > 30.0:
                    events.append(
                        RawEventSignal(
                            event_type="near_miss",
                            confidence=0.85,
                            track_ids=[inter.track_id_1, inter.track_id_2],
                            timestamp=timestamp,
                            details=f"Near-miss evasive proximity TTC={inter.ttc:.2f}s",
                        )
                    )

        return events

    def get_min_ttc(self, interactions: List[CollisionInteraction]) -> float:
        """Returns the most critical (minimum) TTC in the current frame."""
        if not interactions:
            return float("inf")
        return min(inter.ttc for inter in interactions)
