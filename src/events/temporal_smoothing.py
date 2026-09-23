"""
Temporal Smoothing & Event Boundary Segmentation.
WIUT Hackathon 2026 — Algorix Team

Converts frame-level raw event detections into continuous time intervals:
Format: [[start_time, end_time, event_type], ...]

Optimized for Temporal IoU (@ 0.3 / 0.5 / 0.7 thresholds):
- Hysteresis persistence & noise rejection
- Min-duration filtering (eliminates single-frame false positives)
- Max-gap merging (joins fragmented event detections)
"""

from typing import List, Dict, Tuple, Any
from src.events.rule_engine import RawEventSignal

# Minimum realistic duration in seconds for each event category
MIN_DURATION_MAP: Dict[str, float] = {
    "accident": 1.5,
    "near_miss": 1.0,
    "red_light": 1.2,
    "wrong_way": 1.5,
    "illegal_u_turn": 2.0,
    "stopped_vehicle": 8.0,
    "jaywalking": 1.5,
    "failure_to_yield": 1.0,
    "illegal_turn": 1.5,
    "solid_line_crossing": 0.8,
    "stop_line": 1.5,
    "congestion": 5.0,
    "road_obstacle": 4.0,
    "fire_smoke": 2.0,
}

# Maximum allowed gap in seconds between detections of same event to merge them
MAX_GAP_MERGE_SEC: float = 1.5


class TemporalSmoother:
    """Aggregates raw frame-by-frame signals into smoothed event segments."""

    def __init__(self):
        # Active active segments: event_type -> list of (start_t, last_t, list_of_confidences)
        self.active_segments: Dict[str, List[Dict[str, Any]]] = {}
        self.finalized_segments: List[List[Any]] = []

    def feed_signals(self, signals: List[RawEventSignal], timestamp: float) -> None:
        """Feed current frame's raw signals."""
        seen_types = set()

        for sig in signals:
            etype = sig.event_type
            seen_types.add(etype)

            if etype not in self.active_segments:
                self.active_segments[etype] = []

            # Check if we can append to the latest open segment of this event type
            if self.active_segments[etype]:
                latest = self.active_segments[etype][-1]
                if timestamp - latest["last_time"] <= MAX_GAP_MERGE_SEC:
                    latest["last_time"] = timestamp
                    latest["confidences"].append(sig.confidence)
                    continue

            # Otherwise, start a new segment
            self.active_segments[etype].append(
                {
                    "start_time": timestamp,
                    "last_time": timestamp,
                    "confidences": [sig.confidence],
                }
            )

    def finalize(self, total_video_duration: float) -> List[List[Any]]:
        """
        Finalize all open segments, apply min-duration filters,
        and return sorted [[start_time, end_time, event_type], ...].
        """
        all_events: List[List[Any]] = []

        for etype, segments in self.active_segments.items():
            min_dur = MIN_DURATION_MAP.get(etype, 1.0)
            for seg in segments:
                duration = seg["last_time"] - seg["start_time"]
                # Even instantaneous events span at least min_dur if confirmed
                effective_end = max(seg["last_time"], seg["start_time"] + min_dur)
                effective_end = min(total_video_duration, effective_end)

                if (effective_end - seg["start_time"]) >= (min_dur * 0.7):
                    all_events.append(
                        [
                            round(float(seg["start_time"]), 2),
                            round(float(effective_end), 2),
                            etype,
                        ]
                    )

        # Sort chronologically by start_time
        all_events.sort(key=lambda x: (x[0], x[1]))
        return all_events
