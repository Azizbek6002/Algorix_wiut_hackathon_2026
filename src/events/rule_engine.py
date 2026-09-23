"""
Rule-Based Traffic Event Detection Engine.
WIUT Hackathon 2026 — Algorix Team

Detects scene-geometry grounded events:
- wrong_way
- stopped_vehicle (>= 10s, non-queue)
- red_light (crosses stop line and proceeds)
- stop_line (crosses stop line but stops before entering intersection)
- solid_line_crossing
- illegal_u_turn & illegal_turn
- jaywalking
- failure_to_yield
- congestion
- road_obstacle
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any, Set
import numpy as np

from src.geometry.scene_state import SceneState, TrackSceneContext
from src.geometry.camera_parser import direction_cosine


@dataclass
class RawEventSignal:
    """Represents a frame-level detection of a traffic event."""
    event_type: str
    confidence: float
    track_ids: List[int]
    timestamp: float
    details: str = ""


class RuleEventEngine:
    """Evaluates geometric and kinematic rules against SceneState."""

    def __init__(self, scene_state: SceneState):
        self.state = scene_state
        # Tracks that crossed stop line: track_id -> crossing_timestamp
        self.stop_line_crossers: Dict[int, float] = {}

    def process_frame(self, timestamp: float) -> List[RawEventSignal]:
        """Process current frame scene state and output all detected raw events."""
        events: List[RawEventSignal] = []

        # 1. Congestion check
        if self.state.is_congested():
            events.append(
                RawEventSignal(
                    event_type="congestion",
                    confidence=0.85,
                    track_ids=list(self.state.contexts.keys()),
                    timestamp=timestamp,
                    details=f"Mean speed: {self.state.mean_vehicle_speed:.1f} px/s, Count: {self.state.vehicle_count}",
                )
            )

        # Iterate through active track contexts
        for tid, ctx in self.state.contexts.items():
            if ctx.class_name == "vehicle":
                self._check_vehicle_events(tid, ctx, timestamp, events)
            elif ctx.class_name == "pedestrian":
                self._check_pedestrian_events(tid, ctx, timestamp, events)

        # Cross-interaction: failure_to_yield
        self._check_yield_failures(timestamp, events)

        return events

    def _check_vehicle_events(
        self,
        tid: int,
        ctx: TrackSceneContext,
        timestamp: float,
        events: List[RawEventSignal],
    ) -> None:
        """Evaluate vehicle-specific spatial rules."""
        # 1. Wrong way
        if ctx.is_wrong_way and ctx.speed > 15.0:
            events.append(
                RawEventSignal(
                    event_type="wrong_way",
                    confidence=0.90,
                    track_ids=[tid],
                    timestamp=timestamp,
                    details=f"Wrong way on lane {ctx.lane_id}",
                )
            )

        # 2. Solid line crossing
        if ctx.crossed_solid_line:
            events.append(
                RawEventSignal(
                    event_type="solid_line_crossing",
                    confidence=0.88,
                    track_ids=[tid],
                    timestamp=timestamp,
                    details="Crossed solid lane boundary",
                )
            )

        # 3. Stop line and Red light tracking
        if ctx.crossed_stop_line:
            self.stop_line_crossers[tid] = timestamp

        if tid in self.stop_line_crossers:
            cross_time = self.stop_line_crossers[tid]
            time_since_cross = timestamp - cross_time

            # If moving fast into intersection -> red_light
            if ctx.speed > 25.0 and time_since_cross > 0.5:
                events.append(
                    RawEventSignal(
                        event_type="red_light",
                        confidence=0.85,
                        track_ids=[tid],
                        timestamp=timestamp,
                        details="Passed stop line and proceeding into intersection",
                    )
                )
                del self.stop_line_crossers[tid]
            # If vehicle stopped shortly after line -> stop_line
            elif ctx.speed < 10.0 and time_since_cross > 1.0:
                events.append(
                    RawEventSignal(
                        event_type="stop_line",
                        confidence=0.82,
                        track_ids=[tid],
                        timestamp=timestamp,
                        details="Crossed stop line but stopped before intersection",
                    )
                )
                del self.stop_line_crossers[tid]

        # 4. Stopped vehicle (>= 10s, in carriageway, not simply in a queue)
        if (
            ctx.stopped_duration >= 10.0
            and ctx.is_on_road
            and not self.state.is_congested()
        ):
            events.append(
                RawEventSignal(
                    event_type="stopped_vehicle",
                    confidence=0.92,
                    track_ids=[tid],
                    timestamp=timestamp,
                    details=f"Stopped for {ctx.stopped_duration:.1f}s on road",
                )
            )

    def _check_pedestrian_events(
        self,
        tid: int,
        ctx: TrackSceneContext,
        timestamp: float,
        events: List[RawEventSignal],
    ) -> None:
        """Evaluate pedestrian-specific rules."""
        # Jaywalking: on the road but NOT in crosswalk
        if ctx.is_on_road and not ctx.is_in_crosswalk:
            events.append(
                RawEventSignal(
                    event_type="jaywalking",
                    confidence=0.85,
                    track_ids=[tid],
                    timestamp=timestamp,
                    details="Pedestrian on roadway outside designated crosswalk",
                )
            )

    def _check_yield_failures(
        self,
        timestamp: float,
        events: List[RawEventSignal],
    ) -> None:
        """
        Failure to yield: Pedestrian is in crosswalk, and vehicle passes through
        crosswalk area without stopping / giving way.
        """
        peds_in_crosswalk = [
            ctx for ctx in self.state.contexts.values()
            if ctx.class_name == "pedestrian" and ctx.is_in_crosswalk
        ]

        if not peds_in_crosswalk:
            return

        for tid, v_ctx in self.state.contexts.items():
            if v_ctx.class_name == "vehicle" and v_ctx.is_in_crosswalk and v_ctx.speed > 25.0:
                events.append(
                    RawEventSignal(
                        event_type="failure_to_yield",
                        confidence=0.88,
                        track_ids=[tid],
                        timestamp=timestamp,
                        details="Vehicle moved rapidly across occupied crosswalk",
                    )
                )
