"""Tracking / trajectory / motion / interaction state engines."""

from .trajectory import (Detection, TrackTrajectory, TrajectoryEngine,
                         TrajectoryPoint, frame_presence_stats)
from .motion import MotionEngine, MotionState
from .interaction import (PairInteraction, PairwiseInteractionEngine,
                          heading_difference_deg)
from .tracker_wrap import Track, Tracker
from .features import update_track

__all__ = [
    "Detection", "TrackTrajectory", "TrajectoryEngine", "TrajectoryPoint",
    "frame_presence_stats", "MotionEngine", "MotionState", "PairInteraction",
    "PairwiseInteractionEngine", "heading_difference_deg", "Track", "Tracker",
    "update_track",
]