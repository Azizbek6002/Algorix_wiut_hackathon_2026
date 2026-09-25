"""Event detectors + shared evidence layer (temporal confirmation).

`EventManager` is the single detector pool for Part A; `TemporalEventEngine`
turns frame-level evidence into confirmed, non-overlapping segments.
"""

from .accident import AccidentDetector
from .failure_to_yield import FailureToYieldDetector
from .manager import EventManager
from .temporal import EventSegment, TemporalEventEngine

__all__ = ["AccidentDetector", "EventManager", "EventSegment",
           "FailureToYieldDetector", "TemporalEventEngine"]