"""Scene: layout dataclass + calibrated geometry (scene_config.json)."""

from .scene import Crossing, Lane, Scene
from .geometry import (UNKNOWN, Geometry, point_in_polygon, segments_intersect)

__all__ = ["Scene", "Lane", "Crossing", "Geometry", "UNKNOWN",
           "point_in_polygon", "segments_intersect"]