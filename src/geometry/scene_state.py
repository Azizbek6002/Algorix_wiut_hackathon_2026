"""
Scene State & Spatio-Temporal Spatial Aggregator.
WIUT Hackathon 2026 — Algorix Team

Maintains real-time relationship between active tracks and camera scene geometry:
- Object-to-Lane mapping
- Trajectory boundary crossings (stop-line, solid-line)
- Pedestrian crosswalk compliance
- Scene-wide traffic density and flow metrics
"""

from typing import List, Dict, Tuple, Optional, Any
from src.geometry.camera_parser import SceneGeometry, load_scene_geometry
from src.tracking.tracker import Track


class TrackSceneContext:
    """Spatio-temporal scene context for an individual track."""

    def __init__(self, track: Track, scene: SceneGeometry):
        self.track_id = track.track_id
        self.class_name = track.class_name
        self.center = track.center
        self.bottom_center = track.bottom_center
        self.speed = track.speed
        self.velocity = track.velocity
        self.stopped_duration = track.stopped_duration

        # Spatial relations
        self.is_on_road = scene.is_on_road(self.bottom_center)
        self.lane = scene.get_lane_for_point(self.bottom_center)
        self.lane_id = self.lane["id"] if self.lane else None
        self.is_in_crosswalk = scene.is_in_crosswalk(self.bottom_center)
        self.is_wrong_way = False

        if self.lane and track.speed > 10.0:
            self.is_wrong_way = scene.is_wrong_way(self.bottom_center, track.velocity)

        # Crossings history
        self.crossed_stop_line = False
        self.crossed_solid_line = False

        if len(track.trajectory) >= 2:
            prev_pt = track.trajectory[-2][1]
            curr_pt = track.trajectory[-1][1]
            self.crossed_stop_line = scene.crossed_stop_line(prev_pt, curr_pt)
            self.crossed_solid_line = scene.crossed_solid_line(prev_pt, curr_pt)


class SceneState:
    """Maintains snapshot of scene-level dynamics for the current frame."""

    def __init__(self, scene: Optional[SceneGeometry] = None):
        self.scene = scene or load_scene_geometry()
        self.contexts: Dict[int, TrackSceneContext] = {}
        self.traffic_density = 0.0
        self.mean_vehicle_speed = 0.0
        self.vehicle_count = 0
        self.pedestrian_count = 0

    def update(self, tracks: List[Track]) -> None:
        """Update scene context for all active tracks in current frame."""
        self.contexts.clear()
        vehicle_speeds: List[float] = []
        self.vehicle_count = 0
        self.pedestrian_count = 0

        for trk in tracks:
            ctx = TrackSceneContext(trk, self.scene)
            self.contexts[trk.track_id] = ctx

            if trk.class_name == "vehicle":
                self.vehicle_count += 1
                vehicle_speeds.append(trk.speed)
            elif trk.class_name == "pedestrian":
                self.pedestrian_count += 1

        self.mean_vehicle_speed = (
            float(sum(vehicle_speeds) / len(vehicle_speeds)) if vehicle_speeds else 0.0
        )
        # Normalize density: 10 vehicles in frame is considered 1.0 (dense)
        self.traffic_density = min(1.0, self.vehicle_count / 10.0)

    def is_congested(self) -> bool:
        """
        Check if scene is congested (high vehicle density AND low mean speed).
        Class: 'congestion' (12)
        """
        return self.vehicle_count >= 5 and self.mean_vehicle_speed < 15.0
