"""EventManager — the single Part-A detector pool (per video).

Owns the shared per-frame state (scene, track dict) and feeds every registered
detector the SAME prepared data (detections, timestamp), exactly as the target
architecture requires. The default pool is the legacy flag engine
(src/events/rules.py): its emitted segments are BYTE-IDENTICAL to the
pre-refactor pipeline. PHASE detectors (lane-based wrong_way, near_miss,
illegal_turn, illegal_u_turn) are registered only when
TCV_ENABLE_PHASE_DETECTORS=1 and run over trajectory/motion/interaction state
built from the very same detections — no duplicated pipeline, no duplicated
detection pass.

Usage (strictly causal, per video, in order):
    mgr = EventManager(settings, scene, width=W, height=H)
    mgr.step(dets, t_sec)            # every sampled frame
    events = mgr.finalize(duration)  # [[start, end, label], ...]
"""

from __future__ import annotations

from ..config.settings import Settings, settings as _default_settings
from ..postprocessing import clean_events, events_from_flags
from ..scene import Scene
from ..scene.geometry import Geometry
from ..tracking.motion import MotionEngine
from ..tracking.trajectory import Detection, TrajectoryEngine
from . import rules
from .illegal_turn import IllegalTurnDetector
from .illegal_u_turn import IllegalUTurnDetector
from .near_miss import NearMissDetector
from .wrong_way import WrongWayDetector

_ALL_LABELS = (
    "wrong_way", "stopped_vehicle", "congestion", "jaywalking",
    "failure_to_yield", "solid_line_crossing", "illegal_turn",
    "illegal_u_turn", "red_light", "stop_line", "accident",
    "near_miss", "road_obstacle", "fire_smoke")


class EventManager:
    def __init__(self, scene: Scene, settings: Settings | None = None,
                 width: int | None = None, height: int | None = None):
        self.settings = settings if settings is not None else _default_settings
        self.scene = scene
        # per-video reset of the legacy flag engine (same as pre-refactor)
        rules._congestion_hold["on"] = False
        rules._congestion_hold["at"] = 0.0
        self.tracks: dict[int, rules.TrackState] = {}
        self.timestamps: list[float] = []
        self.flags_map: dict[str, list[bool]] = {k: [] for k in _ALL_LABELS}

        self._geometry: Geometry | None = None
        self._phase: list = []
        if self.settings.enable_phase_detectors:
            self._geometry = Geometry.from_json(
                self.settings.scene_config_path,
                frame_w=width, frame_h=height)
            self._trajectory = TrajectoryEngine()
            self._motion = MotionEngine()
            self._phase = [
                WrongWayDetector(),
                NearMissDetector(),
                IllegalTurnDetector(),
                IllegalUTurnDetector(),
            ]

    def step(self, detections: list[dict], t_sec: float) -> None:
        """Ingest one sampled frame (detections in FULL-RES pixels)."""
        self.timestamps.append(t_sec)

        rules.update(self.tracks, detections, t_sec, self.scene)
        flags = rules.frame_flags(self.tracks, self.scene, t_sec)
        for k, v in flags.items():
            self.flags_map[k].append(v)

        if self._phase:
            trajs = self._trajectory.update(
                [Detection.from_dict(d) for d in detections], t_sec)
            motion_states: dict[int, object] = {}
            for tr in trajs:
                st = self._motion.update(tr, t_sec) if tr.last is not None else None
                if st is not None:
                    motion_states[tr.track_id] = st
            for det in self._phase:
                det.update(self._trajectory.tracks, motion_states,
                           self._geometry, t_sec)

    def finalize(self, duration: float) -> list[list]:
        """Flags -> segments (legacy, unchanged) + PHASE detector segments."""
        events = events_from_flags(
            self.timestamps, self.flags_map,
            min_dur=self.settings.min_duration,
            gap_max=self.settings.gap_max)
        extra: list[list] = []
        for det in self._phase:
            extra.extend(seg.to_list() for seg in det.finalize())
        return clean_events(events + extra, duration)