"""
Algorix — Toyota Traffic Event Detection & Accident Anticipation
WIUT Hackathon 2026 — Computer Vision
Topshiriq kodi: BE86F4D5
Jamoa: Algorix

Entrypoint contract (MAJBURIY):
- CLASSES (14 classes exact names)
- detect_events(video_path: str) -> list[list]
- class RiskEstimator (causal update method)
"""

import os
from pathlib import Path
from typing import List, Any, Optional
import numpy as np

from src.utils.determinism import set_global_seed
from src.geometry.camera_parser import load_scene_geometry
from src.geometry.scene_state import SceneState
from src.detection.detector import ObjectDetector
from src.tracking.tracker import ByteTracker
from src.events.rule_engine import RuleEventEngine
from src.events.collision_engine import CollisionEngine
from src.events.smoke_fire import SmokeFireDetector
from src.events.temporal_smoothing import TemporalSmoother
from src.risk.risk_estimator import CausalRiskEstimator

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

CLASSES: List[str] = [
    "accident",            # 1. To'qnashuv / avariya
    "near_miss",           # 2. Avariyaga yaqin holat (keskin tormoz/burilish)
    "red_light",           # 3. Qizil chiroqda stop-line'dan o'tish
    "wrong_way",           # 4. Qarama-qarshi yo'nalishda yurish
    "illegal_u_turn",      # 5. Taqiqlangan U-turn
    "stopped_vehicle",     # 6. >=10s carriageway'da to'xtab qolish (navbat emas)
    "jaywalking",          # 7. Piyoda crossing'dan tashqarida yo'lga chiqishi
    "failure_to_yield",    # 8. Piyodaga yo'l bermaslik
    "illegal_turn",        # 9. Taqiqlangan/noto'g'ri lane'dan burilish
    "solid_line_crossing", # 10. Uzluksiz chiziqni bosib o'tish
    "stop_line",           # 11. Stop-line'dan o'tib, lekin intersection'ga kirmay to'xtash
    "congestion",          # 12. Barcha lane'larda tirbandlik
    "road_obstacle",       # 13. Yo'ldagi to'siq (debris, hayvon, tushgan yuk)
    "fire_smoke",          # 14. Yong'in/tutun
]


def detect_events(video_path: str) -> List[List[Any]]:
    """
    Part A (Majburiy):
    Aniqlangan trafik hodisalari ro'yxatini qaytaradi.
    Format: [[start_time (float), end_time (float), event_type (str)], ...]

    Vaqt limiti: video_duration * 3 dan oshmasligi shart (Rule 4).
    """
    set_global_seed(42)

    path = Path(video_path)
    if not path.exists():
        return []

    # Initialize pipeline components
    scene = load_scene_geometry()
    scene_state = SceneState(scene)
    detector = ObjectDetector(frame_stride=2)
    tracker = ByteTracker()
    rule_engine = RuleEventEngine(scene_state)
    collision_engine = CollisionEngine()
    smoke_fire_detector = SmokeFireDetector()
    smoother = TemporalSmoother()

    fps = 30.0
    total_duration = 0.0

    if CV2_AVAILABLE:
        cap = cv2.VideoCapture(str(path))
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            total_duration = frame_count / fps if fps > 0 else 0.0

            frame_idx = 0
            cached_detections = []

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                timestamp = frame_idx / fps

                # Object detection (with stride for <= 3x time guarantee)
                if frame_idx % detector.frame_stride == 0:
                    cached_detections = detector.detect_frame(frame)

                # Tracking update
                active_tracks = tracker.update(cached_detections, timestamp)

                # Scene geometry context update
                scene_state.update(active_tracks)

                # Frame signals
                signals = []

                # 1. Rule-based events
                signals.extend(rule_engine.process_frame(timestamp))

                # 2. Collision & TTC events
                interactions = collision_engine.compute_interactions(active_tracks, timestamp)
                signals.extend(
                    collision_engine.detect_collision_events(
                        interactions, tracker.tracks, timestamp
                    )
                )

                # 3. Smoke & Fire events
                signals.extend(smoke_fire_detector.process_frame(frame, frame_idx, timestamp))

                # Feed into temporal smoother
                smoother.feed_signals(signals, timestamp)

                frame_idx += 1

            cap.release()

    # Finalize segment intervals
    detected_events = smoother.finalize(max(total_duration, 1.0))
    return detected_events


class RiskEstimator:
    """
    Part B (Bonus): Accident Anticipation
    Faqat causal (kelajaksiz) ishlaydi (Rule 7).
    Har bir chaqiriqda faqat 'hozirgacha' ko'rilgan frame beriladi.
    """

    def __init__(self):
        set_global_seed(42)
        self.risk_core = CausalRiskEstimator()
        self.detector = ObjectDetector(frame_stride=1)
        self.tracker = ByteTracker()
        self.collision_engine = CollisionEngine()
        self.scene = load_scene_geometry()
        self.scene_state = SceneState(self.scene)
        self.meta = {}

    def reset(self, meta: Optional[dict] = None) -> None:
        """
        Reset estimator state for a new video stream (official harness interface).
        meta = {"video_id", "fps", "width", "height", "n_frames"}
        """
        self.meta = meta or {}
        self.risk_core.reset()
        self.tracker = ByteTracker()
        self.scene_state = SceneState(self.scene)

    def step(self, frame: Optional[np.ndarray], t_sec: float) -> float:
        """
        Official harness interface: Return P(accident starts within 5 s) in [0.0, 1.0].
        """
        return self.update(frame, t_sec)

    def update(self, frame: Optional[np.ndarray], timestamp: float) -> float:
        """
        Return risk score in [0.0, 1.0] for next-5-second accident probability.
        """
        if frame is None:
            return 0.0

        # Causal inference on current frame
        dets = self.detector.detect_frame(frame)
        tracks = self.tracker.update(dets, timestamp)
        self.scene_state.update(tracks)

        interactions = self.collision_engine.compute_interactions(tracks, timestamp)
        min_ttc = self.collision_engine.get_min_ttc(interactions)

        has_wrong_way = any(ctx.is_wrong_way for ctx in self.scene_state.contexts.values())
        has_ped_hazard = any(
            ctx.class_name == "pedestrian" and ctx.is_on_road and not ctx.is_in_crosswalk
            for ctx in self.scene_state.contexts.values()
        )

        risk_score = self.risk_core.estimate_risk(
            min_ttc=min_ttc,
            scene_state=self.scene_state,
            has_wrong_way=has_wrong_way,
            has_pedestrian_conflict=has_ped_hazard,
        )

        return float(risk_score)
