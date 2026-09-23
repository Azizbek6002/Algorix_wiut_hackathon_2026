"""
Multi-Object Tracker (ByteTrack-inspired Two-Stage Association).
WIUT Hackathon 2026 — Algorix Team

Features:
- High & Low confidence detection association (ByteTrack principle)
- Trajectory history buffer with velocity and acceleration estimation
- Stopped duration tracking (crucial for stopped_vehicle and congestion)
- Occlusion resilience and ID consistency
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
import numpy as np

from src.detection.detector import Detection


def calculate_iou(boxA: Tuple[float, float, float, float], boxB: Tuple[float, float, float, float]) -> float:
    """Calculate Intersection-over-Union (IoU) between two bounding boxes (x1, y1, x2, y2)."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interWidth = max(0.0, xB - xA)
    interHeight = max(0.0, yB - yA)
    interArea = interWidth * interHeight

    boxAArea = max(0.0, boxA[2] - boxA[0]) * max(0.0, boxA[3] - boxA[1])
    boxBArea = max(0.0, boxB[2] - boxB[0]) * max(0.0, boxB[3] - boxB[1])
    unionArea = boxAArea + boxBArea - interArea

    if unionArea <= 0.0:
        return 0.0
    return interArea / unionArea


@dataclass
class Track:
    """Represents a tracked object across time frames."""
    track_id: int
    class_name: str
    bbox: Tuple[float, float, float, float]
    confidence: float
    first_timestamp: float
    last_timestamp: float
    lost_frames: int = 0
    is_active: bool = True

    # History buffers: list of (timestamp, (cx, cy))
    trajectory: List[Tuple[float, Tuple[float, float]]] = field(default_factory=list)
    # Velocity vector: (vx, vy) in pixels/sec
    velocity: Tuple[float, float] = (0.0, 0.0)
    # Stopped duration in seconds
    stopped_duration: float = 0.0

    @property
    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def bottom_center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, y2)

    @property
    def speed(self) -> float:
        """Scalar speed magnitude in pixels/sec."""
        return float(np.hypot(self.velocity[0], self.velocity[1]))

    def update(self, detection: Detection, timestamp: float) -> None:
        """Update track with new matched detection."""
        self.bbox = detection.bbox
        self.confidence = detection.confidence
        self.lost_frames = 0
        self.is_active = True

        cx, cy = self.center
        dt = timestamp - self.last_timestamp

        if dt > 1e-4 and len(self.trajectory) > 0:
            prev_time, (prev_x, prev_y) = self.trajectory[-1]
            vx = (cx - prev_x) / dt
            vy = (cy - prev_y) / dt

            # Exponential moving average smoothing for velocity
            alpha = 0.6
            self.velocity = (
                alpha * vx + (1 - alpha) * self.velocity[0],
                alpha * vy + (1 - alpha) * self.velocity[1],
            )

            # Check if vehicle is stopped (speed < 15 px/sec)
            if self.speed < 15.0:
                self.stopped_duration += dt
            else:
                self.stopped_duration = max(0.0, self.stopped_duration - dt * 0.5)

        self.last_timestamp = timestamp
        self.trajectory.append((timestamp, (cx, cy)))
        if len(self.trajectory) > 150:  # keep last 5 seconds at 30 FPS
            self.trajectory.pop(0)

    def mark_missed(self) -> None:
        """Mark track as missed in current frame."""
        self.lost_frames += 1
        if self.lost_frames > 30:  # lost for > 1 second
            self.is_active = False


class ByteTracker:
    """
    ByteTrack implementation:
    1. First match high-score detections with active tracks.
    2. Then match remaining low-score detections with unconfirmed/remaining tracks.
    """

    def __init__(
        self,
        high_score_thresh: float = 0.5,
        low_score_thresh: float = 0.15,
        match_iou_thresh: float = 0.3,
        max_lost_frames: int = 30,
    ):
        self.high_score_thresh = high_score_thresh
        self.low_score_thresh = low_score_thresh
        self.match_iou_thresh = match_iou_thresh
        self.max_lost_frames = max_lost_frames
        self.next_id = 1
        self.tracks: Dict[int, Track] = {}

    def update(self, detections: List[Detection], timestamp: float) -> List[Track]:
        """
        Update tracks using new frame detections.
        Returns list of currently active tracks.
        """
        # Split detections into high and low confidence
        high_dets = [d for d in detections if d.confidence >= self.high_score_thresh]
        low_dets = [
            d for d in detections
            if self.low_score_thresh <= d.confidence < self.high_score_thresh
        ]

        active_track_ids = [tid for tid, trk in self.tracks.items() if trk.is_active]

        # Stage 1: Match high confidence detections
        matched_tracks, unmatched_dets, unmatched_track_ids = self._match(
            active_track_ids, high_dets
        )

        for tid, det in matched_tracks:
            self.tracks[tid].update(det, timestamp)

        # Stage 2: Match remaining tracks with low confidence detections (ByteTrack advantage)
        matched_low, _, unmatched_track_ids_final = self._match(
            unmatched_track_ids, low_dets
        )
        for tid, det in matched_low:
            self.tracks[tid].update(det, timestamp)

        # Mark remaining tracks as missed
        for tid in unmatched_track_ids_final:
            self.tracks[tid].mark_missed()

        # Initialize new tracks for unmatched high confidence detections
        for det in unmatched_dets:
            new_track = Track(
                track_id=self.next_id,
                class_name=det.class_name,
                bbox=det.bbox,
                confidence=det.confidence,
                first_timestamp=timestamp,
                last_timestamp=timestamp,
                trajectory=[(timestamp, det.center)],
            )
            self.tracks[self.next_id] = new_track
            self.next_id += 1

        # Clean up stale tracks
        self.tracks = {
            tid: trk for tid, trk in self.tracks.items()
            if trk.lost_frames <= self.max_lost_frames
        }

        return [trk for trk in self.tracks.values() if trk.is_active]

    def _match(
        self, track_ids: List[int], detections: List[Detection]
    ) -> Tuple[List[Tuple[int, Detection]], List[Detection], List[int]]:
        """Greedy IoU bipartite matching."""
        if not track_ids or not detections:
            return [], detections, track_ids

        iou_matrix = np.zeros((len(track_ids), len(detections)), dtype=np.float32)
        for i, tid in enumerate(track_ids):
            for j, det in enumerate(detections):
                iou_matrix[i, j] = calculate_iou(self.tracks[tid].bbox, det.bbox)

        matched_tracks: List[Tuple[int, Detection]] = []
        used_tracks = set()
        used_dets = set()

        # Find best matches above threshold
        while True:
            max_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
            max_iou = iou_matrix[max_idx]
            if max_iou < self.match_iou_thresh:
                break

            i, j = max_idx
            matched_tracks.append((track_ids[i], detections[j]))
            used_tracks.add(i)
            used_dets.add(j)

            # Invalidate row and column
            iou_matrix[i, :] = -1.0
            iou_matrix[:, j] = -1.0

        unmatched_dets = [
            detections[j] for j in range(len(detections)) if j not in used_dets
        ]
        unmatched_track_ids = [
            track_ids[i] for i in range(len(track_ids)) if i not in used_tracks
        ]

        return matched_tracks, unmatched_dets, unmatched_track_ids
