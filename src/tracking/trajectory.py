"""Object trajectory engine (PHASE 3/4).

Holds per-track history of detections: timestamp, center, bottom-center, bbox,
confidence. Deterministic, no randomness. Coordinates are full-resolution
pixels (the caller scales detections back to full res before entering this
module, exactly as pipeline.py already does).

Graph:  detector dict-{xyxy,conf,label,id} -> Detection -> TrajectoryEngine
The detector API is NOT changed; conversion happens at this boundary.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Detection:
    xyxy: tuple[float, float, float, float]
    conf: float
    label: str
    tid: int | None = None

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.xyxy
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def bottom_center(self) -> tuple[float, float]:
        x1, _, x2, y2 = self.xyxy
        return ((x1 + x2) / 2.0, y2)

    @property
    def width(self) -> float:
        return self.xyxy[2] - self.xyxy[0]

    @property
    def height(self) -> float:
        return self.xyxy[3] - self.xyxy[1]

    @property
    def area(self) -> float:
        return self.width * self.height

    @classmethod
    def from_dict(cls, d: dict) -> "Detection":
        return cls(xyxy=tuple(float(v) for v in d["xyxy"]),
                   conf=float(d["conf"]),
                   label=str(d["label"]),
                   tid=int(d["id"]) if d.get("id") is not None else None)


@dataclass(frozen=True)
class TrajectoryPoint:
    t: float
    x: float
    y: float          # center y
    bottom_y: float   # bottom-center y
    xyxy: tuple[float, float, float, float]
    conf: float


@dataclass
class TrackTrajectory:
    track_id: int
    label: str
    keep_sec: float = 4.0
    points: deque[TrajectoryPoint] = field(default_factory=deque)

    def append(self, p: TrajectoryPoint) -> None:
        self.points.append(p)
        cutoff = p.t - self.keep_sec
        while self.points and self.points[0].t < cutoff:
            self.points.popleft()

    def prune(self, t_now: float) -> None:
        cutoff = t_now - self.keep_sec
        while self.points and self.points[0].t < cutoff:
            self.points.popleft()

    @property
    def last(self) -> TrajectoryPoint | None:
        return self.points[-1] if self.points else None

    @property
    def age(self) -> float:
        return self.points[-1].t if self.points else -1.0

    def recent(self, window_sec: float) -> list[TrajectoryPoint]:
        if not self.points:
            return []
        cutoff = self.points[-1].t - window_sec
        return [p for p in self.points if p.t >= cutoff]


class TrajectoryEngine:
    """Maps track ids -> trajectories; prunes stale tracks by time."""

    def __init__(self, keep_sec: float = 4.0):
        self.keep_sec = keep_sec
        self.tracks: dict[int, TrackTrajectory] = {}

    def reset(self) -> None:
        self.tracks.clear()

    def update(self, detections: list[Detection], t_sec: float) -> list[TrackTrajectory]:
        for det in detections:
            if det.tid is None:
                continue
            tr = self.tracks.get(det.tid)
            if tr is None:
                tr = TrackTrajectory(track_id=det.tid, label=det.label,
                                     keep_sec=self.keep_sec)
                self.tracks[det.tid] = tr
            cx, cy = det.center
            tr.append(TrajectoryPoint(t_sec, cx, cy, det.bottom_center[1],
                                      det.xyxy, det.conf))
        stale = [tid for tid, tr in self.tracks.items() if t_sec - tr.age > self.keep_sec]
        for tid in stale:
            del self.tracks[tid]
        return sorted(self.tracks.values(), key=lambda tr: tr.track_id)

    def get(self, track_id: int) -> TrackTrajectory | None:
        return self.tracks.get(track_id)

    def active(self) -> list[TrackTrajectory]:
        return sorted(self.tracks.values(), key=lambda tr: tr.track_id)


def frame_presence_stats(id_sequences: list[list[int]]) -> dict:
    """Pure diagnostics: id churn across sampled frames (no GT needed).

    Returns counts of how many sampled frames each id appeared in, plus churn
    share (ids seen in exactly one sampled frame).
    """
    support: dict[int, int] = {}
    for ids in id_sequences:
        for i in set(ids):
            support[i] = support.get(i, 0) + 1
    frames = max(1, len(id_sequences))
    return {
        "sampled_frames": len(id_sequences),
        "unique_ids": len(support),
        "appeared_once_ids": [i for i, n in support.items() if n == 1],
        "churn_share": (sum(1 for n in support.values() if n == 1) / len(support))
                       if support else 0.0,
        "max_support_frames": max(support.values()) if support else 0,
    }