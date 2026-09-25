"""Tracker wrapper: per-frame detections -> persistent tracks with ground-plane metres.

Primary: UCMCTrack Python (MIT, corfyi/UCMCTrack) + cam_para homography.
Fallback: ultralytics ByteTrack (pixel space, coarser TTC).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Track:
    track_id: int
    label: str
    xyxy: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    x_m: float = 0.0
    y_m: float = 0.0
    speed_mps: float = 0.0
    heading_deg: float = 0.0
    stationary: bool = False
    miss: int = 0
    history_m: list[tuple[float, float, float]] = field(default_factory=list)


class Tracker:
    def __init__(self, meta: dict | None = None, cam_para: dict | None = None):
        self.meta = meta or {}
        self.cam_para = cam_para
        self.tracks: dict[int, Track] = {}
        self._next_id = 1

    def reset(self, meta: dict) -> None:
        self.meta = meta
        self.tracks.clear()
        self._next_id = 1

    def update(self, detections: list[dict], t_sec: float) -> list[Track]:
        """Match detections to tracks, update positions/velocities.

        Skeleton: each detection becomes a new track; ground-plane maths is
        wired in once UCMCTrack/cam_para is integrated.
        """
        for tr in self.tracks.values():
            tr.miss += 1
        out = []
        for d in detections:
            tr = Track(track_id=self._next_id, label=d["label"],
                       xyxy=d["xyxy"], miss=0, history=[])
            self.tracks[tr.track_id] = tr
            out.append(tr)
            self._next_id += 1
        self.tracks = {tid: t for tid, t in self.tracks.items() if t.miss < 30}
        return out