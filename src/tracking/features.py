"""Per-track features per frame: speed, acceleration, stationary flag,
zone membership, heading vs lane direction. Consumed by rules.py."""

from __future__ import annotations

from ..scene import Scene
from .tracker_wrap import Track


def update_track(track: Track, xy_m: tuple[float, float] | None,
                 heading_deg: float, t_sec: float) -> None:
    if xy_m is None:
        track.miss += 1
        return
    prev = track.history_m[-1] if track.history_m else None
    track.history_m.append((xy_m[0], xy_m[1], t_sec))
    if len(track.history_m) > 60:
        track.history_m.pop(0)
    if prev is not None:
        dt = max(1e-3, t_sec - prev[2])
        dx, dy = xy_m[0] - prev[0], xy_m[1] - prev[1]
        track.speed_mps = (dx * dx + dy * dy) ** 0.5 / dt
    track.heading_deg = heading_deg
    track.x_m, track.y_m = xy_m
    track.stationary = len(track.history_m) >= 2 and track.speed_mps < 0.3


def zone_names(scene: Scene, x_px: float, y_px: float) -> list[str]:
    out = []
    for lane in scene.lanes:
        if _in_poly(lane.polygon, x_px, y_px):
            out.append(lane.name)
    return out


def _in_poly(poly, x, y) -> bool:
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1) + x1):
            inside = not inside
    return inside