"""Event rules: track state + scene -> per-frame class flags.

State lives in TrackState (persistent per track id). Coordinates are full-resolution
pixels until ground-plane metres arrive with UCMCTrack. Cross-class overlaps are kept.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..scene import Scene

STILL_PX_S = 2.0
STATIONARY_SEC = 10.0
CONGESTION_FRAC = 0.6
CONGESTION_MIN_VEHICLES = 3
CONGESTION_HOLD_SEC = 6.0
WRONG_WAY_DEV = 80.0


@dataclass
class TrackState:
    label: str
    last_t: float = 0.0
    x: float = 0.0
    y: float = 0.0
    speed: float = 0.0
    heading: float = 0.0
    stationary_at: float | None = None
    moving_at: float | None = None


def _heading(dx: float, dy: float) -> float:
    return math.degrees(math.atan2(dy, dx)) % 360


def _angle_dev(a: float, b: float) -> float:
    d = abs(a - b) % 360
    return min(d, 360 - d)


def _in_band(p, band) -> bool:
    if band is None:
        return False
    (x1, y1), (x2, y2) = band
    return x1 <= p[0] <= x2 and y1 <= p[1] <= y2


def _in_poly(poly, x, y) -> bool:
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1) + x1):
            inside = not inside
    return inside


def update(tracks: dict[int, TrackState], detections: list[dict], t_sec: float,
           scene: Scene) -> None:
    seen = set()
    for d in detections:
        tid = d["id"]
        seen.add(tid)
        x = (d["xyxy"][0] + d["xyxy"][2]) / 2
        y = (d["xyxy"][3] + d["xyxy"][1]) / 2
        st = tracks.get(tid)
        if st is None:
            tracks[tid] = TrackState(label=d["label"], last_t=t_sec, x=x, y=y)
            continue
        dt = max(1e-3, t_sec - st.last_t)
        dx, dy = x - st.x, y - st.y
        st.speed = (dx * dx + dy * dy) ** 0.5 / dt
        st.heading = _heading(dx, dy)
        st.last_t, st.x, st.y = t_sec, x, y
        if st.speed < STILL_PX_S and st.stationary_at is None:
            st.stationary_at = t_sec
        elif st.speed >= STILL_PX_S:
            st.stationary_at = None
        st.moving_at = t_sec
    for tid, st in list(tracks.items()):
        if tid not in seen:
            tracks.pop(tid, None)


def frame_flags(tracks: dict[int, TrackState], scene: Scene, t_sec: float) -> dict[str, bool]:
    flags = {k: False for k in (
        "wrong_way", "stopped_vehicle", "congestion", "jaywalking",
        "failure_to_yield", "solid_line_crossing", "illegal_turn",
        "illegal_u_turn", "red_light", "stop_line", "accident",
        "near_miss", "road_obstacle", "fire_smoke")}

    vehicles = [st for st in tracks.values() if st.label != "person"]
    pedestrians = [st for st in tracks.values() if st.label == "person"]

    for st in vehicles:
        if st.stationary_at is not None and (t_sec - st.stationary_at) >= STATIONARY_SEC:
            flags["stopped_vehicle"] = True
        if st.speed >= STILL_PX_S:
            dev = min(_angle_dev(st.heading, f) for f in scene.dominant_flow_deg)
            if dev >= 180 - WRONG_WAY_DEV:
                flags["wrong_way"] = True

    if len(vehicles) >= CONGESTION_MIN_VEHICLES:
        frac = sum(1 for st in vehicles if st.speed < STILL_PX_S) / len(vehicles)
        if frac >= CONGESTION_FRAC:
            _congestion_hold["on"] = True
            _congestion_hold["at"] = t_sec
    flags["congestion"] = _congestion_hold["on"] and (
        t_sec - _congestion_hold["at"]) <= CONGESTION_HOLD_SEC

    for p in pedestrians:
        if scene.road_poly and _in_poly(scene.road_poly, p.x, p.y):
            if not _in_band((p.x, p.y), scene.crosswalk_band):
                flags["jaywalking"] = True

    if scene.crosswalk_band and pedestrians:
        p_on_crossing = any(_in_band((p.x, p.y), scene.crosswalk_band) for p in pedestrians)
        if p_on_crossing and any(_in_band((v.x, v.y), scene.crosswalk_band) for v in vehicles):
            flags["failure_to_yield"] = True

    return flags


_congestion_hold = {"on": False, "at": 0.0}