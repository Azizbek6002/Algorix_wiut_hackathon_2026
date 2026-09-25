"""Scene geometry API (PHASE 6).

Reads `scene_config.json` (reference-resolution coordinates) and answers spatial
queries in FULL-RESOLUTION frame coordinates. All query points are expected to
be **bottom centers** of objects, `((x1+x2)/2, y2)`.

A conversion layer scales config coordinates (stored in the reference
resolution) to the actual video resolution once per video:
  s = (frame_w / ref_w, frame_h / ref_h)
Queries take full-res (frame) pixels; internally the point is converted to the
reference space before any test. Never mix resized / model / original
coordinates outside this layer.

NOT implemented yet, by design: traffic-light state is always `UNKNOWN` (no
detector); a fake red/green classifier must not be created.
"""

from __future__ import annotations

import json

EPS = 1e-9

UNKNOWN = "UNKNOWN"


def point_in_polygon(point: tuple[float, float], poly: list) -> bool:
    """Boundary-inclusive ray-casting test."""
    x, y = point
    if not poly:
        return False
    inside = False
    n = len(poly)
    for i in range(n):
        p1 = poly[i]
        p2 = poly[(i + 1) % n]
        if _on_segment(point, p1, p2):
            return True
        x1, y1 = float(p1[0]), float(p1[1])
        x2, y2 = float(p2[0]), float(p2[1])
        if (y1 > y) != (y2 > y):
            xint = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < xint:
                inside = not inside
    return inside


def _on_segment(p, a, b) -> bool:
    px, py = float(p[0]), float(p[1])
    ax, ay = float(a[0]), float(a[1])
    bx, by = float(b[0]), float(b[1])
    if abs((bx - ax) * (py - ay) - (by - ay) * (px - ax)) > EPS:
        return False
    return (min(ax, bx) - EPS <= px <= max(ax, bx) + EPS
            and min(ay, by) - EPS <= py <= max(ay, by) + EPS)


def segments_intersect(a, b, c, d) -> bool:
    """Unions two segments inclusive of touching endpoints/overlap."""
    if _on_segment(a, c, d) or _on_segment(b, c, d):
        return True
    if _on_segment(c, a, b) or _on_segment(d, a, b):
        return True
    o1 = _orient(a, b, c)
    o2 = _orient(a, b, d)
    o3 = _orient(c, d, a)
    o4 = _orient(c, d, b)
    return o1 * o2 < 0 and o3 * o4 < 0


def _orient(p, q, r) -> float:
    return ((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]))


class Geometry:
    """Spatial queries over a scene_config.json (CAMERA-LEVEL, shared).

    All sample videos are treated as ONE fixed camera: they share the same
    resolution (3840x2160) and scene geometry. One `scene_config.json` applies
    to every video, including hidden test videos — no per-video configs.

    Args:
        cfg: parsed scene config dict.
        frame_w, frame_h: actual full-res video size. If None, the reference
            resolution from the config is used (scale = 1).
    """

    def __init__(self, cfg: dict, frame_w: int | None = None,
                 frame_h: int | None = None):
        prov = cfg.get("provenance", {})
        ref_w, ref_h = prov.get("reference_resolution", [3840, 2160])
        self.ref_w = float(ref_w)
        self.ref_h = float(ref_h)
        self.frame_w = float(frame_w) if frame_w else self.ref_w
        self.frame_h = float(frame_h) if frame_h else self.ref_h
        self.sx = self.frame_w / self.ref_w
        self.sy = self.frame_h / self.ref_h

        self.road_polygon = [tuple(p) for p in
                             _enabled_entry(cfg.get("road_polygon"), "points") or []]
        self.lanes = _enabled_list(cfg.get("lanes", []))
        self.crosswalks = [_poly(e) for e in _enabled_list(cfg.get("crosswalks", []))]
        self.intersections = [_poly(e) for e in
                              _enabled_list(cfg.get("intersection_zones", []))]
        self.u_turn_zones = [_poly(e) for e in
                             _enabled_list(cfg.get("u_turn_zones", []))]
        self.exclusion_regions = [_poly(e) for e in
                                  _enabled_list(cfg.get("exclusion_regions", []))]
        self.stop_lines = [_line(e) for e in _enabled_list(cfg.get("stop_lines", []))]
        self.solid_lines = [_line(e) for e in _enabled_list(cfg.get("solid_lines", []))]
        self.traffic_light_rois = [_line(e) for e in
                                   _enabled_list(cfg.get("traffic_light_rois", []))]

    # ---- conversion layer ----
    def to_ref(self, point) -> tuple[float, float]:
        return (float(point[0]) / self.sx, float(point[1]) / self.sy)

    # ---- public queries (full-res coordinates in, bool/id out) ----
    def get_lane(self, point) -> str | None:
        rp = self.to_ref(point)
        for lane in self.lanes:
            if point_in_polygon(rp, lane["polygon"]):
                return lane["lane_id"]
        return None

    def get_lane_direction(self, lane_id: str) -> float | None:
        for lane in self.lanes:
            if lane["lane_id"] == lane_id:
                return lane["expected_direction"]
        return None

    def is_on_road(self, point) -> bool:
        return point_in_polygon(self.to_ref(point), self.road_polygon)

    def is_in_crosswalk(self, point) -> bool:
        rp = self.to_ref(point)
        return any(point_in_polygon(rp, cw) for cw in self.crosswalks)

    def is_in_intersection(self, point) -> bool:
        rp = self.to_ref(point)
        return any(point_in_polygon(rp, z) for z in self.intersections)

    def is_in_u_turn_zone(self, point) -> bool:
        rp = self.to_ref(point)
        return any(point_in_polygon(rp, z) for z in self.u_turn_zones)

    def is_in_exclusion(self, point) -> bool:
        """Point inside a sidewalk/island/off-road region that must be treated
        as 'not trafficable' by event rules."""
        rp = self.to_ref(point)
        return any(point_in_polygon(rp, z) for z in self.exclusion_regions)

    def crosses_stop_line(self, prev_point, current_point) -> bool:
        a = self.to_ref(prev_point)
        b = self.to_ref(current_point)
        return any(segments_intersect(a, b, l[0], l[1]) for l in self.stop_lines)

    def crosses_solid_line(self, prev_point, current_point) -> bool:
        a = self.to_ref(prev_point)
        b = self.to_ref(current_point)
        return any(segments_intersect(a, b, l[0], l[1]) for l in self.solid_lines)

    def get_traffic_light_state(self, frame, roi):
        """No traffic-light detector yet -> always UNKNOWN (never fake)."""
        return UNKNOWN

    @classmethod
    def from_json(cls, path: str, frame_w: int | None = None,
                  frame_h: int | None = None) -> "Geometry":
        with open(path, "r", encoding="utf-8") as f:
            return cls(json.load(f), frame_w=frame_w, frame_h=frame_h)


def _enabled_entry(entry, key):
    if not isinstance(entry, dict) or not entry.get("enabled", True):
        return None
    return entry.get(key)


def _enabled_list(el):
    out = []
    for e in el or []:
        if isinstance(e, dict) and e.get("enabled", True):
            out.append(e)
    return out


def _poly(e) -> list[tuple[float, float]]:
    return [(float(p[0]), float(p[1])) for p in e["polygon"]]


def _line(e) -> tuple[tuple[float, float], tuple[float, float]]:
    pts = e["line"]
    return ((float(pts[0][0]), float(pts[0][1])),
            (float(pts[1][0]), float(pts[1][1])))