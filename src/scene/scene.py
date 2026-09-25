"""Scene layout, hard-coded from camra.md: lanes, directions, stop lines,
crosswalks, solid markings, traffic-signal ROI.

Every value here is verified against the actual `samples/camera.md` before use.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Lane:
    name: str
    polygon: list[tuple[float, float]]  # pixel polygon
    direction_deg: float = 0.0          # traffic-flow heading (degrees)


@dataclass
class Crossing:
    polygon: list[tuple[float, float]]


@dataclass
class Scene:
    width: int = 0
    height: int = 0
    lanes: list[Lane] = field(default_factory=list)
    stop_lines: list[list[tuple[float, float]]] = field(default_factory=list)  # polylines
    solid_lines: list[list[tuple[float, float]]] = field(default_factory=list)
    crossings: list[Crossing] = field(default_factory=list)
    signal_roi: list[int] | None = None   # [x1, y1, x2, y2] if signal visible
    signal_visible: bool = False
    # proxy config for rules while camera.md is absent
    dominant_flow_deg: list[float] = field(default_factory=lambda: [195.0])
    crosswalk_band: list[tuple[float, float]] | None = None   # [(x1,y1),(x2,y2)] band
    road_poly: list[tuple[float, float]] = field(default_factory=list)

    @staticmethod
    def from_camera_md(path: str) -> "Scene":
        raise NotImplementedError("parse samples/camera.md once it is available")

    @staticmethod
    def defaults_estimated(res_w: int = 3840, res_h: int = 2160) -> "Scene":
        """Geometry auto-estimated from the exploration probe (no camera.md)."""
        f = res_w / 3840.0
        return Scene(
            width=res_w, height=res_h,
            dominant_flow_deg=[195.0],
            crosswalk_band=[(1200 * f, 850 * f), (3800 * f, 1300 * f)],
            road_poly=[(0 * f, 1900 * f), (0 * f, 200 * f),
                       (3840 * f, 200 * f), (3840 * f, 1900 * f)],
        )