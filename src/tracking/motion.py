"""Motion engine (PHASE 5): smoothed speed / heading / acceleration per object.

Input: a TrackTrajectory (history of TrajectoryPoint, see src/trajectory.py).
Velocity is estimated by least-squares linear fit of x(t) and y(t) over a
sliding time window (inherent smoothing, robust to per-frame box jitter).
Heading is further smoothed per track via circular exponential moving average.
Stationary objects (window displacement below a noise floor) are reported as
stationary with speed 0 and their last heading kept stable (no noise-driven
drift).

Conventions:
- positions used for motion are bottom-centers (x, p.bottom_y) by default
  (ground-contact point, less affected by box-height changes with depth).
- heading_deg: math convention 0 = +x (image right), 90 = +y image top,
  range [0, 360).
- speeds are in px/s at FULL resolution (geometry calibration to real units is
  not applied inside this module).

Deterministic: no randomness.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .trajectory import TrackTrajectory

EPS = 1e-9


def _wrap_deg(a: float) -> float:
    """Wrap angle to [-180, 180)."""
    a = (a + 180.0) % 360.0
    if a < 0:
        a += 360.0
    return a - 180.0


def _linear_fit(ts: list[float], xs: list[float]) -> tuple[float, float, float]:
    """Least-squares slope (units/s), intercept, R^2. Returns (slope, 0, 1.0)
    when x has no variance."""
    n = len(ts)
    tm = sum(ts) / n
    xm = sum(xs) / n
    sxx = sum((t - tm) ** 2 for t in ts)
    sxy = sum((t - tm) * (x - xm) for t, x in zip(ts, xs))
    syy = sum((x - xm) ** 2 for x in xs)
    if sxx < EPS:
        return 0.0, xm, 0.0
    slope = sxy / sxx
    inter = xm - slope * tm
    if syy < EPS:
        r2 = 1.0
    else:
        r2 = max(0.0, min(1.0, sxy * sxy / (sxx * syy)))
    return slope, inter, r2


@dataclass(frozen=True)
class MotionState:
    t: float
    vx: float                  # px/s smoothed
    vy: float                  # px/s smoothed
    speed: float               # px/s (0 when stationary)
    accel: float | None        # px/s^2 (difference of smoothed speeds)
    heading_deg: float | None  # direction of motion, [0,360); last known if stationary
    stationary: bool
    quality: float             # mean R^2 of the x,y fits in the window


class MotionEngine:
    def __init__(self, window_s: float = 0.8, noise_px: float = 4.0,
                 min_points: int = 3, heading_alpha: float = 0.4,
                 accel_alpha: float = 0.5, use_bottom_center: bool = True):
        self.window_s = window_s
        self.noise_px = noise_px
        self.min_points = min_points
        self.heading_alpha = heading_alpha
        self.accel_alpha = accel_alpha
        self.use_bottom_center = use_bottom_center
        self._heading_ema: dict[int, float] = {}
        self._last_state: dict[int, MotionState] = {}
        self._accel_ema: dict[int, float] = {}

    def reset(self) -> None:
        self._heading_ema.clear()
        self._last_state.clear()
        self._accel_ema.clear()

    def update(self, tr: TrackTrajectory, t_now: float) -> MotionState | None:
        points_at = [p for p in tr.points if p.t <= t_now]
        cutoff = t_now - self.window_s
        pts = [p for p in points_at if p.t >= cutoff]
        if len(pts) < self.min_points:
            return None
        return self._compute(tr.track_id, pts)

    def _compute(self, track_id: int, pts) -> MotionState:
        ts = [p.t for p in pts]
        if self.use_bottom_center:
            xs = [(p.x, p.bottom_y) for p in pts]
        else:
            xs = [(p.x, p.y) for p in pts]
        vx, _, r2x = _linear_fit(ts, [p[0] for p in xs])
        vy, _, r2y = _linear_fit(ts, [p[1] for p in xs])
        t_now = ts[-1]

        n2 = len(pts) // 2
        xf = [p[0] for p in xs[:n2]]
        yf = [p[1] for p in xs[:n2]]
        xs_ = [p[0] for p in xs[n2:]]
        ys_ = [p[1] for p in xs[n2:]]
        disp = math.hypot(sum(xs_) / len(xs_) - sum(xf) / len(xf),
                          sum(ys_) / len(ys_) - sum(yf) / len(yf)) if n2 and (len(pts) - n2) else 0.0
        stationary = disp < self.noise_px
        quality = (r2x + r2y) / 2.0

        prev = self._last_state.get(track_id)
        raw_heading = math.degrees(math.atan2(-vy, vx)) % 360.0
        last_heading = self._heading_ema.get(track_id)

        if stationary:
            sp = 0.0
            heading = last_heading  # keep last stable heading, do not drift
        else:
            sp = math.hypot(vx, vy)
            if last_heading is None:
                heading = raw_heading
            else:
                delta = _wrap_deg(raw_heading - last_heading)
                heading = (last_heading + self.heading_alpha * delta) % 360.0
            self._heading_ema[track_id] = heading

        if prev is not None:
            dt = t_now - prev.t
            if dt > EPS:
                accel_raw = (sp - prev.speed) / dt
                last_acc = self._accel_ema.get(track_id)
                if last_acc is None or self.accel_alpha >= 1.0:
                    accel = accel_raw
                else:
                    accel = last_acc + self.accel_alpha * (accel_raw - last_acc)
                self._accel_ema[track_id] = accel
            else:
                accel = self._accel_ema.get(track_id)
        else:
            accel = None

        st = MotionState(t=t_now, vx=vx if not stationary else 0.0,
                         vy=vy if not stationary else 0.0, speed=sp,
                         accel=accel, heading_deg=heading, stationary=stationary,
                         quality=quality)
        self._last_state[track_id] = st
        return st

    def get(self, track_id: int) -> MotionState | None:
        return self._last_state.get(track_id)