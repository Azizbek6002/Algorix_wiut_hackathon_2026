"""Shared harness-contract validator for the tests (no pytest dependency)."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from solution import CLASSES  # noqa: E402


def validate_events(events, duration: float) -> list[str]:
    """Return a list of contract violations; empty means valid."""
    problems = []
    if not isinstance(events, list):
        return ["events is not a list"]
    for ev in events:
        if not isinstance(ev, (list, tuple)) or len(ev) != 3:
            problems.append(f"malformed event {ev!r}")
            continue
        s, e, label = ev
        if label not in CLASSES:
            problems.append(f"label {label!r} not in CLASSES")
        if not (isinstance(s, (int, float)) and isinstance(e, (int, float))):
            problems.append(f"non-numeric bounds {ev!r}")
            continue
        if s < 0 or not s < e:
            problems.append(f"bounds violate 0 <= start < end: {ev!r}")
        if e > duration + 0.5:
            problems.append(f"end {e:.3f} exceeds duration {duration:.3f} + 0.5")
    by_label = {}
    for ev in events:
        by_label.setdefault(ev[2], []).append((float(ev[0]), float(ev[1])))
    for label, segs in by_label.items():
        segs.sort()
        for i in range(1, len(segs)):
            if segs[i][0] < segs[i - 1][1] - 1e-9:
                problems.append(f"overlapping same-class segments {label}: "
                                f"{segs[i - 1]} and {segs[i]}")
    return problems