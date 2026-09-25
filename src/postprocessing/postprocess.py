"""Segment post-processing: flags -> segments; merge, blip-drop, same-class union."""

from __future__ import annotations

from collections import defaultdict


def segments_from_flags(timestamps: list[float], flags: list[bool],
                        min_dur: float = 0.5, gap_max: float = 1.0) -> list[list[float]]:
    """Consecutive True frames -> [start, end]; join gaps <= gap_max; drop < min_dur."""
    segs = []
    start = None
    end = None
    for t, f in zip(timestamps, flags):
        if f:
            if start is None:
                start = t
            end = t
        elif start is not None:
            segs.append([start, end])
            start = None
    if start is not None:
        segs.append([start, end])
    out = []
    for s, e in segs:
        if out and s - out[-1][1] <= gap_max:
            out[-1][1] = e
        else:
            out.append([s, e])
    return [[round(s, 3), round(e, 3)] for s, e in out if e - s >= min_dur]


def events_from_flags(timestamps: list[float], class_flags: dict[str, list[bool]],
                      min_dur: float = 0.5, gap_max: float = 1.0) -> list[list]:
    events = []
    for label, flags in class_flags.items():
        for s, e in segments_from_flags(timestamps, flags, min_dur, gap_max):
            events.append([s, e, label])
    events.sort()
    return events


def clean_events(events: list[list], duration: float) -> list[list]:
    """Merge same-class overlaps (union), clamp end to duration, sort."""
    by_class: dict[str, list] = defaultdict(list)
    for s, e, label in events:
        by_class[label].append([float(s), float(e)])
    out = []
    for label, segs in by_class.items():
        segs.sort()
        merged = []
        for s, e in segs:
            if merged and s <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])
        for s, e in merged:
            e = min(e, duration)
            if s < e:
                out.append([round(s, 3), round(e, 3), label])
    out.sort()
    return out