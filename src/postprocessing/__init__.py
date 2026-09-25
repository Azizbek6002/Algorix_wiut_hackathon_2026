"""Post-processing: flags -> segments -> harness events, same-class merge."""

from .postprocess import clean_events, events_from_flags, segments_from_flags

__all__ = ["clean_events", "events_from_flags", "segments_from_flags"]