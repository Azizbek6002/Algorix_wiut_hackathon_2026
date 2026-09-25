"""Cross-cutting helpers (thread caps, io, timing). Kept deliberately small."""

from .threads import cap_cpu_threads

__all__ = ["cap_cpu_threads"]