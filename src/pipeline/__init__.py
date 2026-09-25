"""Part A: video -> events. Orchestrator (run_pipeline) + VideoReader."""

from .pipeline import run_pipeline
from .video import VideoReader

__all__ = ["run_pipeline", "VideoReader"]