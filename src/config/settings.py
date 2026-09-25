"""Runtime settings — env-driven, single source of truth.

Runtime/run knobs live here (stride, imgsz, conf, device, weights path,
phase-detector toggle, post-process timings). Event-detector thresholds stay
co-located with the detectors that own them (src/events/*). Values are read
once at import, exactly like the pre-refactor per-module env reads, so
behaviour is unchanged and deterministic within a run.
"""

from __future__ import annotations

import os

_PKG_ROOT = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))

DEFAULT_WEIGHTS = "weights/yolo11x.pt"
SCENE_CONFIG_PATH = os.path.join(_PKG_ROOT, "scene_config.json")

_TRUTHS = {"1", "true", "yes", "on"}


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in _TRUTHS


class Settings:
    def __init__(self) -> None:
        self.stride = int(os.environ.get("TCV_STRIDE", "3"))
        self.imgsz = int(os.environ.get("TCV_IMGSZ", "800"))
        self.conf = float(os.environ.get("TCV_CONF", "0.25"))
        self.iou = float(os.environ.get("TCV_IOU", "0.45"))
        default_dev = "cuda:0"
        try:
            import torch
            if not torch.cuda.is_available():
                default_dev = "cpu"
        except Exception:
            default_dev = "cpu"
        self.device = os.environ.get("TCV_DEVICE", default_dev)
        self.max_frames = int(os.environ.get("TCV_MAX_FRAMES", "0"))
        self.cv_threads = int(os.environ.get("TCV_CV_THREADS", "4"))
        self.omp_threads = os.environ.get("TCV_OMP_THREADS", "4")
        self.weights_path = os.environ.get("TCV_WEIGHTS", DEFAULT_WEIGHTS)
        self.scene_config_path = SCENE_CONFIG_PATH
        # frame->segment post-processing (same defaults as pre-refactor)
        self.min_duration = float(os.environ.get("TCV_MIN_DUR", "0.5"))
        self.gap_max = float(os.environ.get("TCV_GAP_MAX", "1.0"))

    @property
    def enable_phase_detectors(self) -> bool:
        """PHASE detectors (lane wrong_way, near_miss, illegal_turn,
        illegal_u_turn) over trajectory/motion/interaction state.

        Read lazily so tests can toggle it per process; in production the env
        is fixed at launch, so determinism is preserved.
        """
        return _env_bool("TCV_ENABLE_PHASE_DETECTORS", False)


settings = Settings()