"""CPU-heat control for local inference (cv2 decode threads + OMP/MKL)."""

from __future__ import annotations

import os


def cap_cpu_threads(cv_threads: int, omp_threads: str) -> None:
    """Cap OpenCV decode threads and OMP/MKL worker threads.

    Local dev only: a hot laptop throttles inference. On the GPU grading box
    CPU is idle-ish and these caps are harmless.
    """
    os.environ.setdefault("OMP_NUM_THREADS", omp_threads)
    os.environ.setdefault("MKL_NUM_THREADS", omp_threads)
    try:
        import cv2
        cv2.setNumThreads(int(cv_threads))
    except Exception:
        pass