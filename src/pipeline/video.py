"""VideoReader: one .mp4 opened once, sampled frames yielded in order.

Sampling semantics are identical to the pre-refactor pipeline loop:
  - process frames idx where idx % stride == 0;
  - t_sec = idx / fps (seconds from the first frame);
  - max_frames > 0 stops the loop after that many frame indices are reached.
"""

from __future__ import annotations

import cv2


class VideoReader:
    def __init__(self, video_path: str):
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            self.opened = False
            self.fps = 25.0
            self.n_frames = 0
            self.duration = 0.0
            self.width = 0
            self.height = 0
            return
        self.opened = True
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.n_frames / self.fps if self.fps else 0.0
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def frames(self, stride: int, max_frames: int = 0):
        """Yield (frame_index, t_sec, frame_bgr) for sampled frames, in order."""
        idx = 0
        while True:
            ok, fr = self.cap.read()
            if not ok:
                break
            if idx % stride != 0:
                idx += 1
                continue
            if max_frames and idx >= max_frames:
                break
            yield idx, idx / self.fps, fr
            idx += 1

    def release(self) -> None:
        self.cap.release()