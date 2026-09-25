"""Camera calibration: one-frame estimate of the ground-plane homography for
UCMCTrack (writes cam_para). Run once per fixed camera."""

from __future__ import annotations


def estimate_camera_parameters():
    raise NotImplementedError("use util/estimate_cam_para.py from UCMCTrack or a wrapper")