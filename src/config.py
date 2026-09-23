"""
Configuration constants and hyperparameters for Algorix pipeline.
"""

from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = PROJECT_ROOT / "weights"
CONFIGS_DIR = PROJECT_ROOT / "camera_configs"

# Global Seed
GLOBAL_SEED = 42

# 14 Target Classes
CLASSES = [
    "accident",
    "near_miss",
    "red_light",
    "wrong_way",
    "illegal_u_turn",
    "stopped_vehicle",
    "jaywalking",
    "failure_to_yield",
    "illegal_turn",
    "solid_line_crossing",
    "stop_line",
    "congestion",
    "road_obstacle",
    "fire_smoke",
]

# Hardware limits (Rule 5: 1x GPU T4 16GB, 8 CPU, 32GB RAM)
MAX_MODEL_SIZE_BYTES = 5 * 1024 * 1024 * 1024  # 5 GB

# Video limits
MAX_TIME_FACTOR = 3.0  # Execution time <= video_duration * 3
