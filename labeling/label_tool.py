"""
Ground Truth (GT) Labeling & Dataset Management Tool.
WIUT Hackathon 2026 — Algorix Team

Facilitates:
- Manual dev-set annotations for sample videos: [start_time, end_time, event_type]
- Strict validation against the 14 competition classes
- Saving/loading in JSON and CSV formats for evaluate.py
"""

import json
import csv
from pathlib import Path
from typing import List, Dict, Any, Optional

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


class DatasetLabelManager:
    """Manages ground-truth event labels for hackathon evaluation."""

    def __init__(self, labels_dir: Optional[Path] = None):
        if labels_dir is None:
            self.labels_dir = Path(__file__).resolve().parent / "labels"
        else:
            self.labels_dir = Path(labels_dir)
        self.labels_dir.mkdir(parents=True, exist_ok=True)

    def validate_event(self, start_time: float, end_time: float, event_type: str) -> None:
        """Validate an individual event tuple."""
        if event_type not in CLASSES:
            raise ValueError(f"Invalid event_type '{event_type}'. Must be one of {CLASSES}")
        if start_time < 0.0:
            raise ValueError(f"start_time ({start_time}) cannot be negative.")
        if end_time <= start_time:
            raise ValueError(f"end_time ({end_time}) must be strictly greater than start_time ({start_time}).")

    def create_gt(self, video_name: str, events: List[Dict[str, Any]]) -> Path:
        """
        Save ground truth events for a video.
        Events format: [{"start_time": float, "end_time": float, "event_type": str, "notes": str}]
        """
        validated_events = []
        for ev in events:
            self.validate_event(ev["start_time"], ev["end_time"], ev["event_type"])
            validated_events.append({
                "start_time": round(float(ev["start_time"]), 2),
                "end_time": round(float(ev["end_time"]), 2),
                "event_type": ev["event_type"],
                "notes": ev.get("notes", ""),
            })

        # Sort by start_time
        validated_events.sort(key=lambda x: x["start_time"])

        data = {
            "video_name": video_name,
            "total_events": len(validated_events),
            "events": validated_events,
        }

        stem = Path(video_name).stem
        json_path = self.labels_dir / f"{stem}_gt.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        # Also export CSV for quick inspection
        csv_path = self.labels_dir / f"{stem}_gt.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["start_time", "end_time", "event_type", "notes"])
            for ev in validated_events:
                writer.writerow([ev["start_time"], ev["end_time"], ev["event_type"], ev.get("notes", "")])

        return json_path

    def load_gt(self, video_name: str) -> List[List[Any]]:
        """
        Load ground truth in solution.py format: [[start_time, end_time, event_type], ...]
        """
        stem = Path(video_name).stem
        json_path = self.labels_dir / f"{stem}_gt.json"
        if not json_path.exists():
            return []

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return [[ev["start_time"], ev["end_time"], ev["event_type"]] for ev in data.get("events", [])]


if __name__ == "__main__":
    manager = DatasetLabelManager()
    # Sample dev-set entry
    sample_events = [
        {"start_time": 12.5, "end_time": 18.0, "event_type": "red_light", "notes": "Car crossed stop-line during red light"},
        {"start_time": 45.0, "end_time": 52.0, "event_type": "stopped_vehicle", "notes": "Van stopped in lane >10s"},
        {"start_time": 78.0, "end_time": 84.5, "event_type": "near_miss", "notes": "Harsh brake near crosswalk"},
    ]
    path = manager.create_gt("sample_traffic_01.mp4", sample_events)
    print(f"Sample GT created at: {path}")
    loaded = manager.load_gt("sample_traffic_01.mp4")
    print(f"Loaded {len(loaded)} events: {loaded}")
