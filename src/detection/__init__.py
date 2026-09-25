"""Detection: YOLO11 object-detector wrapper (detect / track)."""

from .detector import COCO_BICYCLE, COCO_PERSON, COCO_VEHICLE_CLASSES, Detector

__all__ = ["Detector", "COCO_VEHICLE_CLASSES", "COCO_PERSON", "COCO_BICYCLE"]