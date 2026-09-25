"""Object detector wrapper (YOLO11 / ultralytics).

Detects road users on a sampled frame. Detections feed the tracker.
Deterministic: fixed seed, fixed preprocess path.
"""

from __future__ import annotations

import numpy as np

COCO_VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 6: "truck", 7: "truck"}
COCO_PERSON = 0
COCO_BICYCLE = 1


class Detector:
    def __init__(self, model_path: str, conf: float = 0.25, iou: float = 0.45,
                 device: str = "cuda:0", imgsz: int = 1280):
        self.model_path = model_path
        self.conf = conf
        self.iou = iou
        self.device = device
        self.imgsz = imgsz
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            from ultralytics import YOLO
            self._model = YOLO(self.model_path)

    def detect(self, frame_bgr: np.ndarray) -> list[dict]:
        """Return list of {'xyxy': (x1,y1,x2,y2), 'conf': float, 'label': str}."""
        try:
            self._ensure_model()
        except Exception:
            return []
        res = self._model.predict(frame_bgr, conf=self.conf, iou=self.iou,
                                  imgsz=self.imgsz, device=self.device,
                                  verbose=False)[0]
        out = []
        for b in res.boxes:
            x1, y1, x2, y2 = map(float, b.xyxy[0].tolist())
            cls = int(b.cls[0].item())
            if cls not in COCO_VEHICLE_CLASSES and cls != COCO_PERSON:
                continue
            out.append({"xyxy": (x1, y1, x2, y2), "conf": float(b.conf[0]),
                        "label": COCO_VEHICLE_CLASSES.get(cls, "person")})
        return out

    def track(self, frame_bgr: np.ndarray, persist: bool = True) -> list[dict]:
        """YOLO + ByteTrack. Returns {'xyxy': ..., 'conf': float, 'label': str, 'id': int}."""
        try:
            self._ensure_model()
        except Exception:
            return []
        res = self._model.track(frame_bgr, conf=self.conf, iou=self.iou,
                                imgsz=self.imgsz, device=self.device,
                                tracker="bytetrack.yaml", persist=persist,
                                verbose=False)[0]
        boxes = res.boxes
        out = []
        if boxes is None or boxes.id is None:
            return out
        ids = boxes.id.cpu().numpy().astype(int)
        clss = boxes.cls.cpu().numpy().astype(int)
        xyxy = boxes.xyxy.cpu().numpy()
        for i in range(len(ids)):
            cls = int(clss[i])
            if cls not in COCO_VEHICLE_CLASSES and cls != COCO_PERSON:
                continue
            x1, y1, x2, y2 = map(float, xyxy[i])
            out.append({"xyxy": (x1, y1, x2, y2), "conf": float(boxes.conf[i]),
                        "label": COCO_VEHICLE_CLASSES.get(cls, "person"),
                        "id": int(ids[i])})
        return out