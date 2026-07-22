"""Détecteur réel basé sur Ultralytics YOLO (import paresseux).

`ultralytics` n'est PAS une dépendance dure : on l'importe seulement à la
construction, pour que le reste du projet tourne sans lui.
"""
from __future__ import annotations

from ..schemas import BBox, Detection, Frame, ObjectClass

# Mapping des classes COCO (celles pertinentes ADAS) vers nos ObjectClass.
_COCO_TO_CLASS = {
    "car": ObjectClass.CAR,
    "truck": ObjectClass.TRUCK,
    "bus": ObjectClass.BUS,
    "motorcycle": ObjectClass.MOTORCYCLE,
    "bicycle": ObjectClass.BICYCLE,
    "person": ObjectClass.PEDESTRIAN,
    "traffic light": ObjectClass.TRAFFIC_LIGHT,
    "stop sign": ObjectClass.TRAFFIC_SIGN,
    "dog": ObjectClass.ANIMAL,
    "cat": ObjectClass.ANIMAL,
    "horse": ObjectClass.ANIMAL,
}


class YoloDetector:
    def __init__(self, weights: str = "yolov8n.pt", conf: float = 0.35):
        from ultralytics import YOLO  # import paresseux volontaire
        import torch
        self._model = YOLO(weights)
        self._conf = conf
        # Utilise le GPU s'il est disponible (sinon CPU).
        self._device = 0 if torch.cuda.is_available() else "cpu"

    def detect(self, frame: Frame) -> list[Detection]:
        results = self._model.predict(frame.image, conf=self._conf, verbose=False,
                                      device=self._device)
        out: list[Detection] = []
        for r in results:
            names = r.names
            for box in r.boxes:
                label = names[int(box.cls)]
                cls = _COCO_TO_CLASS.get(label, ObjectClass.UNKNOWN)
                if cls is ObjectClass.UNKNOWN:
                    continue
                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
                out.append(Detection(BBox(x1, y1, x2, y2), cls, float(box.conf)))
        return out
