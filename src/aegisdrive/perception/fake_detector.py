"""Détecteur factice déterministe — pas de modèle, pas de GPU.

Sert la CI, les tests et le développement de l'aval (tracking/risque) sans dépendre
d'un poids YOLO. Génère un objet qui traverse l'image de gauche à droite.
"""
from __future__ import annotations

from ..schemas import BBox, Detection, Frame, ObjectClass


class FakeDetector:
    def detect(self, frame: Frame) -> list[Detection]:
        w, h = frame.shape
        # Une "voiture" qui avance horizontalement, et un "piéton" fixe.
        t = frame.index
        car_x = (t * 6) % max(1, w - 120)
        car = Detection(
            bbox=BBox(car_x, h * 0.5, car_x + 120, h * 0.5 + 80),
            cls=ObjectClass.CAR,
            confidence=0.9,
        )
        ped = Detection(
            bbox=BBox(w * 0.75, h * 0.45, w * 0.75 + 40, h * 0.45 + 100),
            cls=ObjectClass.PEDESTRIAN,
            confidence=0.8,
        )
        return [car, ped]
