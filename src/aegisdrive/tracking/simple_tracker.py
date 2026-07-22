"""Tracker par association IoU glouton — placeholder de la Phase 1.

Volontairement minimal : associe chaque détection au track existant de plus fort IoU.
La Phase 2 remplacera ce fichier par un tracker Kalman + Hongrois (type ByteTrack),
SANS changer l'interface `Tracker`. C'est tout l'intérêt du contrat.
"""
from __future__ import annotations

from ..schemas import Detection, Frame, ObjectClass, Track


class SimpleIoUTracker:
    def __init__(self, iou_threshold: float = 0.3, max_missed: int = 15):
        self._iou_threshold = iou_threshold
        self._max_missed = max_missed
        self._tracks: list[Track] = []
        self._next_id = 0

    def update(self, frame: Frame, detections: list[Detection]) -> list[Track]:
        unmatched = set(range(len(detections)))

        # Association gloutonne par IoU décroissant, classe identique.
        for track in self._tracks:
            best_iou, best_j = 0.0, -1
            for j in unmatched:
                det = detections[j]
                if det.cls is not track.cls:
                    continue
                iou = track.bbox.iou(det.bbox)
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_j >= 0 and best_iou >= self._iou_threshold:
                det = detections[best_j]
                track.bbox = det.bbox
                track.confidence = det.confidence
                track.age += 1
                track.missed = 0
                unmatched.discard(best_j)
            else:
                track.missed += 1

        # Nouvelles détections non associées -> nouveaux tracks.
        for j in unmatched:
            det = detections[j]
            self._tracks.append(Track(
                id=self._next_id, cls=det.cls, bbox=det.bbox,
                confidence=det.confidence,
            ))
            self._next_id += 1

        # Purge des tracks perdus trop longtemps.
        self._tracks = [t for t in self._tracks if t.missed <= self._max_missed]
        return [t for t in self._tracks if t.missed == 0]
