"""Tracker Kalman à vitesse constante — Phase 2b (esprit SORT/ByteTrack).

Chaque track porte un filtre de Kalman sur l'état [cx, cy, w, h, vx, vy] :
  - PREDICT à chaque frame : avance la position selon la vitesse estimée.
  - UPDATE si associé à une détection : corrige l'état avec la mesure.
Si un objet est brièvement masqué, le track survit `max_missed` frames en se fiant à
la prédiction — l'ID reste stable de l'autre côté de l'occlusion.

Association : IoU glouton entre boîtes PRÉDITES et détections (même classe). Pas de
Hongrois (pas de scipy) — glouton suffit pour des scènes routières typiques et se
remplacera par un vrai assignement optimal plus tard sans changer l'interface.

Implémente l'interface `Tracker` : mêmes entrées/sorties que SimpleIoUTracker, donc
interchangeable via `--tracker`.
"""
from __future__ import annotations

import numpy as np

from ..schemas import BBox, Detection, Frame, Track


def _to_z(b: BBox) -> np.ndarray:
    cx, cy = b.center
    return np.array([cx, cy, b.width, b.height], dtype=float)


def _to_bbox(state: np.ndarray) -> BBox:
    cx, cy, w, h = state[0], state[1], max(1.0, state[2]), max(1.0, state[3])
    return BBox(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


class _KalmanBox:
    """Filtre de Kalman vitesse-constante pour une boîte."""

    def __init__(self, z0: np.ndarray):
        # État : [cx, cy, w, h, vx, vy]
        self.x = np.array([z0[0], z0[1], z0[2], z0[3], 0.0, 0.0], dtype=float)

        # Transition : cx += vx, cy += vy (dt=1 frame).
        self.F = np.eye(6)
        self.F[0, 4] = 1.0
        self.F[1, 5] = 1.0

        # Mesure : on observe cx, cy, w, h.
        self.H = np.zeros((4, 6))
        self.H[0, 0] = self.H[1, 1] = self.H[2, 2] = self.H[3, 3] = 1.0

        self.P = np.eye(6) * 10.0
        self.P[4:, 4:] *= 100.0          # forte incertitude initiale sur la vitesse
        self.Q = np.eye(6) * 1.0
        self.Q[4:, 4:] *= 0.01           # vitesse suppose lisse
        self.R = np.eye(4) * 1.0

    def predict(self) -> None:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z: np.ndarray) -> None:
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        # pinv (pseudo-inverse) au lieu de inv : ne lève jamais sur une S singulière
        # (dégénérescence numérique possible) -> la correction est simplement ignorée
        # au lieu de faire planter tout le pipeline.
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)
        K = self.P @ self.H.T @ S_inv
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P


class KalmanTracker:
    def __init__(self, iou_threshold: float = 0.3, max_missed: int = 35,
                 min_hits: int = 2):
        self._iou_threshold = iou_threshold
        self._max_missed = max_missed
        self._min_hits = min_hits          # frames avant qu'un track soit "confirmé"
        self._tracks: list[Track] = []
        self._filters: dict[int, _KalmanBox] = {}
        self._next_id = 0

    def update(self, frame: Frame, detections: list[Detection]) -> list[Track]:
        # 1) PREDICT tous les tracks existants.
        for t in self._tracks:
            kf = self._filters[t.id]
            kf.predict()
            t.bbox = _to_bbox(kf.x)

        # 2) Association gloutonne IoU (boîte prédite ↔ détection), même classe.
        unmatched = set(range(len(detections)))
        for t in self._tracks:
            best_iou, best_j = 0.0, -1
            for j in unmatched:
                d = detections[j]
                if d.cls is not t.cls:
                    continue
                iou = t.bbox.iou(d.bbox)
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_j >= 0 and best_iou >= self._iou_threshold:
                d = detections[best_j]
                self._filters[t.id].update(_to_z(d.bbox))
                t.bbox = _to_bbox(self._filters[t.id].x)
                t.confidence = d.confidence
                t.age += 1
                t.missed = 0
                unmatched.discard(best_j)
            else:
                t.missed += 1

        # 3) Nouvelles détections -> nouveaux tracks.
        for j in unmatched:
            d = detections[j]
            kf = _KalmanBox(_to_z(d.bbox))
            self._filters[self._next_id] = kf
            self._tracks.append(Track(
                id=self._next_id, cls=d.cls, bbox=d.bbox, confidence=d.confidence,
            ))
            self._next_id += 1

        # 4) Purge des tracks perdus trop longtemps.
        survivors = [t for t in self._tracks if t.missed <= self._max_missed]
        for t in self._tracks:
            if t.missed > self._max_missed:
                self._filters.pop(t.id, None)
        self._tracks = survivors

        # 5) Sortie : tracks vus cette frame et suffisamment confirmés.
        return [t for t in self._tracks
                if t.missed == 0 and t.age + 1 >= self._min_hits]
