"""Lecture vidéo via OpenCV — implémente l'interface Source."""
from __future__ import annotations

from typing import Iterator

import cv2

from ..schemas import Frame


class VideoFileSource:
    def __init__(self, path: str, proc_width: int | None = None, stride: int = 1,
                 start_s: float = 0.0, end_s: float | None = None):
        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            raise FileNotFoundError(f"Impossible d'ouvrir la vidéo : {path}")
        self._src_fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._stride = max(1, stride)
        self._fps = self._src_fps / self._stride   # fps effectif après stride
        src_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        src_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        raw_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Segment [start_s, end_s) — on ne traite qu'une portion de la vidéo. Bornes
        # exprimées en secondes, converties en index de frame source (exclusif pour la fin).
        self._start_frame = max(0, int(round(max(0.0, start_s) * self._src_fps)))
        self._end_frame = (int(round(end_s * self._src_fps))
                           if end_s is not None and end_s > 0 else None)
        if (self._end_frame is not None
                and self._end_frame <= self._start_frame):
            raise ValueError(f"--end ({end_s}s) doit être postérieur à --start ({start_s}s)")

        # Nombre de frames du SEGMENT (0 si inconnu) — pour une barre de progression juste.
        if raw_count > 0:
            seg_end = min(raw_count, self._end_frame) if self._end_frame else raw_count
            self._count = max(0, seg_end - self._start_frame) // self._stride
        else:
            self._count = 0

        # Résolution de traitement : on réduit si la source dépasse proc_width.
        # Tout le pipeline travaille alors à cette taille (cohérent de bout en bout).
        if proc_width and src_w > proc_width:
            scale = proc_width / src_w
            self._w = proc_width
            self._h = int(round(src_h * scale))
            self._resize = (self._w, self._h)
        else:
            self._w, self._h = src_w, src_h
            self._resize = None

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def frame_count(self) -> int:
        """Nombre total de frames (0 si inconnu)."""
        return self._count

    @property
    def codec(self) -> str:
        """Code FourCC du codec de la vidéo (ex. 'hvc1' = HEVC, 'avc1' = H.264)."""
        v = int(self._cap.get(cv2.CAP_PROP_FOURCC))
        return "".join(chr((v >> (8 * i)) & 0xFF) for i in range(4)).strip()

    @property
    def size(self) -> tuple[int, int]:
        return self._w, self._h

    def frames(self) -> Iterator[Frame]:
        # Saut direct au début du segment (rapide sur les gros fichiers).
        if self._start_frame > 0:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, self._start_frame)
        src_idx = self._start_frame   # index ABSOLU dans la source (timestamp correct)
        out_idx = 0                   # index des frames réellement produites
        try:
            while True:
                if self._end_frame is not None and src_idx >= self._end_frame:
                    break             # fin du segment atteinte
                ok, img = self._cap.read()
                if not ok:
                    break
                if (src_idx - self._start_frame) % self._stride == 0:
                    if self._resize is not None:
                        img = cv2.resize(img, self._resize, interpolation=cv2.INTER_AREA)
                    # timestamp basé sur le temps RÉEL (indispensable pour vitesse/TTC).
                    yield Frame(index=out_idx, timestamp=src_idx / self._src_fps, image=img)
                    out_idx += 1
                src_idx += 1
        finally:
            # Libère la capture même en cas d'exception ou d'arrêt anticipé
            # (générateur fermé par le consommateur) -> pas de fuite de ressource.
            self._cap.release()
