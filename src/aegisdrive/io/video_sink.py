"""Écriture : vidéo annotée (.mp4) + log JSONL de l'état monde (pour le replay)."""
from __future__ import annotations

import json
import queue
import threading
import time

import cv2
import numpy as np

from ..schemas import Frame, WorldState
from ..viz.annotator import draw_overlay
from ..viz.dashboard import render_panel


class VideoSink:
    """Écrit output.mp4 + log JSONL. Le dessin/panneau/encodage tournent dans un THREAD
    séparé (multithreading) : pendant que le CPU encode une frame, le GPU enchaîne la
    suivante -> meilleur débit sans toucher à la qualité."""

    def __init__(self, path: str, fps: float, size: tuple[int, int],
                 log_path: str | None = None, panel: bool = True,
                 panel_width: int = 360):
        self._panel = panel
        self._panel_w = panel_width
        w, h = size
        out_size = (w + panel_width, h) if panel else size
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(path, fourcc, fps, out_size)
        if not self._writer.isOpened():
            raise RuntimeError(f"Impossible d'ouvrir en écriture : {path}")
        self._h = h
        self._log = open(log_path, "w", encoding="utf-8") if log_path else None
        self._last_t = None
        self._fps_ema = None       # FPS de traitement, lissé
        # File bornée (backpressure) + thread d'écriture.
        self._q: queue.Queue = queue.Queue(maxsize=8)
        self._error: BaseException | None = None   # exception remontée du worker
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    def _run(self) -> None:
        try:
            while True:
                item = self._q.get()
                if item is None:
                    break
                image, world, fps = item
                annotated = draw_overlay(image, world)
                if self._panel:
                    panel = render_panel(world, fps, self._panel_w, self._h)
                    annotated = np.hstack([annotated, panel])
                self._writer.write(annotated)
                if self._log is not None:
                    self._log.write(json.dumps(world.to_json_dict()) + "\n")
        except BaseException as exc:   # noqa: BLE001 — on remonte l'erreur au thread principal
            # Le worker ne doit JAMAIS mourir en silence : sinon la file se remplit et
            # `consume` se bloque à jamais. On mémorise l'erreur et on vide la file.
            self._error = exc
            self._drain()

    def _drain(self) -> None:
        """Vide la file sans traiter, pour débloquer un `consume` en attente."""
        while True:
            try:
                if self._q.get_nowait() is None:
                    break
            except queue.Empty:
                break

    def _tick_fps(self) -> float | None:
        now = time.perf_counter()
        if self._last_t is not None:
            dt = now - self._last_t
            inst = 1.0 / dt if dt > 1e-6 else None
            if inst is not None:
                self._fps_ema = inst if self._fps_ema is None else 0.1 * inst + 0.9 * self._fps_ema
        self._last_t = now
        return self._fps_ema

    def consume(self, frame: Frame, world: WorldState) -> None:
        if self._error is not None:                # le worker a échoué -> on arrête net
            raise RuntimeError("Le thread d'écriture vidéo a échoué") from self._error
        fps = self._tick_fps()
        # On passe l'image (copie) et un SNAPSHOT figé de l'état au thread d'écriture :
        # les tracks sont persistants et réécrits à la frame suivante, la copie évite
        # que le worker dessine/sérialise un mélange de deux frames (course de données).
        self._q.put((frame.image.copy(), world.snapshot(), fps))

    def close(self) -> None:
        self._q.put(None)
        self._worker.join()
        self._writer.release()
        if self._log is not None:
            self._log.close()
        if self._error is not None:                # propage une erreur survenue en fin de flux
            raise RuntimeError("Le thread d'écriture vidéo a échoué") from self._error
