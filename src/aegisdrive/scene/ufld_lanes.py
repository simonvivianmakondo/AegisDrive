"""Détection de voies par réseau dédié — Ultra-Fast-Lane-Detection v2 (ONNX).

Remplace la détection de lignes classique (fenêtres glissantes sur le masque YOLOPv2)
par un modèle entraîné sur CULane (dizaines de milliers d'images de route labellisées).
Il prédit directement jusqu'à 4 lignes de voie par classification ordinale sur ancres.

Produit une `list[LaneLine]` dans les coordonnées de la frame -> se branche sur le
MÊME tracker + corridor + affichage que le reste (aucune autre modification).

Détail : la dashcam ayant beaucoup de ciel, on RECADRE sur la zone route avant
inférence (le modèle CULane attend une image cadrée route), puis on remappe les points.
GPU via onnxruntime en réutilisant les DLL CUDA de torch.
"""
from __future__ import annotations

import os

import cv2
import numpy as np

from .lanes import LaneLine

_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)
_IN_W, _IN_H = 1600, 320


def _softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


class UFLDLaneDetector:
    def __init__(self, model_path: str = "models/ufldv2_culane_res34_320x1600.onnx",
                 crop_frac: float = 0.33, local_width: int = 1):
        try:
            import torch
            os.add_dll_directory(os.path.join(os.path.dirname(torch.__file__), "lib"))
        except Exception:
            pass
        import onnxruntime as ort
        self._sess = ort.InferenceSession(
            model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        self._on_gpu = self._sess.get_providers()[0] == "CUDAExecutionProvider"
        self._crop = crop_frac
        self._lw = local_width
        self._row_anchor = np.linspace(0.42, 1.0, 72)   # ancres CULane

    @property
    def on_gpu(self) -> bool:
        return self._on_gpu

    def detect(self, frame) -> list:
        img = frame.image
        h, w = img.shape[:2]
        y0 = int(self._crop * h)
        crop = img[y0:, :]
        ch = h - y0

        blob = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), (_IN_W, _IN_H))
        blob = ((blob.astype(np.float32) / 255.0 - _MEAN) / _STD).transpose(2, 0, 1)[None]
        loc_row, loc_col, exist_row, exist_col = self._sess.run(None, {"input": blob})

        n_grid, n_cls = loc_row.shape[1], loc_row.shape[2]
        max_i = loc_row.argmax(1)          # [1, n_cls, 4]
        valid = exist_row.argmax(1)        # [1, n_cls, 4]

        lines = []
        for lane in range(4):
            if valid[0, :, lane].sum() <= n_cls / 2:
                continue
            xs, ys = [], []
            for k in range(n_cls):
                if not valid[0, k, lane]:
                    continue
                c = max_i[0, k, lane]
                ind = np.arange(max(0, c - self._lw), min(n_grid, c + self._lw + 1))
                loc = (_softmax(loc_row[0, ind, k, lane]) * ind).sum() + 0.5
                xs.append(loc / (n_grid - 1) * w)
                ys.append(y0 + self._row_anchor[k] * ch)
            if len(xs) >= 6:
                line = LaneLine.fit(ys, xs, degree=2, y_scale=h)
                if line is not None and (line.y_hi - line.y_lo) >= 0.08 * h:
                    lines.append(line)
        lines.sort(key=lambda l: l.x_at(h - 1))
        return lines
