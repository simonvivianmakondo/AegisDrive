"""Estimation du mouvement propre (ego-motion) — Lot 4, classique (flux optique).

Estime la vitesse d'avance de l'ego-véhicule à partir du défilement du bitume dans
l'image, SANS capteur véhicule. Principe (odométrie visuelle monoculaire simplifiée) :

  - On détecte des points caractéristiques UNIQUEMENT sur la zone roulable (masque de
    segmentation). Comme cette zone exclut les véhicules, ce sont des points STATIQUES
    du sol -> pas pollués par le trafic.
  - On les suit d'une frame à l'autre par flux optique (Lucas-Kanade). En avançant, un
    point du sol descend dans l'image.
  - Modèle sol plat + pinhole : un point à la ligne y (sous l'horizon y_h) est à la
    profondeur Z = f·H / (y − y_h). S'il descend de dy en dt, la vitesse ego vaut
        v ≈ dy · f · H / ((y − y_h)² · dt)
    ROBUSTESSE : le bitume lisse donne beaucoup de points SANS mouvement (flux optique
    en échec) qui tireraient la médiane vers 0, et les points près de l'horizon
    (petit (y−y_h)) amplifient le bruit (d'où d'anciennes valeurs aberrantes ~250 km/h).
    On ne garde donc que les points BIEN sous l'horizon ET réellement en mouvement, puis
    on prend la médiane trimmée de leur vitesse. L'horizon est borné à une bande plausible.

LIMITE HONNÊTE : l'échelle absolue dépend de la hauteur caméra supposée (`cam_height_m`)
et du champ de vision. La vitesse est donc une estimation ; sa cohérence relative
(accélère / ralentit / arrêté) est fiable, sa valeur exacte l'est à un facteur près.
En mode temps réel, une vraie vitesse (CAN/GPS) remplacerait ce module, même interface.
"""
from __future__ import annotations

import cv2
import numpy as np

from ..estimation.kinematics import focal_from_fov
from ..scene.lanes import corridor_horizon

_LK = dict(winSize=(21, 21), maxLevel=3,
           criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))


class EgoMotionEstimator:
    def __init__(self, fov_deg: float = 60.0, cam_height_m: float = 1.2,
                 ema: float = 0.4, max_speed_mps: float = 70.0):
        self._fov = fov_deg
        self._H = cam_height_m
        self._ema = ema
        self._max = max_speed_mps
        self._prev_gray = None
        self._prev_ts = None
        self._speed = None

    @staticmethod
    def _horizon(ctx, h: int) -> float:
        """Ligne d'horizon partagée (`ctx.horizon_y`, calculée une fois par le module
        scene) ; repli sur un calcul direct puis sur 0.45·h si indisponible."""
        y_shared = getattr(ctx, "horizon_y", 0.0) if ctx is not None else 0.0
        if y_shared and 0 < y_shared < h:
            return float(y_shared)
        if ctx is not None and getattr(ctx, "found", False):
            y = corridor_horizon(getattr(ctx, "left", None), getattr(ctx, "right", None), h)
            if y is not None:
                return y
        return 0.45 * h

    def update(self, frame, lane_ctx) -> float | None:
        gray = cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        ts = frame.timestamp

        prev_gray, prev_ts = self._prev_gray, self._prev_ts
        self._prev_gray, self._prev_ts = gray, ts
        if prev_gray is None or prev_ts is None:
            return self._speed

        dt = ts - prev_ts
        if dt <= 1e-6:
            return self._speed

        # Horizon borné : un horizon aberrant corrompt (y - y_h)² et fait exploser la
        # vitesse. On le contraint dans une bande physiquement plausible.
        y_h = float(np.clip(self._horizon(lane_ctx, h), 0.40 * h, 0.52 * h))
        f = focal_from_fov(w, self._fov)

        # Points suivis : sur la route (masque roulable), sous l'horizon.
        dm = getattr(lane_ctx, "drivable_mask", None)
        mask = np.zeros((h, w), np.uint8)
        y_start = int(min(h - 1, y_h + 0.12 * h))
        mask[y_start:h, :] = 255
        if dm is not None:
            mask &= (dm.astype(np.uint8) * 255)

        pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01,
                                      minDistance=8, mask=mask)
        if pts is None or len(pts) < 6:
            return self._speed

        nxt, stt, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts, None, **_LK)
        p0 = pts.reshape(-1, 2)
        p1 = nxt.reshape(-1, 2)
        ok = stt.reshape(-1).astype(bool)
        Y = p0[:, 1] - y_h                     # distance sous l'horizon (px)
        DY = p1[:, 1] - p0[:, 1]               # déplacement vertical (px), + = avance
        # Points BIEN sous l'horizon (grand Y = géométrie stable, pas d'amplification).
        stable = ok & (Y > 0.14 * h)
        n_stable = int(stable.sum())
        if n_stable < 8:
            return self._speed

        Yg, DYg = Y[stable], DY[stable]
        # Le bitume lisse produit beaucoup de points SANS mouvement (flux optique en échec
        # faute de texture) : le vrai signal est la minorité de points qui bougent vraiment
        # (marquages, fissures). On isole donc les points en mouvement réel (DY > 0.5 px)
        # et on prend la médiane robuste de LEUR vitesse pinhole.
        moving = DYg > 0.5
        if int(moving.sum()) < max(5, int(0.06 * n_stable)):
            v = 0.0                            # quasi aucun point ne bouge -> à l'arrêt
        else:
            vi = DYg[moving] * f * self._H / (Yg[moving] ** 2 * dt)
            vi.sort()
            t = max(0, len(vi) // 10)          # trim 10 % de chaque côté (anti-aberrants)
            vi = vi[t:len(vi) - t] if len(vi) - 2 * t >= 3 else vi
            v = float(np.clip(np.median(vi), 0.0, self._max))

        self._speed = v if self._speed is None else self._ema * v + (1 - self._ema) * self._speed
        return self._speed
