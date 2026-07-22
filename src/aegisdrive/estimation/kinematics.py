"""Estimation cinématique monoculaire — Phase 2.

Remplit distance / vitesse / accélération / TTC de chaque track.

MÉTHODE (assumée, car la distance monoculaire est mal posée) :
  Distance par a priori de taille. On connaît la largeur réelle typique d'un objet
  et la focale caméra (px). Modèle pinhole :  distance ≈ f * W_reel / w_pixels.

  Vitesse de rapprochement = -d(distance)/dt, lissée par EMA.
  Accélération = d(vitesse)/dt, lissée.
  TTC = distance / vitesse_de_rapprochement   (si l'objet se rapproche).

LIMITES (à documenter, honnête pour un portfolio) :
  - suppose que l'objet est vu de face/arrière (largeur ~ largeur réelle) ;
  - la focale par défaut est estimée depuis un champ de vision horizontal ~60° ;
  - pas de compensation du mouvement propre (ego-motion) : la vitesse est *relative*.
  La Phase 2b (Kalman) et une vraie calibration lèveront ces limites, même interface.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..schemas import ObjectClass, WorldState

# Largeur réelle typique par catégorie, en mètres.
_REAL_WIDTH_M = {
    ObjectClass.CAR: 1.8,
    ObjectClass.TRUCK: 2.5,
    ObjectClass.BUS: 2.6,
    ObjectClass.MOTORCYCLE: 0.8,
    ObjectClass.BICYCLE: 0.6,
    ObjectClass.PEDESTRIAN: 0.5,
    ObjectClass.ANIMAL: 0.6,
    ObjectClass.OBSTACLE: 1.0,
}


def focal_from_fov(image_width_px: int, fov_deg: float = 60.0) -> float:
    """Focale en pixels à partir du champ de vision horizontal (approx)."""
    return (image_width_px / 2.0) / math.tan(math.radians(fov_deg) / 2.0)


@dataclass
class _State:
    """Historique lissé par track (persiste entre les frames)."""
    last_t: float
    distance: float
    speed: float = 0.0        # vitesse de rapprochement (m/s), + = se rapproche
    accel: float = 0.0


class KinematicsEstimator:
    def __init__(self, image_width_px: int, image_height_px: int = 540,
                 focal_px: float | None = None, fov_deg: float = 60.0,
                 cam_height_m: float = 1.3, ema: float = 0.5):
        self._f = focal_px if focal_px is not None else focal_from_fov(image_width_px, fov_deg)
        self._h = image_height_px
        self._H = cam_height_m
        self._ema = ema
        self._states: dict[int, _State] = {}
        self._horizon_mem = None   # horizon APPRIS en session (EMA), utilisé si corridor absent

    def _horizon(self, world: WorldState) -> float:
        """Ligne d'horizon = point de fuite du corridor de voie, sinon défaut.

        L'horizon est calculé UNE seule fois en amont (module scene) et exposé via
        `ctx.horizon_y` : on le réutilise ici au lieu de refaire le même balayage.
        """
        ctx = world.lane_ctx
        best_y = getattr(ctx, "horizon_y", 0.0) if ctx is not None else 0.0
        if best_y and 0 < best_y < self._h:
            # Mémorise l'horizon observé -> distances stables même sans corridor.
            self._horizon_mem = (best_y if self._horizon_mem is None
                                 else 0.05 * best_y + 0.95 * self._horizon_mem)
            return best_y
        return self._horizon_mem if self._horizon_mem is not None else 0.5 * self._h

    def _raw_distance(self, track, horizon: float) -> float | None:
        """Distance GÉNÉRALE : a priori de TAILLE en principal (ne dépend que du champ
        de vision -> stable d'une vidéo à l'autre), plan-sol en clamp de sécurité.

        Le plan-sol seul dépend d'un horizon correct qui change à chaque vidéo (montées,
        descentes) -> distances aberrantes. La largeur, elle, généralise.
        """
        real_w = _REAL_WIDTH_M.get(track.cls)
        z_w = (self._f * real_w / track.bbox.width
               if real_w is not None and track.bbox.width > 1.0 else None)

        denom = track.bbox.y2 - horizon
        z_g = self._f * self._H / denom if denom > 8 else None

        z = z_w if z_w is not None else z_g     # largeur PRINCIPALE, sol en repli
        if z is None:
            return None
        # Plan-sol comme borne basse de sécurité (empêche de sur-estimer un objet collé).
        if z_g is not None:
            z = min(z, max(z_g, 1.0) * 1.6)
        return float(min(200.0, max(1.0, z)))

    def update(self, world: WorldState) -> None:
        horizon = self._horizon(world)
        alive_ids = set()
        for t in world.tracks:
            dist = self._raw_distance(t, horizon)
            if dist is None:
                continue
            alive_ids.add(t.id)

            st = self._states.get(t.id)
            if st is None:
                self._states[t.id] = _State(last_t=world.timestamp, distance=dist)
                t.distance_m = round(dist, 1)
                continue

            dt = world.timestamp - st.last_t
            if dt <= 1e-6:
                t.distance_m = round(st.distance, 1)
                continue

            # LISSAGE de la distance, avec REJET DES SAUTS BRUSQUES : un véhicule qui
            # tourne / change de voie voit sa largeur apparente sauter -> la distance par
            # a priori de taille bondit artificiellement. Un saut > 40 % en une frame est
            # traité comme suspect et fortement amorti (au lieu de polluer vitesse/accél.
            # et de déclencher de faux « freinage » / « changement de voie »).
            ratio = dist / st.distance if st.distance > 1e-3 else 1.0
            alpha = 0.15 if (ratio > 1.4 or ratio < 0.71) else 0.4
            dist_s = alpha * dist + (1 - alpha) * st.distance
            closing = -(dist_s - st.distance) / dt         # + si la distance diminue
            # PLAFOND réaliste de la vitesse de rapprochement (anti-TTC aberrant).
            closing = max(-15.0, min(15.0, closing))
            prev_speed = st.speed
            st.speed = self._ema * closing + (1 - self._ema) * st.speed
            raw_accel = (st.speed - prev_speed) / dt
            st.accel = self._ema * raw_accel + (1 - self._ema) * st.accel
            st.distance = dist_s
            st.last_t = world.timestamp

            t.distance_m = round(dist_s, 1)
            t.speed_mps = round(st.speed, 2)
            t.accel_mps2 = round(st.accel, 2)

            # TTC : seulement si l'objet se rapproche réellement.
            if st.speed > 0.3:
                t.ttc_s = round(dist_s / st.speed, 2)
            else:
                t.ttc_s = None

        # Nettoyage des états dont le track a disparu.
        for tid in list(self._states):
            if tid not in alive_ids:
                del self._states[tid]
