"""Enrichissement de l'état des objets — Étape 1 (100% classique, aucune IA).

À partir de grandeurs DÉJÀ calculées (distance/vitesse/accél. par Kalman) et d'un petit
historique de position par track, on dérive :

  - `motion_state` : état longitudinal.
        * Sans vitesse ego (dashcam) -> repère RELATIF : CLOSING / CLOSING_FAST /
          MATCHING / RECEDING. C'est honnête : on ne peut pas savoir si un objet est
          "arrêté" dans le monde sans connaître notre propre vitesse.
        * Avec `world.ego_speed_mps` (mode temps réel, CAN/GPS) -> repère ABSOLU :
          STOPPED / CRUISING / ACCELERATING / DECELERATING / BRAKING_HARD.
          (hypothèse : objet dans le même sens ; V_obj = V_ego - V_rapprochement)
  - `lateral_state` : dérive latérale (KEEPING / TO_LEFT / TO_RIGHT), calculée sur la
    vitesse du centre de la bbox normalisée par sa largeur -> INDÉPENDANTE de l'ego et
    de l'échelle. C'est l'amorce du "changement de voie" exploité à l'Étape 2.

Implémente l'interface `StateEstimator`. Ne casse aucune interface : s'insère après
l'estimateur cinématique et mute les tracks en place.
"""
from __future__ import annotations

from collections import deque

from ..schemas import LateralState, MotionState, WorldState

# Seuils (réglables). Vitesses en m/s, accél. en m/s².
_MATCH_MS = 0.5        # |v_rel| en deçà -> même allure
_FAST_MS = 8.0         # rapprochement rapide
_STOP_MS = 0.8         # |v_abs| en deçà -> arrêté
_DECEL = -0.7
_BRAKE = -3.5
_ACCEL = 0.7
_LAT_NORM = 0.9        # dérive latérale (largeurs de bbox par seconde) -> changement
                       # durci (0.5->0.9) : le défilement des véhicules adjacents sous
                       # l'effet du mouvement ego créait de fausses dérives latérales.


class StateEstimator:
    def __init__(self, history: int = 6):
        # Historique (timestamp, cx) par id de track, pour la vitesse latérale.
        self._hist: dict[int, deque] = {}
        self._history = history
        # Vitesse absolue lissée par track : id -> (timestamp, v_lisse, a_lisse).
        self._abs: dict[int, tuple] = {}

    def update(self, world: WorldState) -> None:
        ego = world.ego_speed_mps
        measured = world.ego_speed_measured
        now = world.timestamp
        alive = set()
        for t in world.tracks:
            alive.add(t.id)
            t.motion_state = self._longitudinal(t, ego, now, measured)
            t.lateral_state = self._lateral(t, now)

        # Purge des historiques dont le track a disparu.
        for tid in list(self._hist):
            if tid not in alive:
                del self._hist[tid]
        for tid in list(self._abs):
            if tid not in alive:
                del self._abs[tid]

    def _longitudinal(self, t, ego, now, measured) -> MotionState:
        if t.speed_mps is None:
            return MotionState.UNKNOWN

        if ego is not None:
            # Vitesse absolue brute (même sens) : V_obj = V_ego - V_rapprochement.
            v_raw = ego - t.speed_mps
            # Lissage FORT de la vitesse ET dérivation de l'accél. sur cette valeur lissée.
            prev = self._abs.get(t.id)
            if prev is None:
                sv, sa = v_raw, 0.0
            else:
                p_ts, p_sv, p_sa = prev
                dt = now - p_ts
                sv = 0.35 * v_raw + 0.65 * p_sv
                a_raw = (sv - p_sv) / dt if dt > 1e-6 else 0.0
                sa = 0.3 * a_raw + 0.7 * p_sa
            self._abs[t.id] = (now, sv, sa)
            t.speed_abs_mps = round(sv, 2)

            if abs(sv) < _STOP_MS:
                return MotionState.STOPPED
            # Les états accél/freinage ne sont fiables qu'avec une vitesse MESURÉE.
            # En vitesse ESTIMÉE (monoculaire, bruitée) on se limite à arrêté/roule.
            if not measured:
                return MotionState.CRUISING
            if sa < _BRAKE:
                return MotionState.BRAKING_HARD
            if sa < _DECEL:
                return MotionState.DECELERATING
            if sa > _ACCEL:
                return MotionState.ACCELERATING
            return MotionState.CRUISING

        # Repère relatif (dashcam).
        s = t.speed_mps
        if abs(s) < _MATCH_MS:
            return MotionState.MATCHING
        if s > _FAST_MS:
            return MotionState.CLOSING_FAST
        if s > 0:
            return MotionState.CLOSING
        return MotionState.RECEDING

    def _lateral(self, t, now) -> LateralState:
        hist = self._hist.setdefault(t.id, deque(maxlen=self._history))
        cx = t.bbox.center[0]
        hist.append((now, cx))
        if len(hist) < 3:
            return LateralState.KEEPING

        (t0, x0), (t1, x1) = hist[0], hist[-1]
        dt = t1 - t0
        if dt <= 1e-6:
            return LateralState.KEEPING
        # Vitesse latérale normalisée par la largeur de bbox -> invariante à l'échelle.
        v_norm = ((x1 - x0) / dt) / max(1.0, t.bbox.width)
        if v_norm > _LAT_NORM:
            return LateralState.TO_RIGHT
        if v_norm < -_LAT_NORM:
            return LateralState.TO_LEFT
        return LateralState.KEEPING
