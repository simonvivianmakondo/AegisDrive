"""Reconnaissance de comportements — Étape 2 (classique, avec garde-fous anti-faux-positifs).

Consomme l'historique par track (zone de voie, dérive latérale, cinématique, état de
mouvement) et en déduit des ÉVÉNEMENTS lisibles.

ROBUSTESSE (leçon des premières mesures : trop de faux "changement de voie") :
  1. ZONES CONFIRMÉES (debounce) — une zone n'est prise en compte que si elle PERSISTE
     `confirm` frames. Un changement de voie n'est émis qu'entre deux zones *confirmées*
     successives et différentes -> immunisé au jitter image par image de la détection.
  2. ÂGE MINIMUM — aucun comportement tant que le track n'a pas `min_age` frames
     (évite le bruit de démarrage du filtre de Kalman).
  3. ÉVÉNEMENTS PERSISTANTS — un événement ponctuel (changement de voie, cut-in) reste
     "actif" `hold` frames pour rester lisible sans être ré-émis en boucle.
  4. SEUILS DURCIS — le freinage exige une accélération de rapprochement forte, SOUTENUE,
     et un objet réellement en rapprochement à distance plausible.

Aucune IA : transitions d'état explicites, lisibles et débuggables.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from ..schemas import (Behavior, LaneZone, LateralState, MotionState,
                       ObjectClass, WorldState)

_ZONE_IDX = {
    LaneZone.OPPOSITE: -2,
    LaneZone.ADJACENT_LEFT: -1,
    LaneZone.EGO: 0,
    LaneZone.ADJACENT_RIGHT: 1,
}
_VEHICLES = {ObjectClass.CAR, ObjectClass.TRUCK, ObjectClass.BUS,
            ObjectClass.MOTORCYCLE}

# --- Paramètres de robustesse (réglables) ---
# Seuils DURCIS : en monoculaire, boîtes et cinématique sont bruitées -> on n'émet un
# comportement que sur un signal FRANC et SOUTENU, quitte à en manquer quelques-uns
# (moins de faux « freinage » / « changement de voie », qui polluaient l'affichage).
_MIN_AGE = 6           # frames avant d'évaluer un comportement (bruit de démarrage)
_CONFIRM = 12          # frames identiques pour "confirmer" une zone (~0.4s, anti-virage)
_HOLD = 15             # frames de persistance d'un événement ponctuel (~0.5 s @30fps)
_LAT_SUSTAIN = 14      # frames de dérive latérale soutenue (secours, zones inconnues)
_BRAKE_ACCEL = 9.0     # m/s² sur la vitesse de rapprochement (durci : anti-faux-freinage)
_BRAKE_HOLD = 5        # frames soutenues pour valider un freinage
_FAST_HOLD = 6         # frames de closing_fast -> arrivée rapide


@dataclass
class _Snap:
    zone: LaneZone
    lateral: LateralState
    distance: float | None
    speed: float | None
    accel: float | None
    motion: MotionState
    on_road: bool | None = None   # le point de contact est-il dans la zone roulable ?


@dataclass
class _TrackState:
    hist: deque = field(default_factory=lambda: deque(maxlen=12))
    confirmed: LaneZone | None = None       # zone actuellement confirmée
    last_confirmed: LaneZone | None = None   # zone confirmée précédente (pour le delta)
    holds: dict = field(default_factory=dict)  # Behavior -> frames restantes


class BehaviorEngine:
    def __init__(self):
        self._states: dict[int, _TrackState] = {}

    def update(self, world: WorldState) -> None:
        drivable = getattr(world.lane_ctx, "drivable_mask", None)
        alive = set()
        for t in world.tracks:
            alive.add(t.id)
            ts = self._states.setdefault(t.id, _TrackState())
            ts.hist.append(_Snap(t.lane_zone, t.lateral_state, t.distance_m,
                                 t.speed_mps, t.accel_mps2, t.motion_state,
                                 on_road=self._on_road(drivable, t)))
            t.behaviors = self._detect(t, ts)
        for tid in list(self._states):
            if tid not in alive:
                del self._states[tid]

    @staticmethod
    def _on_road(drivable, t) -> bool | None:
        """Le point de contact au sol (bas de bbox) est-il dans la zone roulable ?"""
        if drivable is None:
            return None
        h, w = drivable.shape[:2]
        x = int(min(w - 1, max(0, t.bbox.center[0])))
        y = int(min(h - 1, max(0, t.bbox.y2)))
        return bool(drivable[y, x])

    # ------------------------------------------------------------------ #
    def _detect(self, t, ts: _TrackState) -> list:
        snaps = list(ts.hist)
        active: list = []

        # Décrémente les événements persistants en cours.
        for beh in list(ts.holds):
            ts.holds[beh] -= 1
            if ts.holds[beh] <= 0:
                del ts.holds[beh]

        # Garde-fou : track trop jeune -> on ne fait que maintenir les holds.
        if t.age < _MIN_AGE:
            return self._holds_as_list(ts)

        # 1) Mise à jour de la zone CONFIRMÉE (debounce).
        self._update_confirmed_zone(ts, snaps)

        # 2) Changement de voie = transition entre deux zones confirmées (véhicules).
        #    Garde-fou : un objet TRÈS proche (<5 m) ne peut pas basculer de voie d'un
        #    coup -> c'est une erreur de projection en virage. On ignore ces bascules.
        too_close = t.distance_m is not None and t.distance_m < 5.0
        if (t.cls in _VEHICLES and not too_close and ts.confirmed is not None
                and ts.last_confirmed is not None
                and ts.confirmed != ts.last_confirmed):
            delta = _ZONE_IDX[ts.confirmed] - _ZONE_IDX[ts.last_confirmed]
            beh = Behavior.LANE_CHANGE_LEFT if delta < 0 else Behavior.LANE_CHANGE_RIGHT
            ts.holds[beh] = _HOLD
            # Cut-in : aboutit dans MA voie depuis une adjacente, en se rapprochant.
            if ts.confirmed is LaneZone.EGO and (t.speed_mps or 0) > 0:
                ts.holds[Behavior.CUT_IN] = _HOLD
            ts.last_confirmed = ts.confirmed   # consomme la transition (émise une fois)

        # 2b) Secours (zones jamais confirmées) : dérive latérale TRÈS soutenue.
        elif (t.cls in _VEHICLES and ts.confirmed is None
              and self._sustained_lateral(snaps)):
            d = snaps[-1].lateral
            beh = (Behavior.LANE_CHANGE_LEFT if d is LateralState.TO_LEFT
                   else Behavior.LANE_CHANGE_RIGHT)
            ts.holds.setdefault(beh, _HOLD)

        # 3) Comportements CONTINUS (recalculés chaque frame, seuils durcis).
        if self._tail(snaps, lambda s: s.motion is MotionState.CLOSING_FAST, _FAST_HOLD):
            active.append(Behavior.APPROACHING_FAST)

        if (self._tail(snaps, lambda s: (s.accel or 0) > _BRAKE_ACCEL, _BRAKE_HOLD)
                and (t.speed_mps or 0) > 2.0
                and t.distance_m is not None and 4.0 < t.distance_m < 60.0):
            active.append(Behavior.HARD_BRAKING)

        if (t.cls in _VEHICLES
                and t.lane_zone in (LaneZone.ADJACENT_LEFT, LaneZone.ADJACENT_RIGHT)
                and self._tail(snaps, lambda s: (s.speed or 0) < -1.0, 4)):
            active.append(Behavior.OVERTAKING)

        if self._tail(snaps, lambda s: s.motion is MotionState.STOPPED, 4):
            active.append(Behavior.STOPPED)

        if t.cls is ObjectClass.PEDESTRIAN:
            on_road = snaps[-1].on_road
            if on_road is True:
                # Sur la chaussée -> traverse (le vrai danger), qu'il bouge ou non.
                active.append(Behavior.PED_CROSSING)
            elif on_road is False:
                # Sur le trottoir / hors chaussée -> attend, ne traverse pas.
                active.append(Behavior.PED_WAITING)
            else:
                # Pas de masque roulable (mode classique) : ancienne heuristique stricte.
                if self._sustained_lateral(snaps):
                    active.append(Behavior.PED_CROSSING)
                elif self._tail(snaps, lambda s: s.lateral is LateralState.KEEPING, 5):
                    active.append(Behavior.PED_WAITING)

        # Résultat = continus + événements persistants encore actifs.
        return active + self._holds_as_list(ts, exclude=active)

    # ------------------------------------------------------------------ #
    def _update_confirmed_zone(self, ts: _TrackState, snaps) -> None:
        """Confirme une zone si les `_CONFIRM` dernières frames sont identiques & connues."""
        if len(snaps) < _CONFIRM:
            return
        tail = [s.zone for s in snaps[-_CONFIRM:]]
        z = tail[0]
        if z in _ZONE_IDX and all(x is z for x in tail):
            if ts.confirmed is None:
                ts.confirmed = z
                ts.last_confirmed = z          # init sans événement
            elif z != ts.confirmed:
                ts.last_confirmed = ts.confirmed
                ts.confirmed = z

    @staticmethod
    def _holds_as_list(ts: _TrackState, exclude=()) -> list:
        return [b for b in ts.holds if b not in exclude]

    @staticmethod
    def _tail(snaps, pred, k) -> bool:
        return len(snaps) >= k and all(pred(s) for s in snaps[-k:])

    def _sustained_lateral(self, snaps) -> bool:
        for d in (LateralState.TO_LEFT, LateralState.TO_RIGHT):
            if self._tail(snaps, lambda s, d=d: s.lateral is d, _LAT_SUSTAIN):
                return True
        return False
