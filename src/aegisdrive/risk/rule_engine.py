"""Moteur de risque à règles — explicable par construction (PAS un réseau de neurones).

Score de danger 0..100 = somme pondérée de facteurs interprétables. Chaque facteur qui
dépasse un seuil ajoute une phrase d'explication au track. On peut lire, débugger et
justifier chaque décision — exigence clé du PRD.

Phase 1 : facteurs disponibles = classe + taille apparente (proxy de proximité).
Phase 2+ : distance/vitesse/TTC réels affineront ces règles, même interface.
"""
from __future__ import annotations

from ..schemas import Behavior, LaneZone, ObjectClass, WorldState

# Bonus de risque additifs par comportement (appliqués après pondération de voie).
_BEHAVIOR_BONUS = {
    Behavior.CUT_IN: (28.0, "Se rabat vers ma voie"),
    Behavior.HARD_BRAKING: (22.0, "Freinage brusque devant"),
    Behavior.PED_CROSSING: (30.0, "Piéton qui traverse"),
    Behavior.STOPPED: (12.0, "Véhicule immobilisé"),
}

# Pondération du risque selon la zone de voie. La chaussée opposée (séparée par un
# terre-plein) est fortement dévaluée : pas de trajectoire de collision avec l'ego.
_LANE_WEIGHT = {
    LaneZone.EGO: 1.0,
    LaneZone.ADJACENT_RIGHT: 0.7,   # rabattement/dépassement possible
    LaneZone.ADJACENT_LEFT: 0.55,
    LaneZone.OPPOSITE: 0.15,        # chaussée séparée -> risque faible
    LaneZone.UNKNOWN: 0.8,
}

# Vulnérabilité intrinsèque par catégorie (usagers vulnérables = plus prudent).
_VULNERABILITY = {
    ObjectClass.PEDESTRIAN: 40.0,
    ObjectClass.BICYCLE: 35.0,
    ObjectClass.MOTORCYCLE: 30.0,
    ObjectClass.ANIMAL: 25.0,
    ObjectClass.CAR: 15.0,
    ObjectClass.TRUCK: 20.0,
    ObjectClass.BUS: 20.0,
    ObjectClass.OBSTACLE: 20.0,
}


class RuleRiskEngine:
    def __init__(self, frame_size: tuple[int, int]):
        self._w, self._h = frame_size
        self._frame_area = float(self._w * self._h)

    def assess(self, world: WorldState) -> None:
        cond = world.conditions
        for track in world.tracks:
            score = 0.0
            reasons: list[str] = []

            # Facteur 1 : vulnérabilité de la catégorie.
            vuln = _VULNERABILITY.get(track.cls, 10.0)
            score += vuln
            if vuln >= 30.0:
                reasons.append(f"Usager vulnérable ({track.cls.value})")

            # Facteur 2 : proximité apparente (aire de la bbox / aire image).
            occupancy = track.bbox.area / self._frame_area if self._frame_area else 0.0
            proximity = min(60.0, occupancy * 400.0)
            score += proximity
            if occupancy > 0.08:
                reasons.append("Objet proche / grande emprise visuelle")

            # Facteur 3 : TTC si disponible (Phase 2+).
            if track.ttc_s is not None and track.ttc_s < 3.0:
                score += 30.0
                reasons.append(f"Collision probable dans {track.ttc_s:.1f} s")

            # Facteur 4 : vitesse de rapprochement élevée.
            if track.speed_mps is not None and track.speed_mps > 8.0:
                score += 15.0
                reasons.append("Objet se rapprochant rapidement")

            # Facteur 5 : visibilité réduite (nuit/météo) -> marge de réaction moindre.
            if cond is not None and cond.visibility < 0.5:
                score += (0.5 - cond.visibility) * 40.0
                reasons.append(f"Visibilité réduite ({cond.label})")

            # Pondération par zone de voie (chaussée opposée séparée -> risque faible).
            weight = _LANE_WEIGHT.get(track.lane_zone, 0.8)
            score *= weight
            if track.lane_zone is LaneZone.OPPOSITE:
                reasons.append("Voie opposée (séparée) — risque réduit")

            # Bonus additifs par comportement (Étape 2), après pondération.
            for beh in track.behaviors:
                bonus = _BEHAVIOR_BONUS.get(beh)
                if bonus is not None:
                    score += bonus[0]
                    reasons.append(bonus[1])

            # PLANCHER de danger par PROXIMITÉ absolue : un objet très proche est ROUGE
            # quelle que soit sa voie, pour qu'un système automatique reste prudent.
            d = track.distance_m
            if d is not None:
                if d < 6.0:
                    score = max(score, 78.0)          # rouge (prudence)
                    reasons.append("Objet très proche (<6 m)")
                elif d < 10.0 and track.lane_zone is not LaneZone.OPPOSITE:
                    score = max(score, 50.0)          # orange
                    reasons.append("Objet proche (<10 m)")

            track.danger_score = round(min(100.0, score), 1)
            track.explanations = reasons
