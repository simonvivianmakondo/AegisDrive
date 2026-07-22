"""Panneau latéral (dashboard) — rendu SÉPARÉ de l'image vidéo.

Auparavant ces infos étaient superposées en haut à gauche de la vidéo et la
surchargeaient. On les déplace dans un panneau dédié, composité à côté de la vidéo
par le sink. La vidéo reste lisible (route + boîtes + voies), le panneau porte la
synthèse. Même donnée source (`WorldState`), simplement un autre support d'affichage.
"""
from __future__ import annotations

import cv2
import numpy as np

from ..schemas import Behavior, LaneZone, WorldState

_BEHAV_TAG = {
    Behavior.LANE_CHANGE_LEFT: "chgt voie G",
    Behavior.LANE_CHANGE_RIGHT: "chgt voie D",
    Behavior.CUT_IN: "SE RABAT",
    Behavior.OVERTAKING: "depasse",
    Behavior.HARD_BRAKING: "FREINE",
    Behavior.STOPPED: "arrete",
    Behavior.APPROACHING_FAST: "arrive vite",
    Behavior.PED_CROSSING: "traverse",
    Behavior.PED_WAITING: "attend",
}
_ALERT_BEHAVIORS = (Behavior.CUT_IN, Behavior.HARD_BRAKING,
                    Behavior.PED_CROSSING, Behavior.APPROACHING_FAST)
_ZONE_TAG = {
    LaneZone.EGO: "MA VOIE", LaneZone.ADJACENT_LEFT: "adj.G",
    LaneZone.ADJACENT_RIGHT: "adj.D", LaneZone.OPPOSITE: "OPPOSEE",
    LaneZone.UNKNOWN: "",
}

_BG = (32, 30, 28)
_FONT = cv2.FONT_HERSHEY_SIMPLEX


from .annotator import risk_color as _color_for_score  # convention couleurs unifiée


def _text(img, s, x, y, scale=0.45, color=(220, 220, 220), thick=1):
    cv2.putText(img, s, (x, y), _FONT, scale, color, thick, cv2.LINE_AA)


def render_panel(world: WorldState, fps: float | None, width: int, height: int) -> np.ndarray:
    panel = np.full((height, width, 3), _BG, np.uint8)
    x = 12
    y = 26

    _text(panel, "AEGISDRIVE", x, y, 0.6, (0, 255, 255), 2); y += 26

    cond = world.conditions
    if cond is not None:
        _text(panel, f"Conditions : {cond.label}", x, y); y += 18
        _text(panel, f"Visibilite : {int(cond.visibility * 100)}%", x, y); y += 18
    _text(panel, f"FPS traitement : {fps:.1f}" if fps else "FPS : --", x, y); y += 18
    if world.ego_speed_mps is not None:
        _text(panel, f"Vitesse ego : {world.ego_speed_mps * 3.6:.0f} km/h (est.)",
              x, y, 0.5, (120, 220, 255)); y += 18
    y += 4

    # Compteurs par catégorie.
    counts: dict[str, int] = {}
    for t in world.tracks:
        counts[t.cls.value] = counts.get(t.cls.value, 0) + 1
    ctx = world.lane_ctx
    if ctx is not None and getattr(ctx, "num_lanes", 0) >= 1:
        _text(panel, f"Voies : {ctx.num_lanes}  |  ego en voie {ctx.ego_index + 1}",
              x, y, 0.5, (120, 220, 255)); y += 18
    # Courbure de la voie (exploite l'ajustement degré 2) : sens + rayon estimé.
    r_m = getattr(ctx, "curvature_radius_m", None) if ctx is not None else None
    if r_m is not None and 50 <= r_m < 1500:      # bande plausible (au-delà = ligne droite)
        arrow = "<-" if getattr(ctx, "curvature_dir", "") == "left" else "->"
        tight = "serre" if r_m < 150 else ("moyen" if r_m < 500 else "large")
        _text(panel, f"Virage {arrow} {tight} (~{r_m:.0f} m)", x, y, 0.5, (120, 220, 255)); y += 18

    _text(panel, f"Objets suivis : {len(world.tracks)}", x, y, 0.5, (255, 255, 255)); y += 18
    if counts:
        _text(panel, ", ".join(f"{k}:{v}" for k, v in counts.items()), x, y, 0.4); y += 20

    # Séparateur.
    cv2.line(panel, (x, y), (width - x, y), (80, 80, 80), 1); y += 20

    # Top danger avec barres.
    _text(panel, "TOP DANGER", x, y, 0.5, (255, 255, 255)); y += 20
    top = sorted(world.tracks, key=lambda t: t.danger_score, reverse=True)[:4]
    if not top:
        _text(panel, "(aucun objet)", x, y, 0.4, (150, 150, 150)); y += 18
    for t in top:
        col = _color_for_score(t.danger_score)
        dist = f"{t.distance_m:.0f}m" if t.distance_m is not None else "?"
        ttc = f" TTC{t.ttc_s:.1f}s" if t.ttc_s is not None else ""
        zone = _ZONE_TAG.get(t.lane_zone, "")
        _text(panel, f"#{t.id} {t.cls.value} {dist}{ttc} {zone}", x, y, 0.42, col); y += 15
        # Barre de danger.
        bar_w = int((width - 2 * x) * min(1.0, t.danger_score / 100.0))
        cv2.rectangle(panel, (x, y - 8), (x + bar_w, y), col, -1)
        cv2.rectangle(panel, (x, y - 8), (width - x, y), (70, 70, 70), 1)
        _text(panel, f"{t.danger_score:.0f}", width - x - 26, y - 1, 0.4, (255, 255, 255))
        y += 16

    # Alertes comportementales.
    y += 8
    cv2.line(panel, (x, y), (width - x, y), (80, 80, 80), 1); y += 20
    _text(panel, "ALERTES", x, y, 0.5, (0, 140, 255)); y += 20
    alerts = []
    for t in world.tracks:
        for beh in t.behaviors:
            if beh in _ALERT_BEHAVIORS:
                alerts.append((t.id, _BEHAV_TAG.get(beh, beh.value)))
    if not alerts:
        _text(panel, "-", x, y, 0.45, (150, 150, 150)); y += 18
    for tid, txt in alerts[:6]:
        _text(panel, f"! #{tid}  {txt}", x, y, 0.45, (0, 165, 255)); y += 18

    return panel
