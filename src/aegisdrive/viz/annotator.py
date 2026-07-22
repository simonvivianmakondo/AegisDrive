"""Overlay ADAS : voies, boîtes (bbox/id/zone/cinématique), et panneau HUD stable.

Le HUD en haut à gauche donne un RÉSUMÉ persistant à chaque frame — conditions,
FPS, compteurs, et les objets les plus dangereux — parce que les infos par-objet
défilent trop vite pour être lues.
"""
from __future__ import annotations

import cv2
import numpy as np

from ..schemas import Behavior, LaneZone, LateralState, MotionState, WorldState

_BEHAV_TAG = {
    Behavior.LANE_CHANGE_LEFT: "chgt voie G",
    Behavior.LANE_CHANGE_RIGHT: "chgt voie D",
    Behavior.CUT_IN: "SE RABAT!",
    Behavior.OVERTAKING: "depasse",
    Behavior.HARD_BRAKING: "FREINE!",
    Behavior.STOPPED: "ARRETE",
    Behavior.APPROACHING_FAST: "ARRIVE VITE",
    Behavior.PED_CROSSING: "TRAVERSE!",
    Behavior.PED_WAITING: "attend",
}
# Comportements remontés en alertes HUD (les plus critiques).
_ALERT_BEHAVIORS = (Behavior.CUT_IN, Behavior.HARD_BRAKING,
                    Behavior.PED_CROSSING, Behavior.APPROACHING_FAST)

_MAX_LABELS = 6   # nombre maximal d'objets étiquetés en détail (anti-surcharge)

_MOTION_TAG = {
    MotionState.CLOSING: "rappr.",
    MotionState.CLOSING_FAST: "rappr!!",
    MotionState.MATCHING: "m.allure",
    MotionState.RECEDING: "eloigne",
    MotionState.STOPPED: "ARRETE",
    MotionState.CRUISING: "roule",
    MotionState.ACCELERATING: "accel.",
    MotionState.DECELERATING: "ralentit",
    MotionState.BRAKING_HARD: "FREINE!",
    MotionState.UNKNOWN: "",
}
_LAT_TAG = {
    LateralState.TO_LEFT: "<G",
    LateralState.TO_RIGHT: "D>",
    LateralState.KEEPING: "",
    LateralState.UNKNOWN: "",
}

_ZONE_TAG = {
    LaneZone.EGO: "MA VOIE",
    LaneZone.ADJACENT_LEFT: "adj.G",
    LaneZone.ADJACENT_RIGHT: "adj.D",
    LaneZone.OPPOSITE: "OPPOSEE",
    LaneZone.UNKNOWN: "",
}


def risk_color(score: float) -> tuple[int, int, int]:
    """Convention de couleurs par bandes de risque (BGR) :
    vert=sûr, jaune=surveillance, orange=attention, rouge=danger."""
    if score >= 70:
        return (0, 0, 255)      # rouge
    if score >= 45:
        return (0, 140, 255)    # orange
    if score >= 20:
        return (0, 220, 255)    # jaune
    return (0, 200, 0)          # vert


# Alias conservé pour compatibilité interne.
_color_for_score = risk_color


def _draw_masks(image: np.ndarray, ctx) -> None:
    """Superpose la zone roulable (vert) et les marquages détectés (rouge) — segmentation IA."""
    dm = getattr(ctx, "drivable_mask", None)
    lm = getattr(ctx, "lane_mask", None)
    if dm is not None:
        overlay = image.copy()
        overlay[dm] = (0, 180, 0)
        cv2.addWeighted(overlay, 0.35, image, 0.65, 0, image)
    # Marquages détectés en rouge translucide : rend la perception des lignes VISIBLE
    # (indépendamment du corridor ego jaune), y compris les lignes non retenues.
    if lm is not None:
        overlay = image.copy()
        overlay[lm] = (0, 0, 220)
        cv2.addWeighted(overlay, 0.45, image, 0.55, 0, image)


def _draw_lanes(image: np.ndarray, ctx) -> None:
    # Le corridor de voie ego (traits jaunes) a été retiré : la détection monoculaire
    # ne le posait pas de façon fiable sur toutes les vidéos. On conserve la zone
    # roulable (vert) et les marquages détectés (rouge), qui sont robustes et utiles.
    _draw_masks(image, ctx)


def _draw_lane_change(image: np.ndarray, direction: str) -> None:
    """Bandeau proéminent qui matérialise un changement de voie de l'ego."""
    h, w = image.shape[:2]
    arrow = "<<<" if direction == "left" else ">>>"
    text = (f"{arrow}  CHANGEMENT DE VOIE  {arrow}")
    scale = 1.0
    (tw, tht), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 3)
    x = (w - tw) // 2
    y = int(0.16 * h)
    overlay = image.copy()
    cv2.rectangle(overlay, (x - 16, y - tht - 12), (x + tw + 16, y + 14), (30, 20, 60), -1)
    cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0, 230, 255), 3, cv2.LINE_AA)


def draw_overlay(image: np.ndarray, world: WorldState, fps: float | None = None) -> np.ndarray:
    """Dessine UNIQUEMENT sur l'image route : voies, boîtes, tags. La synthèse
    (conditions, top danger, alertes) est rendue à part par `viz/dashboard.py`."""
    _draw_lanes(image, world.lane_ctx)
    lc = getattr(world.lane_ctx, "lane_change", "")
    if lc:
        _draw_lane_change(image, lc)

    # Priorité : on n'étiquette que le TOP N (danger puis proximité). Le reste = discret.
    def _priority(t):
        crit = 1 if any(bh in _ALERT_BEHAVIORS for bh in t.behaviors) else 0
        return (crit, t.danger_score, -(t.distance_m if t.distance_m is not None else 1e9))

    ranked = sorted(world.tracks, key=_priority, reverse=True)
    labeled_ids = set()
    for t in ranked[:_MAX_LABELS]:
        near = t.distance_m is None or t.distance_m <= 60.0
        crit = any(bh in _ALERT_BEHAVIORS for bh in t.behaviors)
        if crit or near or t.danger_score >= 20.0:
            labeled_ids.add(t.id)

    for t in world.tracks:
        b = t.bbox
        color = risk_color(t.danger_score)
        p1, p2 = (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2))
        if t.id not in labeled_ids:
            cv2.rectangle(image, p1, p2, color, 1)   # secondaire : boîte fine, aucun texte
            continue

        cv2.rectangle(image, p1, p2, color, 2)

        # Carte objet aérée, à fond sombre pour la lisibilité.
        title = f"{t.cls.value.upper()} #{t.id}"
        info = []
        if t.distance_m is not None:
            info.append(f"{t.distance_m:.0f}m")
        if t.ttc_s is not None:
            info.append(f"TTC {t.ttc_s:.1f}s")
        crit_tag = next((_BEHAV_TAG[bh] for bh in t.behaviors if bh in _ALERT_BEHAVIORS), "")
        _draw_card(image, p1, title, "  ".join(info), crit_tag, color)

    return image


def _draw_card(image, p1, title, info, alert, color):
    """Petite carte à deux lignes (+ alerte) avec fond sombre translucide."""
    x, y = p1[0], p1[1]
    lines = [title] + ([info] if info else [])
    fs = 0.45
    wtxt = max(cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)[0][0] for s in lines)
    lh = 16
    top = max(0, y - lh * len(lines) - (18 if alert else 0) - 4)
    # Fond sombre.
    overlay = image.copy()
    cv2.rectangle(overlay, (x, top), (x + wtxt + 10, y), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.5, image, 0.5, 0, image)
    yy = top + 13
    if alert:
        cv2.putText(image, alert, (x + 4, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 0, 255), 2, cv2.LINE_AA)
        yy += 18
    for i, s in enumerate(lines):
        col = color if i == 0 else (230, 230, 230)
        cv2.putText(image, s, (x + 4, yy), cv2.FONT_HERSHEY_SIMPLEX, fs, col, 1, cv2.LINE_AA)
        yy += lh
