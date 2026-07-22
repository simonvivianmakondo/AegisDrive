"""Analyse des conditions + rehaussement d'image — robustesse nuit / météo.

Deux rôles combinés (une seule passe, économe) :
  1. ESTIMER les conditions de la scène à partir de statistiques de luminance :
       - luminosité moyenne  -> jour vs nuit
       - contraste (écart-type) -> proxy de brume / pluie / faible visibilité
  2. REHAUSSER l'image quand c'est utile (nuit ou faible visibilité) via CLAHE
     (égalisation d'histogramme adaptative à contraste limité) + correction gamma,
     pour que le détecteur reste performant. La géométrie n'est pas modifiée : les
     coordonnées restent valides, seul le rendu de la frame de détection change.

La `visibility` (0..1) est réutilisée par le moteur de risque (le PRD la liste comme
entrée de risque). L'affichage, lui, garde toujours l'image ORIGINALE.
"""
from __future__ import annotations

import cv2
import numpy as np

from ..schemas import Frame, SceneConditions


class SceneAnalyzer:
    def __init__(self, night_brightness: float = 70.0,
                 low_contrast: float = 45.0, enhance: bool = True):
        self._night_brightness = night_brightness
        self._low_contrast = low_contrast
        self._enhance = enhance
        self._clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        # LUT gamma (<1 éclaircit les basses lumières).
        gamma = 0.6
        self._gamma_lut = np.array(
            [((i / 255.0) ** gamma) * 255 for i in range(256)], dtype=np.uint8)

    def _conditions(self, image: np.ndarray) -> SceneConditions:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        brightness = float(gray.mean())
        contrast = float(gray.std())
        is_night = brightness < self._night_brightness

        # Visibilité : dégradée par un faible contraste (brume/pluie) et par la nuit.
        vis_contrast = np.clip(contrast / 60.0, 0.0, 1.0)
        vis_light = np.clip(brightness / self._night_brightness, 0.2, 1.0)
        visibility = float(np.clip(min(vis_contrast, vis_light), 0.0, 1.0))

        if is_night:
            label = "nuit" if visibility > 0.4 else "nuit / faible visibilité"
        elif visibility < 0.5:
            label = "faible visibilité (brume/pluie)"
        else:
            label = "jour clair"
        return SceneConditions(round(brightness, 1), round(contrast, 1),
                               is_night, round(visibility, 2), label)

    def _enhanced(self, image: np.ndarray, cond: SceneConditions) -> np.ndarray:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = self._clahe.apply(l)
        out = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
        if cond.is_night:
            out = cv2.LUT(out, self._gamma_lut)
        return out

    def process(self, frame: Frame) -> tuple[Frame, SceneConditions]:
        cond = self._conditions(frame.image)
        if self._enhance and (cond.is_night or cond.visibility < 0.6):
            img = self._enhanced(frame.image, cond)
            det_frame = Frame(frame.index, frame.timestamp, img)
        else:
            det_frame = frame
        return det_frame, cond
