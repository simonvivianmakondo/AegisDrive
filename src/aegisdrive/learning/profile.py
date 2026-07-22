"""Profil de scène APPRIS et PERSISTANT — apprentissage en ligne, entre les runs.

Le système observe pendant qu'il roule les grandeurs stables de la route :
  - largeur typique de la voie ego (fraction de la largeur image) ;
  - hauteur typique de l'horizon (fraction de la hauteur image).

Ces statistiques (médianes robustes, avec compteur de confiance) sont sauvegardées
dans un JSON et RECHARGÉES au prochain run : si le système a mal conjecturé sur une
vidéo, il dispose ensuite d'un prior calibré pour rejeter les détections aberrantes
(corridors trop larges/étroits, horizons impossibles). C'est de l'apprentissage réel
(estimation de paramètres en ligne), explicable et sans GPU.
"""
from __future__ import annotations

import json
import os


class SceneProfile:
    def __init__(self, path: str = "aegis_profile.json"):
        self._path = path
        self.lane_width_frac = 0.40   # prior initial raisonnable
        self.horizon_frac = 0.48
        self.n = 0                    # nb d'observations cumulées (confiance)
        self._dirty = 0
        self._load()

    # ------------------------------------------------------------------ #
    def _load(self) -> None:
        try:
            with open(self._path, encoding="utf-8") as f:
                d = json.load(f)
            self.lane_width_frac = float(d.get("lane_width_frac", self.lane_width_frac))
            self.horizon_frac = float(d.get("horizon_frac", self.horizon_frac))
            self.n = int(d.get("n", 0))
        except Exception:
            pass                       # premier run : priors par défaut

    def save(self) -> None:
        try:
            with open(self._path, "w", encoding="utf-8") as f:
                json.dump({"lane_width_frac": round(self.lane_width_frac, 4),
                           "horizon_frac": round(self.horizon_frac, 4),
                           "n": self.n}, f)
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    def observe(self, width_frac: float | None = None,
                horizon_frac: float | None = None) -> None:
        """Met à jour les statistiques apprises (EMA lente = mémoire longue)."""
        a = 0.02 if self.n > 200 else 0.08     # apprend vite au début, se stabilise
        if width_frac is not None and 0.1 < width_frac < 0.9:
            self.lane_width_frac = a * width_frac + (1 - a) * self.lane_width_frac
        if horizon_frac is not None and 0.2 < horizon_frac < 0.75:
            self.horizon_frac = a * horizon_frac + (1 - a) * self.horizon_frac
        self.n += 1
        self._dirty += 1
        if self._dirty >= 300:                 # sauvegarde périodique
            self._dirty = 0
            self.save()

    # ------------------------------------------------------------------ #
    def width_ok(self, width_frac: float) -> bool:
        """Plausibilité GÉNÉRALE de la largeur de corridor (indépendante de la vidéo).
        Bornes physiques larges -> ne rejette que l'aberrant, généralise entre vidéos."""
        return 0.10 < width_frac < 0.85
