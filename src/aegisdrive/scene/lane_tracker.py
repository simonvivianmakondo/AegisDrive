"""Suivi temporel des lignes de voie (tracker) — mémoire entre frames.

Problème résolu : jusqu'ici les lignes étaient re-détectées à ZÉRO chaque frame ->
tracé court, clignotant, instable en virage. Ici, comme pour les objets, chaque ligne
reçoit un ID persistant et est LISSÉE dans le temps (EMA sur ses coefficients).

Effets :
  - lignes stables et longues (une ligne perdue quelques frames est maintenue) ;
  - virages gérés (chaque ligne suit sa propre trajectoire, lissée) ;
  - détection FIABLE du changement de voie : quand une ligne SUIVIE (identité stable)
    traverse le centre de l'image (position de l'ego), c'est un vrai franchissement.

Reste 100% classique (association gloutonne + EMA), même esprit que le tracker d'objets.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .lanes import LaneLine


@dataclass
class _TLine:
    id: int
    line: LaneLine
    age: int = 0
    missed: int = 0
    prev_side: int = 0     # -1 = à gauche du centre, +1 = à droite, 0 = indéterminé


class LaneLineTracker:
    def __init__(self, ref_frac: float = 0.9, assoc_frac: float = 0.06,
                 max_missed: int = 6, min_hits: int = 2, alpha: float = 0.4):
        self._ref = ref_frac
        self._assoc = assoc_frac
        self._max_missed = max_missed
        self._min_hits = min_hits
        self._alpha = alpha
        self._tracks: list[_TLine] = []
        self._next = 0
        self._change_hold = 0
        self._change_dir = ""
        # Dérive latérale COLLECTIVE des lignes (px, intégrateur à fuite) : pendant un
        # changement de voie de l'ego, toutes les lignes glissent ensemble dans l'image.
        # Signal robuste même si les pistes individuelles perdent leur identité.
        self._drift = 0.0
        self._drift_cd = 0

    # ------------------------------------------------------------------ #
    def update(self, raw_lines: list, h: int, w: int) -> list:
        ref_y = self._ref * h
        assoc = self._assoc * w
        center = w / 2.0
        band = 0.035 * w

        # 1) Association gloutonne (piste <-> détection) par proximité en bas d'image.
        cand = []
        for ti, t in enumerate(self._tracks):
            tx = t.line.x_at(ref_y)
            for ri, r in enumerate(raw_lines):
                d = abs(r.x_at(ref_y) - tx)
                if d < assoc:
                    cand.append((d, ti, ri))
        cand.sort()
        m_t, m_r, pair = set(), set(), {}
        for d, ti, ri in cand:
            if ti in m_t or ri in m_r:
                continue
            m_t.add(ti); m_r.add(ri); pair[ti] = ri

        # 2) Mise à jour des pistes (lissage) / vieillissement + mesure de dérive.
        dxs = []
        kept: list[_TLine] = []
        for ti, t in enumerate(self._tracks):
            if ti in pair:
                dxs.append(raw_lines[pair[ti]].x_at(ref_y) - t.line.x_at(ref_y))
                t.line = self._blend(t.line, raw_lines[pair[ti]], self._alpha)
                t.age += 1; t.missed = 0
            else:
                t.age += 1; t.missed += 1
            if t.missed <= self._max_missed:
                kept.append(t)
        # 3) Nouvelles pistes pour les détections non associées.
        for ri, r in enumerate(raw_lines):
            if ri not in m_r:
                kept.append(_TLine(self._next, r, age=1, missed=0))
                self._next += 1
        self._tracks = kept

        # 4a) Changement de voie par DÉRIVE COLLECTIVE : toutes les lignes glissent
        # ensemble quand l'ego change de voie (robuste aux pertes d'identité).
        if self._change_hold > 0:
            self._change_hold -= 1
        med = float(np.median(dxs)) if dxs else 0.0
        self._drift = 0.95 * self._drift + med
        # Discriminant VIRAGE : en courbe, les lignes glissent aussi collectivement,
        # mais elles sont COURBÉES (|a| élevé). On n'émet pas dans ce cas.
        curv = (float(np.median([abs(t.line.a) for t in self._tracks
                                 if t.missed == 0])) if any(t.missed == 0 for t in self._tracks)
                else 0.0)
        in_curve = curv * (h ** 2) > 0.10 * w   # courbure ramenée en px sur la hauteur
        if self._drift_cd > 0:
            self._drift_cd -= 1
        elif abs(self._drift) > 0.13 * w and not in_curve:
            # lignes qui glissent vers la GAUCHE (drift<0) = ego qui va à DROITE.
            self._change_dir = "right" if self._drift < 0 else "left"
            self._change_hold = 22
            self._drift_cd = 90            # ~6 s : pas de rafale d'événements
            self._drift = 0.0
        elif in_curve:
            self._drift *= 0.8             # purge en virage (évite l'accumulation)

        # 4b) Et par franchissement d'une ligne suivie (cas lent, identité conservée).
        for t in self._tracks:
            if t.age < 4 or t.missed > 0:
                continue
            x = t.line.x_at(ref_y)
            side = -1 if x < center - band else (1 if x > center + band else t.prev_side)
            if (t.prev_side != 0 and side != 0 and side != t.prev_side
                    and self._drift_cd == 0):        # cooldown PARTAGÉ avec la dérive
                # ligne gauche->droite = monde vers la droite = ego vers la GAUCHE.
                self._change_dir = "left" if t.prev_side < 0 else "right"
                self._change_hold = 22
                self._drift_cd = 90
            t.prev_side = side

        # 5) Sortie : pistes confirmées (assez vues, récemment matchées).
        return [t.line for t in self._tracks
                if t.age >= self._min_hits and t.missed <= 2]

    @property
    def lane_change(self) -> str:
        return self._change_dir if self._change_hold > 0 else ""

    @staticmethod
    def _blend(old: LaneLine, new: LaneLine, a: float) -> LaneLine:
        return LaneLine(a * new.a + (1 - a) * old.a,
                        a * new.b + (1 - a) * old.b,
                        a * new.c + (1 - a) * old.c,
                        new.y_lo, new.y_hi)
