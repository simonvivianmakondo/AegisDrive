"""Détection de voies (CV classique) et assignation de zone par objet — compréhension de scène.

MÉTHODE : Canny + Hough probabiliste dans une ROI trapézoïdale au sol. Les segments
sont séparés en frontière gauche / droite (par position et pente), puis chacune est
approchée par une droite x = m·y + b. On en déduit le **corridor de l'ego-voie**.

Chaque track est ensuite classé selon la position latérale de son point de contact au
sol (bas de la bbox) par rapport au centre du corridor, normalisée par la largeur de
voie :
  |offset| < 0.5 voie          -> EGO
  côté conducteur adjacent     -> ADJACENT_RIGHT (dépassement/rabattement possible)
  côté opposé, proche          -> ADJACENT_LEFT
  côté opposé, loin (> ~1.5 v) -> OPPOSITE  (chaussée séparée / terre-plein)

HYPOTHÈSE : circulation à droite par défaut (les véhicules venant en face sont à
GAUCHE). Réglable via `drive_side`. La détection du terre-plein physique en monoculaire
est mal posée : on l'infère du fait qu'un objet est nettement sur la chaussée opposée.
La Phase segmentation (deep) remplacera ce module sans changer l'interface.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from ..schemas import LaneZone, WorldState


@dataclass
class LaneLine:
    """Ligne de voie polynomiale : x = a·y² + b·y + c (a=0 -> droite)."""
    a: float
    b: float
    c: float
    y_lo: float = 0.0    # plage VERTICALE des données réelles (pour ne pas extrapoler)
    y_hi: float = 0.0

    def x_at(self, y: float) -> float:
        return self.a * y * y + self.b * y + self.c

    @classmethod
    def fit(cls, ys, xs, degree: int = 2, y_scale: float = 1.0) -> Optional["LaneLine"]:
        """Ajuste x=f(y) avec rejet ROBUSTE des points aberrants (RANSAC allégé).

        `y_scale` (=hauteur image) normalise y dans [0,1] pour un conditionnement
        numérique correct (sans ça, y² ~1e6 -> coefficients aberrants).

        Un `polyfit` brut se laisse tordre par un seul segment parasite (glissière,
        ombre, marquage voisin). On ajuste donc, puis on écarte itérativement les
        points dont le résidu dépasse 2σ, et on ré-ajuste sur les points restants
        (méthode IRLS/RANSAC simplifiée) -> lignes bien plus stables.
        """
        if len(xs) < 2:
            return None
        ys = np.asarray(ys, dtype=float)
        xs = np.asarray(xs, dtype=float)
        t = ys / y_scale                                # y normalisé
        spread = float(t.max() - t.min()) if len(t) else 0.0
        deg = 2 if (degree >= 2 and len(xs) >= 4 and spread > 0.15) else 1
        coef = cls._robust_polyfit(t, xs, deg)
        y_lo, y_hi = float(ys.min()), float(ys.max())
        if deg == 1:
            b, c = coef
            return cls(0.0, float(b) / y_scale, float(c), y_lo, y_hi)
        a, b, c = coef
        return cls(float(a) / (y_scale ** 2), float(b) / y_scale, float(c), y_lo, y_hi)

    @staticmethod
    def _robust_polyfit(t: np.ndarray, xs: np.ndarray, deg: int,
                        iters: int = 2, sigma: float = 2.0) -> np.ndarray:
        """polyfit avec réjection itérative des résidus > `sigma`·écart-type."""
        keep = np.ones(len(xs), dtype=bool)
        coef = np.polyfit(t, xs, deg)
        min_pts = deg + 1
        for _ in range(iters):
            resid = np.abs(np.polyval(coef, t) - xs)
            s = resid[keep].std()
            if s < 1e-6:
                break
            new_keep = resid <= sigma * s
            # On ne réduit jamais sous le minimum de points requis pour le degré.
            if new_keep.sum() < min_pts or np.array_equal(new_keep, keep):
                break
            keep = new_keep
            coef = np.polyfit(t[keep], xs[keep], deg)
        return coef


@dataclass
class LaneContext:
    found: bool
    height: int
    width: int
    left: Optional[LaneLine] = None
    right: Optional[LaneLine] = None
    center_x_bottom: float = 0.0
    lane_width_px: float = 0.0
    y_top: float = 0.0              # ligne la plus haute où le corridor a des données
    # Détection voie-par-voie (Étape voies individuelles).
    lane_lines: object = None       # list[LaneLine] ordonnées gauche->droite
    num_lanes: int = 0              # nombre de voies détectées
    ego_index: int = 0              # index (0-based depuis la gauche) de la voie de l'ego
    lane_change: str = ""           # "" / "left" / "right" : changement de voie en cours
    # Point de fuite (horizon) du corridor — calculé UNE fois ici puis partagé par la
    # cinématique et l'ego-motion (évite deux balayages identiques et garantit la
    # cohérence distance/vitesse). 0.0 = inconnu.
    horizon_y: float = 0.0
    # Fiabilité du corridor affiché : 1.0 = corridor frais (cible détectée cette frame),
    # décroît quand on « glisse » sans cible (perte brève). Pilote la couleur du tracé.
    corridor_confidence: float = 1.0
    # Rayon de courbure estimé de la voie (m) + sens ("left"/"right"/""). Exploite
    # l'ajustement degré 2 ; None/"" si voie droite ou non calculable.
    curvature_radius_m: Optional[float] = None
    curvature_dir: str = ""
    # Masques optionnels (remplis par la segmentation IA — Étape 3). None en classique.
    drivable_mask: object = None    # np.ndarray bool HxW (zone roulable)
    lane_mask: object = None        # np.ndarray bool HxW (lignes de voie)

    def center_x_at(self, y: float) -> float:
        if self.left and self.right:
            return (self.left.x_at(y) + self.right.x_at(y)) / 2.0
        return self.center_x_bottom


# --------------------------------------------------------------------------- #
# Fonctions géométriques PARTAGÉES (corridor -> horizon, courbure).
# Un seul endroit de vérité, réutilisé par kinematics et ego_motion.
# --------------------------------------------------------------------------- #
def corridor_horizon(left: Optional[LaneLine], right: Optional[LaneLine],
                     height: int, samples: int = 60) -> Optional[float]:
    """Ligne d'horizon = point de fuite du corridor (là où gauche et droite convergent)."""
    if left is None or right is None:
        return None
    best_y, best_gap = None, 1e9
    for i in range(samples):
        y = height * i / (samples - 1)
        gap = abs(left.x_at(y) - right.x_at(y))
        if gap < best_gap:
            best_gap, best_y = gap, y
    if best_y is not None and 0 < best_y < height:
        return float(best_y)
    return None


def lane_curvature(left: Optional[LaneLine], right: Optional[LaneLine],
                   height: int, lane_width_px: float,
                   lane_width_m: float = 3.5) -> tuple[Optional[float], str]:
    """Rayon de courbure (m) et sens de la voie, au bas de l'image.

    Depuis x = a·y² + b·y + c : κ = |2a| / (1+(2a·y+b)²)^1.5, R_px = 1/κ. L'échelle
    px→m vient de la largeur de voie apparente (≈3.5 m réels). Estimation honnête
    (plan-sol, vue de face) : le SENS est fiable, la valeur est à un facteur près.
    """
    if left is None or right is None or lane_width_px <= 1.0:
        return None, ""
    y = float(height - 1)
    a = (left.a + right.a) / 2.0
    b = (left.b + right.b) / 2.0
    if abs(a) < 1e-7:
        return None, ""                         # voie quasi droite
    kappa = abs(2.0 * a) / (1.0 + (2.0 * a * y + b) ** 2) ** 1.5
    if kappa < 1e-9:
        return None, ""
    r_px = 1.0 / kappa
    r_m = r_px * (lane_width_m / lane_width_px)
    # Sens : comparer le centre du corridor en haut vs en bas de la plage commune.
    y_top = max(left.y_lo, right.y_lo)
    c_top = (left.x_at(y_top) + right.x_at(y_top)) / 2.0
    c_bot = (left.x_at(y) + right.x_at(y)) / 2.0
    direction = "left" if c_top < c_bot else "right"
    return round(r_m, 1), direction


class LaneEstimator:
    def __init__(self, drive_side: str = "right"):
        self._drive_side = drive_side  # "right" -> face-à-face à gauche

    def estimate(self, frame) -> LaneContext:
        img = frame.image
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)

        # ROI trapézoïdale (moitié basse = la route devant).
        mask = np.zeros_like(edges)
        poly = np.array([[
            (int(0.05 * w), h), (int(0.42 * w), int(0.62 * h)),
            (int(0.58 * w), int(0.62 * h)), (int(0.95 * w), h),
        ]], dtype=np.int32)
        cv2.fillPoly(mask, poly, 255)
        edges = cv2.bitwise_and(edges, mask)

        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=40,
                                minLineLength=40, maxLineGap=120)
        if lines is None:
            return LaneContext(False, h, w)

        lx, ly, rx, ry = [], [], [], []
        for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):
            if y2 == y1:
                continue
            slope = (y2 - y1) / (x2 - x1 + 1e-6)   # dy/dx (y vers le bas)
            if abs(slope) < 0.5:            # ignore les segments quasi horizontaux
                continue
            xm = (x1 + x2) / 2.0
            # SÉPARATION par SIGNE DE PENTE (plus robuste que la position du milieu, qui
            # se croise en virage / voie décentrée) : la ligne gauche monte vers la
            # droite (pente < 0), la droite monte vers la gauche (pente > 0). La position
            # ne sert que de garde-fou léger contre le bruit du mauvais côté.
            if slope < 0 and xm < w * 0.60:
                lx += [x1, x2]; ly += [y1, y2]
            elif slope > 0 and xm > w * 0.40:
                rx += [x1, x2]; ry += [y1, y2]

        left = self._fit(ly, lx)
        right = self._fit(ry, rx)
        if left is None or right is None:
            return LaneContext(False, h, w, left, right)

        xl_b, xr_b = left.x_at(h), right.x_at(h)
        lane_w = max(1.0, xr_b - xl_b)
        r_m, r_dir = lane_curvature(left, right, h, lane_w)
        return LaneContext(True, h, w, left, right,
                           center_x_bottom=(xl_b + xr_b) / 2.0,
                           lane_width_px=lane_w,
                           horizon_y=corridor_horizon(left, right, h) or 0.0,
                           corridor_confidence=1.0,
                           curvature_radius_m=r_m, curvature_dir=r_dir)

    @staticmethod
    def _fit(ys, xs) -> Optional[LaneLine]:
        # Hough -> droite (degré 1). La segmentation (road_segmenter) fera du degré 2.
        return LaneLine.fit(ys, xs, degree=1)

    def assign(self, world: WorldState, ctx: LaneContext) -> None:
        if not ctx.found:
            for t in world.tracks:
                t.lane_zone = LaneZone.UNKNOWN
            return
        opposite_side = -1 if self._drive_side == "right" else +1  # -1 = gauche
        for t in world.tracks:
            gx, gy = t.bbox.center[0], t.bbox.y2   # point de contact au sol
            offset = (gx - ctx.center_x_at(gy)) / ctx.lane_width_px
            if abs(offset) < 0.5:
                t.lane_zone = LaneZone.EGO
            elif np.sign(offset) == opposite_side:
                t.lane_zone = (LaneZone.OPPOSITE if abs(offset) > 1.5
                               else LaneZone.ADJACENT_LEFT)
            else:
                t.lane_zone = LaneZone.ADJACENT_RIGHT
