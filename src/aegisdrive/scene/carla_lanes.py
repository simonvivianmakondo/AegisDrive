"""Fournisseur de voies basé CARTE CARLA (waypoints) — vérité terrain de voie.

Sur CARLA, la détection de voies par IMAGE échoue : les réseaux (YOLOPv2/UFLD) sont
entraînés sur de vraies vidéos et ne lisent pas le rendu synthétique. On utilise donc
la carte du simulateur, qui connaît EXACTEMENT les voies.

Ce module implémente la même interface `LaneEstimator` (estimate/assign) : il construit
le corridor de l'ego en projetant les bords de sa voie (waypoints) dans l'image, puis
délègue l'assignation de zone à la géométrie classique (désormais fiable, car le corridor
est exact). Il expose en plus, pour le futur contrôleur (étape C) : virage à venir,
voies adjacentes disponibles, légalité du changement de voie.

Sur une vraie dashcam on garderait la détection image ; ici on a la vérité carte.
"""
from __future__ import annotations

from typing import Optional

from ..schemas import WorldState
from .lanes import (LaneContext, LaneEstimator, LaneLine, corridor_horizon,
                   lane_curvature)


class CarlaLaneProvider:
    """Corridor de voie de l'ego depuis les waypoints CARLA, projeté dans l'image.

    Args:
        source     : le CarlaSource actif (accès carte / ego / projection caméra).
        drive_side : sens de circulation (pour l'assignation des zones).
        ahead_m    : distance de voie échantillonnée devant l'ego (m).
        step_m     : pas d'échantillonnage le long de la voie (m).
    """

    def __init__(self, source, drive_side: str = "right",
                 ahead_m: float = 30.0, step_m: float = 3.0):
        import carla
        self._src = source
        self._map = source._world.get_map()
        self._carla = carla
        self._assigner = LaneEstimator(drive_side=drive_side)
        self._ahead_m = ahead_m
        self._step_m = step_m
        # Infos de conduite exposées au contrôleur (étape C), mises à jour chaque frame.
        self.ego_waypoint = None
        self.turn_ahead = ""            # "" | "gauche" | "droite"
        self.left_lane_available = False
        self.right_lane_available = False
        self.lane_change_allowed = ""   # str(carla.LaneChange) : None/Left/Right/Both

    # ------------------------------------------------------------------ #
    def estimate(self, frame) -> LaneContext:
        carla = self._carla
        h, w = frame.image.shape[:2]
        ego = self._src._ego
        wp = self._map.get_waypoint(ego.get_location(), project_to_road=True,
                                    lane_type=carla.LaneType.Driving)
        if wp is None:
            return LaneContext(False, h, w)
        self.ego_waypoint = wp
        self._update_driving_info(wp)

        # Échantillonne la ligne médiane : un peu en arrière (pour atteindre le bas de
        # l'image) puis devant en suivant la voie (virages inclus).
        samples = []
        prev = wp.previous(4.0)
        if prev:
            samples.append(prev[0])
        samples.append(wp)
        cur = wp
        for _ in range(int(self._ahead_m / self._step_m)):
            nxt = cur.next(self._step_m)
            if not nxt:
                break
            cur = nxt[0]
            samples.append(cur)

        lu, lv, ru, rv = [], [], [], []
        for s in samples:
            loc = s.transform.location
            r = s.transform.get_right_vector()      # direction "droite" de la voie
            half = s.lane_width / 2.0
            left = carla.Location(loc.x - r.x * half, loc.y - r.y * half, loc.z - r.z * half)
            right = carla.Location(loc.x + r.x * half, loc.y + r.y * half, loc.z + r.z * half)
            pl = self._src.project_location(left)
            pr = self._src.project_location(right)
            if pl is not None:
                lu.append(pl[0]); lv.append(pl[1])
            if pr is not None:
                ru.append(pr[0]); rv.append(pr[1])

        if len(lu) < 2 or len(ru) < 2:
            return LaneContext(False, h, w)
        left_line = LaneLine.fit(lv, lu, degree=2, y_scale=h)
        right_line = LaneLine.fit(rv, ru, degree=2, y_scale=h)
        if left_line is None or right_line is None:
            return LaneContext(False, h, w)

        # Garantit gauche < droite dans l'image (au cas où la voie serait orientée).
        if left_line.x_at(h - 1) > right_line.x_at(h - 1):
            left_line, right_line = right_line, left_line

        xl_b, xr_b = left_line.x_at(h - 1), right_line.x_at(h - 1)
        lane_w = max(1.0, xr_b - xl_b)
        r_m, r_dir = lane_curvature(left_line, right_line, h, lane_w)
        ctx = LaneContext(True, h, w, left_line, right_line,
                          center_x_bottom=(xl_b + xr_b) / 2.0,
                          lane_width_px=lane_w,
                          y_top=min(left_line.y_lo, right_line.y_lo),
                          horizon_y=corridor_horizon(left_line, right_line, h) or 0.0,
                          corridor_confidence=1.0,
                          curvature_radius_m=r_m,
                          curvature_dir=r_dir or self.turn_ahead)
        # Rasterise le corridor pour le dessin : masque roulable (vert) + bords (rouge).
        self._paint_masks(ctx, lu, lv, ru, rv, h, w)
        return ctx

    @staticmethod
    def _paint_masks(ctx, lu, lv, ru, rv, h: int, w: int) -> None:
        import cv2
        import numpy as np
        left_pts = sorted(zip(lu, lv), key=lambda p: p[1])   # haut -> bas (v croissant)
        right_pts = sorted(zip(ru, rv), key=lambda p: p[1])
        if len(left_pts) < 2 or len(right_pts) < 2:
            return
        ring = np.array([(int(u), int(v)) for u, v in left_pts]
                        + [(int(u), int(v)) for u, v in reversed(right_pts)], dtype=np.int32)
        drivable = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(drivable, [ring], 1)
        lanes = np.zeros((h, w), dtype=np.uint8)
        cv2.polylines(lanes, [np.array([(int(u), int(v)) for u, v in left_pts], np.int32)],
                      False, 1, thickness=max(3, w // 240))
        cv2.polylines(lanes, [np.array([(int(u), int(v)) for u, v in right_pts], np.int32)],
                      False, 1, thickness=max(3, w // 240))
        ctx.drivable_mask = drivable.astype(bool)
        ctx.lane_mask = lanes.astype(bool)

    def assign(self, world: WorldState, ctx: LaneContext) -> None:
        # Corridor exact -> l'assignation géométrique classique devient fiable.
        self._assigner.assign(world, ctx)

    # ------------------------------------------------------------------ #
    def _update_driving_info(self, wp) -> None:
        """Renseigne virage / voies dispo / changement autorisé pour le contrôleur."""
        carla = self._carla
        left = wp.get_left_lane()
        right = wp.get_right_lane()
        self.left_lane_available = (left is not None
                                    and left.lane_type == carla.LaneType.Driving)
        self.right_lane_available = (right is not None
                                     and right.lane_type == carla.LaneType.Driving)
        self.lane_change_allowed = str(wp.lane_change)
        nxt = wp.next(10.0)
        if nxt:
            dyaw = (nxt[0].transform.rotation.yaw - wp.transform.rotation.yaw + 540) % 360 - 180
            self.turn_ahead = "gauche" if dyaw < -8 else "droite" if dyaw > 8 else ""
        else:
            self.turn_ahead = ""
