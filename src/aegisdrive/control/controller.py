"""Contrôleur de conduite (étape C) — pilote l'ego dans CARLA.

Fusion de capteurs (comme les vraies voitures autonomes) :
  - CAMÉRA + YOLO (AegisDrive) : détecte/classe la scène (perception).
  - RADAR VIRTUEL (vérité terrain CARLA) : distance/vitesse EXACTES de ce qui est devant.
  - CARTE (waypoints) : où va la voie, virages, voies adjacentes.

Produit à chaque tick une commande `throttle / brake / steer`. Trois couches :
  1. Suivi de voie (pure-pursuit) + régulation de vitesse/ freinage (ACC).
  2. Virages : au carrefour, choisit la branche qui suit la route.
  3. Changements de voie : si un lent bloque devant et qu'une voie adjacente est libre.

La distance des décisions vient du radar virtuel (précis), PAS de la distance monoculaire
(±15 %). Voir la discussion de calibration.
"""
from __future__ import annotations

import math

import carla


def _yaw_diff(a: float, b: float) -> float:
    """Différence d'angle (deg) ramenée dans [-180, 180]."""
    return (a - b + 540.0) % 360.0 - 180.0


class Controller:
    def __init__(self, source, lane_provider,
                 target_kmh: float = 30.0, time_gap_s: float = 1.8,
                 min_gap_m: float = 6.0, wheelbase_m: float = 2.8):
        self._src = source
        self._ego = source._ego
        self._world = source._world
        self._map = self._world.get_map()
        self._lane = lane_provider
        self._target = target_kmh / 3.6          # m/s
        self._time_gap = time_gap_s
        self._min_gap = min_gap_m
        self._L = wheelbase_m
        # état changement de voie
        self._state = "SUIVI"                    # "SUIVI" | "CHANGEMENT"
        self._change_side = ""                   # "left" | "right"
        self._change_t = 0.0
        # anti-blocage : sans ça, un obstacle immobile fige l'ego indéfiniment
        self._stuck = 0                          # frames consécutives à l'arrêt
        self._recover = 0                        # frames de marche arrière restantes
        # --- capteur d'obstacle CARLA : voit TOUTE la géométrie (murs, poteaux, décors),
        # pas seulement les véhicules. Sans lui, l'ego fonçait dans les décors statiques.
        bl = self._world.get_blueprint_library()
        obp = bl.find("sensor.other.obstacle")
        obp.set_attribute("distance", "35")
        obp.set_attribute("hit_radius", "0.5")
        obp.set_attribute("only_dynamics", "False")     # statique inclus
        self._obs_sensor = self._world.spawn_actor(
            obp, carla.Transform(carla.Location(x=2.2, z=0.9)), attach_to=self._ego)
        self._obs_d = None
        self._obs_t = -1e9
        self._obs_sensor.listen(self._on_obstacle)

    def _on_obstacle(self, ev) -> None:
        self._obs_d = float(ev.distance)
        self._obs_t = float(ev.timestamp)
        # télémétrie
        self.action = ""                         # texte pour l'affichage
        self.lead_dist = None

    # ------------------------------------------------------------------ #
    def _speed(self) -> float:
        """Vitesse longitudinale de l'ego (m/s, le long de son cap)."""
        v = self._ego.get_velocity()
        f = self._ego.get_transform().get_forward_vector()
        return v.x * f.x + v.y * f.y + v.z * f.z

    def _radar_lead(self, max_range: float = 45.0):
        """RADAR VIRTUEL : (distance, vitesse) de l'obstacle le plus proche DEVANT, dans
        la voie de l'ego. Distance exacte (vérité terrain CARLA). None si voie libre."""
        ego_tf = self._ego.get_transform()
        loc0 = ego_tf.location
        fwd = ego_tf.get_forward_vector()
        ego_wp = self._map.get_waypoint(loc0, project_to_road=True,
                                        lane_type=carla.LaneType.Driving)
        best_d, best_v = None, 0.0
        for a in self._world.get_actors():
            tid = a.type_id
            if not (tid.startswith("vehicle.") or tid.startswith("walker.")):
                continue
            if a.id == self._ego.id:
                continue
            loc = a.get_location()
            dx, dy = loc.x - loc0.x, loc.y - loc0.y
            forward = dx * fwd.x + dy * fwd.y             # distance devant
            if forward <= 0.5 or forward > max_range:
                continue
            right = -dx * fwd.y + dy * fwd.x              # écart latéral
            # dans MA voie ? même lane carte, sinon tolérance latérale SERRÉE (1.2 m) :
            # trop large, on freinait pour des voitures garées ou d'en face.
            awp = self._map.get_waypoint(loc, project_to_road=True,
                                         lane_type=carla.LaneType.Driving)
            same_lane = (awp is not None and ego_wp is not None
                         and awp.road_id == ego_wp.road_id and awp.lane_id == ego_wp.lane_id)
            if not (same_lane or abs(right) < 1.2):
                continue
            if best_d is None or forward < best_d:
                best_d = forward
                av = a.get_velocity()
                best_v = av.x * fwd.x + av.y * fwd.y      # vitesse du lead le long de mon cap
        return best_d, best_v

    # ------------------------------------------------------------------ #
    def _straightest(self, cur, branches):
        """Couche 2 (virages) : au carrefour, garde la branche la plus DEVANT l'ego.

        On se base sur la géométrie réelle (projection dans le repère de l'ego) et non sur
        le cap du waypoint : ça reste valable qu'on remonte la voie par next() ou previous().
        """
        if len(branches) == 1:
            return branches[0]
        tf = self._ego.get_transform()
        loc0, fwd = tf.location, tf.get_forward_vector()

        def score(w):
            d = w.transform.location
            dx, dy = d.x - loc0.x, d.y - loc0.y
            forward = dx * fwd.x + dy * fwd.y
            right = -dx * fwd.y + dy * fwd.x
            return (0 if forward > 0 else 1, abs(right))   # devant d'abord, puis le + centré

        return min(branches, key=score)

    def _ego_lane_wp(self):
        """Waypoint de la voie de l'ego, ALIGNÉ avec son cap.

        `get_waypoint()` renvoie la voie la plus proche — parfois celle d'EN FACE. Suivre
        son `next()` ferait avancer à contresens et braquer n'importe comment. On cherche
        donc une voie orientée dans notre sens ; sinon on signale qu'il faut remonter la
        voie via `previous()`.
        """
        tf = self._ego.get_transform()
        wp = self._map.get_waypoint(tf.location, project_to_road=True,
                                    lane_type=carla.LaneType.Driving)
        if wp is None:
            return None, True
        if abs(_yaw_diff(wp.transform.rotation.yaw, tf.rotation.yaw)) < 90.0:
            return wp, True
        for cand in (wp.get_left_lane(), wp.get_right_lane()):
            if (cand is not None and cand.lane_type == carla.LaneType.Driving
                    and abs(_yaw_diff(cand.transform.rotation.yaw, tf.rotation.yaw)) < 90.0):
                return cand, True
        return wp, False                     # voie à contresens : on suivra previous()

    def _advance(self, wp, aligned: bool, dist: float):
        """Remonte la voie de `dist` mètres dans le sens de marche de l'ego."""
        cur = wp
        done = 0.0
        while done < dist:
            nxt = cur.next(2.0) if aligned else cur.previous(2.0)
            if not nxt:
                break
            cur = self._straightest(cur, nxt)
            done += 2.0
        return cur

    def _target_waypoint(self, lookahead: float):
        """Point de référence sur le chemin voulu (voie courante, ou voie adjacente
        pendant un changement de voie), `lookahead` m devant."""
        wp, aligned = self._ego_lane_wp()
        if wp is None:
            return None
        if self._state == "CHANGEMENT" and self._change_side:
            adj = wp.get_left_lane() if self._change_side == "left" else wp.get_right_lane()
            if adj is not None and adj.lane_type == carla.LaneType.Driving:
                wp = adj
        return self._advance(wp, aligned, lookahead)

    def _steer_to(self, ref_wp, speed: float) -> float:
        """Contrôleur de STANLEY : corrige À LA FOIS l'erreur de cap et l'ÉCART LATÉRAL
        à l'axe de la voie.

        Le pure-pursuit seul ne corrigeait pas l'écart à la ligne : l'ego oscillait puis
        débordait de sa voie (et finissait hors route). Stanley ramène activement le
        véhicule sur l'axe :  braquage = erreur_de_cap + atan(k · écart / vitesse).
        """
        if ref_wp is None:
            return 0.0
        tf = self._ego.get_transform()
        loc0, fwd = tf.location, tf.get_forward_vector()
        _, aligned = self._ego_lane_wp()

        # 1) erreur de CAP entre la voie (au point de référence) et l'ego
        lane_yaw = ref_wp.transform.rotation.yaw + (0.0 if aligned else 180.0)
        heading_err = math.radians(_yaw_diff(lane_yaw, tf.rotation.yaw))

        # 2) ÉCART LATÉRAL : où est l'axe de la voie par rapport à moi
        #    (+ = axe à ma droite => je suis à gauche => je dois braquer à droite)
        d = ref_wp.transform.location
        dx, dy = d.x - loc0.x, d.y - loc0.y
        cross = -dx * fwd.y + dy * fwd.x

        steer = heading_err + math.atan2(1.3 * cross, max(2.0, abs(speed)))
        return max(-1.0, min(1.0, steer / 0.7))            # 0.7 rad ≈ braquage max

    # ------------------------------------------------------------------ #
    def _maybe_lane_change(self, lead_d, lead_v, now: float) -> None:
        """Couche 3 : décide/entretient un changement de voie."""
        if self._state == "CHANGEMENT":
            ego_wp = self._map.get_waypoint(self._ego.get_location(),
                                            lane_type=carla.LaneType.Driving)
            # terminé quand on est dans la voie cible (ou après 4 s de sécurité)
            if now - self._change_t > 4.0:
                self._state, self._change_side = "SUIVI", ""
            return
        # déclenche seulement si un LENT bloque devant, à portée
        speed = self._speed()
        blocked = (lead_d is not None and lead_d < max(self._min_gap * 2.5, 18.0)
                   and lead_v < 0.6 * self._target)
        if not blocked:
            return
        allowed = str(self._lane.lane_change_allowed)     # None/Left/Right/Both
        # essaie gauche puis droite si dispo + autorisé + voie adjacente libre
        for side, avail in (("left", self._lane.left_lane_available),
                            ("right", self._lane.right_lane_available)):
            legal = allowed in ("Both",) or allowed.lower() == side
            if avail and legal and self._adjacent_clear(side):
                self._state, self._change_side, self._change_t = "CHANGEMENT", side, now
                return

    def _adjacent_clear(self, side: str, ahead: float = 25.0, behind: float = 10.0) -> bool:
        """Voie adjacente libre (pas d'acteur dans une fenêtre autour de l'ego) ?"""
        ego_tf = self._ego.get_transform()
        loc0, fwd = ego_tf.location, ego_tf.get_forward_vector()
        ego_wp = self._map.get_waypoint(loc0, lane_type=carla.LaneType.Driving)
        adj = ego_wp.get_left_lane() if side == "left" else ego_wp.get_right_lane()
        if adj is None or adj.lane_type != carla.LaneType.Driving:
            return False
        adj_id = adj.lane_id
        for a in self._world.get_actors():
            if not a.type_id.startswith(("vehicle.", "walker.")) or a.id == self._ego.id:
                continue
            loc = a.get_location()
            dx, dy = loc.x - loc0.x, loc.y - loc0.y
            forward = dx * fwd.x + dy * fwd.y
            if not (-behind < forward < ahead):
                continue
            awp = self._map.get_waypoint(loc, lane_type=carla.LaneType.Driving)
            if awp is not None and awp.lane_id == adj_id:
                return False
        return True

    # ------------------------------------------------------------------ #
    def compute(self, now: float) -> "carla.VehicleControl":
        speed = self._speed()
        lead_d, lead_v = self._radar_lead()
        # FUSION : le capteur d'obstacle voit aussi le statique. On garde le plus proche
        # des deux (un mur à 5 m prime sur une voiture à 20 m). Lecture périmée -> ignorée.
        if self._obs_d is not None and (now - self._obs_t) < 0.25:
            if lead_d is None or self._obs_d < lead_d:
                lead_d, lead_v = self._obs_d, 0.0      # obstacle statique : vitesse nulle
        self.lead_dist = lead_d

        # --- ANTI-BLOCAGE : sorti de voie / nez contre un obstacle immobile ---
        if self._recover > 0:                              # manœuvre de dégagement
            self._recover -= 1
            self.action = "DÉGAGEMENT"
            rec = carla.VehicleControl()
            rec.throttle, rec.reverse, rec.steer, rec.brake = 0.4, True, 0.25, 0.0
            return rec
        if abs(speed) < 0.3 and lead_d is not None and lead_d < self._min_gap:
            self._stuck += 1
        else:
            self._stuck = 0
        if self._stuck > 60:                               # ~3 s figé -> on se dégage
            self._stuck, self._recover = 0, 30
            self._state, self._change_side = "SUIVI", ""

        self._maybe_lane_change(lead_d, lead_v, now)

        # --- LATÉRAL : Stanley sur un point de référence PROCHE (colle à la ligne).
        # Un point trop lointain faisait couper les virages -> sortie de voie.
        lookahead = max(3.0, min(8.0, 2.0 + 0.35 * speed))
        target_wp = self._target_waypoint(lookahead)
        steer = self._steer_to(target_wp, speed)

        # --- LONGITUDINAL : vitesse cible, réduite par le lead et les virages ---
        # Réduction PROGRESSIVE en virage (plancher 4 m/s) : un plafond brutal faisait
        # ramper la voiture au lieu de négocier le virage.
        target = min(self._target,
                     max(4.0, self._target * (1.0 - 0.7 * min(1.0, abs(steer)))))
        # Écarté de l'axe de ma voie -> je ralentis pour me recaler (évite de partir
        # hors route et de finir coincé contre un obstacle).
        ego_wp_now = self._map.get_waypoint(self._ego.get_location(), project_to_road=True,
                                            lane_type=carla.LaneType.Driving)
        if ego_wp_now is not None:
            el = self._ego.get_location()
            cl = ego_wp_now.transform.location
            off = math.hypot(el.x - cl.x, el.y - cl.y)
            if off > 1.5:
                target = min(target, 4.0)
        if lead_d is not None:
            desired_gap = max(self._min_gap, self._time_gap * speed)
            if lead_d < desired_gap:                       # ACC : se cale sur le lead
                target = max(0.0, min(target, lead_v * (lead_d / desired_gap)))

        # commande throttle/brake depuis l'erreur de vitesse
        ctrl = carla.VehicleControl()
        ctrl.steer = float(steer)
        err = target - speed
        # freinage d'urgence : très proche ou TTC court
        ttc = (lead_d / max(0.1, speed - lead_v)) if (lead_d is not None and speed > lead_v) else 1e9
        if (lead_d is not None and lead_d < self._min_gap * 0.9) or ttc < 1.6:
            ctrl.throttle, ctrl.brake = 0.0, 1.0
            self.action = "FREINE"
        elif err > 0:
            thr = min(0.7, err * 0.5)
            if speed < 1.0:
                thr = max(thr, 0.35)                       # anti-calage au démarrage/virage
            ctrl.throttle, ctrl.brake = thr, 0.0
            self.action = ("VIRAGE" if abs(steer) > 0.35 else
                           ("CHANGEMENT " + self._change_side if self._state == "CHANGEMENT"
                            else "ROULE"))
        else:
            ctrl.throttle, ctrl.brake = 0.0, min(1.0, -err * 0.4)
            self.action = "RALENTIT"
        return ctrl
