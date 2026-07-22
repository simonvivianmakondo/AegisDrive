"""Source CARLA — fournit des frames depuis une caméra virtuelle du simulateur.

Implémente la MÊME interface `Source` que `VideoFileSource` : le reste du pipeline
AegisDrive (détection, tracking, risque…) ne voit aucune différence entre une vidéo
et le monde simulé. On remplace juste la source de frames.

Bonus décisif : CARLA connaît la **vérité terrain** (position/vitesse réelles de
chaque acteur). `ground_truth()` l'expose — base de l'étape B (validation de la
perception) puis de la récompense RL (étape D).

Prérequis :
  - paquet client `carla` (livré DANS le zip CARLA 0.10.0 ; version client == serveur) ;
  - un serveur CARLA lancé (CarlaUnreal.exe) et joignable sur host:port.

────────────────────────────────────────────────────────────────────────────────
Note de robustesse (CARLA 0.10.0 / Windows) — IMPORTANT
────────────────────────────────────────────────────────────────────────────────
La librairie native `carla` **plante à la destruction** (fast-fail 0xC0000409) dès
qu'on rappelle le serveur APRÈS avoir arrêté le rythme de ticks en mode synchrone
(stop capteur / apply_settings / destroy). Ce crash natif n'est PAS rattrapable en
Python (ce n'est pas une exception) et n'affecte pas les données déjà capturées.

Parade adoptée ici, la seule fiable :
  • `close()` ne fait AUCUN appel RPC (il ne peut pas nettoyer sans crasher) ;
  • le nettoyage (purge des acteurs + retour en async) se fait dans un PROCESS FRAIS
    → `carla_reset.py`, appelé automatiquement par les entrypoints avant de sortir ;
  • à l'init, on purge défensivement les orphelins d'un run précédent et on repart
    d'un monde propre.
Un programme qui utilise CarlaSource doit donc terminer par `os._exit(0)`.
"""
from __future__ import annotations

import random
from queue import Queue, Empty
from typing import Iterator, Optional

import numpy as np

from ..schemas import Frame

try:
    import carla  # livré avec CARLA 0.10.0 (PythonAPI/carla/dist/*.whl)
except ImportError:  # pragma: no cover - dépend de l'install CARLA
    carla = None

_MOBILE = ("vehicle.", "walker.", "sensor.")


class CarlaSource:
    """Caméra RGB embarquée sur un véhicule ego dans CARLA, exposée en `Source`.

    Args:
        host / port : serveur CARLA (défaut localhost:2000).
        fps         : pas de simulation (frames/s). 20 = compromis classique.
        width/height: résolution de la caméra (= résolution de traitement).
        fov         : champ de vision horizontal (°). À REPORTER dans `--fov` du
                      pipeline pour que le calcul de distance pinhole soit exact.
        cam_height / cam_forward : position caméra sur l'ego (m), style dashcam.
        autopilot   : laisse le Traffic Manager conduire l'ego (scène vivante).
        n_traffic   : nb de véhicules PNJ à faire apparaître autour.
        max_frames  : arrêt après N frames (None = infini).
        town        : charge une carte précise (ex. "Town10HD"). None = carte courante.
        seed        : graine pour un trafic/spawn reproductible.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 2000,
        fps: float = 20.0,
        width: int = 1280,
        height: int = 720,
        fov: float = 90.0,
        cam_height: float = 1.2,
        cam_forward: float = 2.3,   # devant le pare-brise -> pas de toit/capot dans le champ
        autopilot: bool = True,
        n_traffic: int = 30,
        max_frames: Optional[int] = None,
        town: Optional[str] = None,
        seed: int = 0,
        timeout_s: float = 30.0,
        spectator_follow: bool = True,
    ):
        if carla is None:
            raise ImportError(
                "Le paquet client `carla` est introuvable. Installe le wheel livré "
                "dans le zip CARLA 0.10.0 :  pip install "
                "<CARLA>/PythonAPI/carla/dist/carla-0.10.0-cp312-*.whl"
            )
        self._fps = float(fps)
        self._w, self._h = int(width), int(height)
        self._fov = float(fov)
        self._max_frames = max_frames
        self._spectator_follow = spectator_follow
        self._rng = random.Random(seed)

        self._client = carla.Client(host, port)
        self._client.set_timeout(timeout_s)
        self._world = self._client.load_world(town) if town else self._client.get_world()

        # --- départ propre : purge des orphelins d'un run précédent + async garanti ---
        # (sûr ici : aucun capteur en écoute, aucun tick synchrone encore lancé.)
        self._reset_world_async()
        self._purge_actors()

        self._tm = self._client.get_trafficmanager()
        self._tm.set_synchronous_mode(True)
        self._tm.set_random_device_seed(seed)

        # --- mode synchrone : 1 tick = 1 pas fixe = 1 image ---
        settings = self._world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / self._fps
        self._world.apply_settings(settings)

        bp_lib = self._world.get_blueprint_library()
        spawn_points = self._world.get_map().get_spawn_points()
        if not spawn_points:
            raise RuntimeError("Aucun point d'apparition sur cette carte CARLA.")

        # --- véhicule ego : spawn ROBUSTE (essaie plusieurs points -> pas de collision) ---
        ego_bp = bp_lib.filter("vehicle.*")[0]
        pts = spawn_points[:]
        self._rng.shuffle(pts)
        self._ego = None
        for p in pts:
            self._ego = self._world.try_spawn_actor(ego_bp, p)
            if self._ego is not None:
                break
        if self._ego is None:
            raise RuntimeError("Impossible de faire apparaître l'ego (tous les points occupés).")
        if autopilot:
            self._ego.set_autopilot(True, self._tm.get_port())

        # --- caméra RGB façon dashcam, attachée à l'ego ---
        cam_bp = bp_lib.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(self._w))
        cam_bp.set_attribute("image_size_y", str(self._h))
        cam_bp.set_attribute("fov", str(self._fov))
        cam_bp.set_attribute("sensor_tick", str(1.0 / self._fps))
        cam_tf = carla.Transform(carla.Location(x=cam_forward, z=cam_height), carla.Rotation())
        self._camera = self._world.spawn_actor(cam_bp, cam_tf, attach_to=self._ego)

        # file image (maxsize 1 : on ne garde que la dernière, jamais de retard)
        self._queue: "Queue" = Queue(maxsize=1)
        self._camera.listen(self._on_image)

        # --- trafic PNJ pour peupler la scène ---
        self._traffic: list = []
        self._spawn_traffic(bp_lib, [p for p in pts if p is not None], n_traffic)

        # --- spectateur : la vue de la FENÊTRE serveur suit l'ego (chase-cam) ---
        self._spectator = self._world.get_spectator() if spectator_follow else None

        self._closed = False

    # ------------------------------------------------------------------ #
    def _reset_world_async(self) -> None:
        s = self._world.get_settings()
        s.synchronous_mode = False
        s.fixed_delta_seconds = None
        self._world.apply_settings(s)

    def _purge_actors(self) -> None:
        for a in self._world.get_actors():
            if a.type_id.startswith(_MOBILE):
                try:
                    a.destroy()
                except Exception:
                    pass

    def _on_image(self, image) -> None:
        """Callback capteur : convertit l'image CARLA (BGRA) en BGR (conv. OpenCV)."""
        buf = np.frombuffer(image.raw_data, dtype=np.uint8)
        bgra = buf.reshape((image.height, image.width, 4))
        bgr = bgra[:, :, :3].copy()          # drop alpha, contigu pour OpenCV
        if self._queue.full():
            try:
                self._queue.get_nowait()     # jette l'ancienne, garde la plus récente
            except Empty:
                pass
        self._queue.put(bgr)

    def _spawn_traffic(self, bp_lib, points, n: int) -> None:
        vehicles = bp_lib.filter("vehicle.*")
        for tf in points[: max(0, n)]:
            bp = vehicles[self._rng.randrange(len(vehicles))]
            actor = self._world.try_spawn_actor(bp, tf)
            if actor is not None:
                actor.set_autopilot(True, self._tm.get_port())
                self._traffic.append(actor)

    # ------------------------------------------------------------------ #
    @property
    def fps(self) -> float:
        return self._fps

    @property
    def size(self) -> tuple[int, int]:
        return self._w, self._h

    @property
    def fov(self) -> float:
        """FOV horizontal (°) — à passer au pipeline pour un calcul de distance exact."""
        return self._fov

    def frames(self) -> Iterator[Frame]:
        # Pas de `finally: close()` : tout appel RPC après l'arrêt des ticks synchrones
        # crashe la lib native. Le consommateur sort par os._exit(0) et délègue le
        # nettoyage à carla_reset.py (process frais). Voir l'entête du module.
        idx = 0
        while not self._closed:
            if self._max_frames is not None and idx >= self._max_frames:
                break
            self._world.tick()                       # avance d'un pas -> 1 image
            if self._spectator is not None:
                self._update_spectator()             # la fenêtre serveur suit l'ego
            try:
                img = self._queue.get(timeout=2.0)   # attend l'image de ce tick
            except Empty:
                continue                             # capteur en retard : on re-tick
            yield Frame(index=idx, timestamp=idx / self._fps, image=img)
            idx += 1

    def _update_spectator(self) -> None:
        """Place la caméra spectateur du serveur derrière/au-dessus de l'ego (chase-cam)."""
        import math
        tf = self._ego.get_transform()
        yaw = math.radians(tf.rotation.yaw)
        loc = carla.Location(
            x=tf.location.x - 6.0 * math.cos(yaw),
            y=tf.location.y - 6.0 * math.sin(yaw),
            z=tf.location.z + 3.0,
        )
        self._spectator.set_transform(
            carla.Transform(loc, carla.Rotation(pitch=-12.0, yaw=tf.rotation.yaw))
        )

    def ground_truth(self) -> list[dict]:
        """Vérité terrain de l'instant courant, dans le repère de l'ego.

        Retourne pour chaque véhicule/piéton (hors ego) : distance réelle (m) et
        vitesse (m/s). C'est l'étalon-or pour valider distance/TTC (étape B) et
        construire la récompense RL (étape D) — introuvable avec une simple dashcam.
        """
        import math
        ego_tf = self._ego.get_transform()
        ego_loc = ego_tf.location
        yaw = math.radians(ego_tf.rotation.yaw)
        fx, fy = math.cos(yaw), math.sin(yaw)     # vecteur "avant" de l'ego (plan sol)
        out: list[dict] = []
        for actor in self._world.get_actors():
            tid = actor.type_id
            if not (tid.startswith("vehicle.") or tid.startswith("walker.")):
                continue
            if actor.id == self._ego.id:
                continue
            loc = actor.get_location()
            dx, dy, dz = loc.x - ego_loc.x, loc.y - ego_loc.y, loc.z - ego_loc.z
            dist = float((dx * dx + dy * dy + dz * dz) ** 0.5)
            forward = dx * fx + dy * fy            # >0 : devant l'ego
            right = -dx * fy + dy * fx             # composante latérale
            bearing = math.degrees(math.atan2(right, forward))  # 0 = pile devant
            vel = actor.get_velocity()
            speed = float((vel.x**2 + vel.y**2 + vel.z**2) ** 0.5)
            out.append({
                "id": actor.id,
                "type": "pedestrian" if tid.startswith("walker.") else "vehicle",
                "distance_m": dist,
                "speed_mps": speed,
                "ahead": forward > 0.0,            # devant l'ego ?
                "bearing_deg": bearing,            # angle p/r à l'axe de l'ego
            })
        return out

    def project_location(self, loc) -> tuple[float, float, float] | None:
        """Projette un point monde (carla.Location) dans l'image : (u, v, profondeur_m).

        Renvoie None si le point est derrière la caméra. Réutilise la géométrie de
        `visible_gt` — sert au fournisseur de voies carte (projection des bords de voie).
        """
        import math
        import numpy as np
        w, h = self._w, self._h
        f = w / (2.0 * math.tan(math.radians(self._fov) / 2.0))
        K = np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1]])
        world_2_cam = np.array(self._camera.get_transform().get_inverse_matrix())
        pc = world_2_cam @ np.array([loc.x, loc.y, loc.z, 1.0])
        pc = np.array([pc[1], -pc[2], pc[0]])
        depth = float(pc[2])
        if depth <= 0.1:
            return None
        uvw = K @ pc
        return float(uvw[0] / uvw[2]), float(uvw[1] / uvw[2]), depth

    def visible_gt(self) -> list[dict]:
        """Vérité terrain PROJETÉE dans l'image caméra (appariement rigoureux).

        Pour chaque acteur (hors ego) réellement dans le champ, renvoie sa position pixel
        (u,v), sa profondeur réelle (m, le long de l'axe optique) et son type. Permet de
        rattacher chaque détection à son objet réel et de mesurer l'erreur de distance
        sur le MÊME objet (étape B) — méthode de projection standard CARLA.
        """
        import math
        import numpy as np
        w, h = self._w, self._h
        f = w / (2.0 * math.tan(math.radians(self._fov) / 2.0))
        K = np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1]])
        world_2_cam = np.array(self._camera.get_transform().get_inverse_matrix())
        out: list[dict] = []
        for actor in self._world.get_actors():
            tid = actor.type_id
            if not (tid.startswith("vehicle.") or tid.startswith("walker.")):
                continue
            if actor.id == self._ego.id:
                continue
            loc = actor.get_location()
            pw = np.array([loc.x, loc.y, loc.z, 1.0])
            pc = world_2_cam @ pw                        # repère caméra UE (x avant, y droite, z haut)
            pc = np.array([pc[1], -pc[2], pc[0]])        # -> repère image standard (x droite, y bas, z avant)
            depth = float(pc[2])
            if depth <= 0.1:                              # derrière la caméra
                continue
            uvw = K @ pc
            u, v = float(uvw[0] / uvw[2]), float(uvw[1] / uvw[2])
            if not (0 <= u < w and 0 <= v < h):           # hors cadre
                continue
            out.append({
                "id": actor.id,
                "type": "pedestrian" if tid.startswith("walker.") else "vehicle",
                "u": u, "v": v, "depth_m": depth,
            })
        return out

    def nearest_ahead_gt(self, fov_deg: float | None = None) -> float | None:
        """Distance réelle de l'objet le plus proche DEVANT l'ego, dans le champ caméra.

        Comparable à la distance estimée par la perception (qui ne voit que l'avant).
        Renvoie None si rien dans le cône. Sert à mesurer l'erreur de perception (étape B).
        """
        half = (fov_deg if fov_deg is not None else self._fov) / 2.0
        cands = [g["distance_m"] for g in self.ground_truth()
                 if g["ahead"] and abs(g["bearing_deg"]) <= half]
        return min(cands) if cands else None

    def close(self) -> None:
        """Marque la source fermée. NE FAIT AUCUN appel RPC (la lib native crasherait).

        Le vrai nettoyage — purge des acteurs + retour en mode asynchrone — est réalisé
        par `carla_reset.py` dans un process séparé. Terminer le programme par os._exit(0).
        """
        self._closed = True
