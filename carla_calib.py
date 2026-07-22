"""carla_calib — calibration de la distance perçue par SCÉNARIO CONTRÔLÉ.

Ego immobile ; une voiture-cible est placée EXACTEMENT à des distances connues devant
(via les waypoints de la voie). À chaque palier on mesure la distance ESTIMÉE par la
perception vs la distance RÉELLE (projection caméra) pour CE véhicule. On en tire le
biais/échelle du modèle de distance -> correction de --fov / --cam-height.

    .venv-carla\\Scripts\\python.exe carla_calib.py
"""
from __future__ import annotations

import os
import statistics as st
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import carla

from aegisdrive.io.carla_source import CarlaSource
from aegisdrive.perception.yolo_detector import YoloDetector
from aegisdrive.tracking.kalman_tracker import KalmanTracker
from aegisdrive.estimation.kinematics import KinematicsEstimator
from aegisdrive.schemas import WorldState

FOV = 60.0
CAM_H = 1.2
W, H = 960, 540
TARGETS = [10, 13, 16, 20, 25, 30, 35, 40, 50]     # distances réelles voulues (m)
SETTLE = 8                                          # frames de stabilisation par palier


def _match_est(tracks, uv):
    """Distance estimée du track dont la boîte contient la projection de la cible."""
    u, v = uv
    cont = [t for t in tracks if t.distance_m is not None
            and t.bbox.x1 <= u <= t.bbox.x2 and t.bbox.y1 <= v <= t.bbox.y2]
    if not cont:
        return None
    return min(cont, key=lambda x: x.bbox.area).distance_m


def main() -> int:
    src = CarlaSource(fps=20.0, width=W, height=H, fov=FOV, cam_height=CAM_H,
                      autopilot=False, n_traffic=0, spectator_follow=True,
                      max_frames=len(TARGETS) * SETTLE + 5, seed=1)
    ego = src._ego
    ego.set_simulate_physics(False)                 # ego figé -> caméra stable
    cmap = src._world.get_map()

    bl = src._world.get_blueprint_library()
    # une berline nette (bien classée "car" par YOLO), pas un bus/camion.
    lead_bp = None
    for name in ("vehicle.tesla.model3", "vehicle.audi.tt", "vehicle.nissan.micra"):
        f = bl.filter(name)
        if f:
            lead_bp = f[0]
            break
    if lead_bp is None:
        lead_bp = bl.filter("vehicle.*")[0]
    lead = None                                     # apparu au 1er placement

    det = YoloDetector()
    trk = KalmanTracker()
    est = KinematicsEstimator(image_width_px=W, image_height_px=H,
                              fov_deg=FOV, cam_height_m=CAM_H)

    results = []                                     # (cible_m, reel_m, estime_m)
    for frame in src.frames():
        ti = frame.index // SETTLE
        if ti >= len(TARGETS):
            break
        phase = frame.index % SETTLE

        if phase == 0:                              # place la cible pour ce palier
            d = float(TARGETS[ti])
            ego_tf = ego.get_transform()            # LU dans la boucle (après tick -> valide)
            fwd = ego_tf.get_forward_vector()       # sens où l'ego REGARDE
            loc = carla.Location(ego_tf.location.x + fwd.x * d,
                                 ego_tf.location.y + fwd.y * d,
                                 ego_tf.location.z + fwd.z * d)
            wp = cmap.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)
            tf = carla.Transform(carla.Location(wp.transform.location.x,
                                                wp.transform.location.y,
                                                wp.transform.location.z + 0.3),
                                 ego_tf.rotation)
            if lead is None:
                lead = src._world.try_spawn_actor(lead_bp, tf)
                print(f"  [palier {TARGETS[ti]}m] spawn lead: "
                      f"{'OK id='+str(lead.id) if lead else 'ÉCHEC'}", flush=True)
                if lead is not None:
                    lead.set_simulate_physics(False)
            else:
                lead.set_transform(tf)

        # Perception à CHAQUE frame -> le tracker Kalman garde sa continuité (sinon
        # aucun track confirmé, donc pas de distance estimée).
        dets = det.detect(frame)
        tracks = trk.update(frame, dets)
        world = WorldState(frame.index, frame.timestamp, tracks)
        est.update(world)

        if phase == SETTLE - 1:                        # mesure en fin de palier (track stable)
            if lead is None:
                print(f"  cible {TARGETS[ti]:2d}m : pas de lead (spawn échoué)", flush=True)
                continue
            proj = src.project_location(lead.get_location())
            if proj is None:
                print(f"  cible {TARGETS[ti]:2d}m : projection None (hors champ)", flush=True)
                continue
            u, v, depth = proj
            e = _match_est(tracks, (u, v))
            results.append((TARGETS[ti], depth, e))
            tag = f"{e:.1f}m" if e is not None else f"NON APPARIÉ ({len(tracks)} tracks, uv=({u:.0f},{v:.0f}))"
            print(f"  cible {TARGETS[ti]:2d}m  reel(proj)={depth:5.1f}m  estimé={tag}", flush=True)

    # ---- analyse ----
    good = [(r, e) for _, r, e in results if e is not None]
    print("\n================ CALIBRATION DISTANCE ================")
    if good:
        ratios = [e / r for r, e in good]
        med = st.median(ratios)
        print(f"  paliers mesurés : {len(good)}/{len(TARGETS)}")
        print(f"  ratio estimé/réel : médian={med:.2f}  "
              f"(min={min(ratios):.2f} max={max(ratios):.2f})")
        if med > 1.05:
            print(f"  -> la perception SURESTIME (~×{med:.2f}). Correction : ÷{med:.2f}.")
        elif med < 0.95:
            print(f"  -> la perception SOUS-ESTIME (~×{med:.2f}). Correction : ×{1/med:.2f}.")
        else:
            print("  -> distance globalement JUSTE (biais < 5%).")
        print(f"  (le modèle pinhole est ~linéaire : un seul facteur d'échelle corrige)")
    else:
        print("  aucune cible détectée — vérifier le placement.")
    print("=====================================================")

    if lead is not None:
        try:
            lead.destroy()
        except Exception:
            pass
    import subprocess
    try:
        subprocess.run([sys.executable, "carla_reset.py", "--quiet"], timeout=40)
    except Exception:
        pass
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
