"""carla_calib — vérification RIGOUREUSE de la distance perçue (scénario contrôlé).

Ego immobile ; une voiture-cible est placée EXACTEMENT à des distances connues devant
(vecteur avant de l'ego). À chaque palier, on laisse le tracker se verrouiller puis on
MOYENNE l'estimation sur plusieurs frames (bruit réduit). On compare à la distance
RÉELLE (projection caméra) pour CE véhicule.

Sortie : table par distance (réel, estimé moyen, erreur %, classe détectée) + synthèse
(biais, |erreur| médiane, dispersion). Prérequis d'un contrôle fiable (étape C).

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
TARGETS = [8, 10, 12, 15, 18, 22, 26, 30, 35, 40, 45, 50]  # distances réelles (m)
SETTLE = 16                 # frames par palier (téléportation + verrouillage tracker)
MEASURE_LAST = 6            # on moyenne la mesure sur les N dernières frames du palier


def _match(tracks, uv):
    """(distance_estimée, classe) du track dont la boîte contient la projection cible."""
    u, v = uv
    cont = [t for t in tracks if t.distance_m is not None
            and t.bbox.x1 <= u <= t.bbox.x2 and t.bbox.y1 <= v <= t.bbox.y2]
    if not cont:
        return None
    t = min(cont, key=lambda x: x.bbox.area)
    return t.distance_m, t.cls.value


def main() -> int:
    src = CarlaSource(fps=20.0, width=W, height=H, fov=FOV, cam_height=CAM_H,
                      autopilot=False, n_traffic=0, spectator_follow=False,
                      max_frames=len(TARGETS) * SETTLE + 5, seed=1)
    ego = src._ego
    ego.set_simulate_physics(False)
    cmap = src._world.get_map()

    bl = src._world.get_blueprint_library()
    lead_bp = None
    for name in ("vehicle.tesla.model3", "vehicle.audi.tt", "vehicle.nissan.micra"):
        f = bl.filter(name)
        if f:
            lead_bp = f[0]
            break
    lead_bp = lead_bp or bl.filter("vehicle.*")[0]

    det = YoloDetector()
    trk = KalmanTracker()
    est = KinematicsEstimator(image_width_px=W, image_height_px=H,
                              fov_deg=FOV, cam_height_m=CAM_H)

    lead = None
    samples: dict[int, list] = {d: [] for d in TARGETS}   # cible -> [(est, depth, cls), ...]

    for frame in src.frames():
        ti = frame.index // SETTLE
        if ti >= len(TARGETS):
            break
        phase = frame.index % SETTLE
        target = TARGETS[ti]

        if phase == 0:                              # (re)place la cible pour ce palier
            etf = ego.get_transform()
            fwd = etf.get_forward_vector()
            loc = carla.Location(etf.location.x + fwd.x * target,
                                 etf.location.y + fwd.y * target,
                                 etf.location.z + fwd.z * target)
            wp = cmap.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)
            tf = carla.Transform(carla.Location(wp.transform.location.x,
                                                wp.transform.location.y,
                                                wp.transform.location.z + 0.3),
                                 etf.rotation)
            if lead is None:
                lead = src._world.try_spawn_actor(lead_bp, tf)
                if lead is not None:
                    lead.set_simulate_physics(False)
            elif lead is not None:
                lead.set_transform(tf)

        # Perception à CHAQUE frame -> continuité du tracker.
        dets = det.detect(frame)
        tracks = trk.update(frame, dets)
        world = WorldState(frame.index, frame.timestamp, tracks)
        est.update(world)

        # Mesure moyennée sur les dernières frames du palier (tracker stabilisé).
        if lead is not None and phase >= SETTLE - MEASURE_LAST:
            proj = src.project_location(lead.get_location())
            if proj is not None:
                u, v, depth = proj
                m = _match(tracks, (u, v))
                if m is not None:
                    samples[target].append((m[0], depth, m[1]))

    # ---------------- table + synthèse ----------------
    print("\n==================== VÉRIFICATION DISTANCE ====================")
    print(f"  {'réel(m)':>8} {'estimé(m)':>10} {'erreur':>8} {'err%':>7} {'classe':>10} {'n':>3}")
    all_ratios, all_abs = [], []
    for d in TARGETS:
        s = samples[d]
        if not s:
            print(f"  {d:8.1f} {'—':>10} {'non détecté':>17}")
            continue
        ests = [e for e, _, _ in s]
        depth = st.median([dp for _, dp, _ in s])
        cls = max(set(c for _, _, c in s), key=[c for _, _, c in s].count)
        em = st.median(ests)
        err = em - depth
        ratio = em / depth
        all_ratios.append(ratio)
        all_abs.append(abs(err))
        print(f"  {depth:8.1f} {em:10.1f} {err:+8.1f} {100*(ratio-1):+6.0f}% {cls:>10} {len(s):3d}")
    print("  " + "-" * 58)
    if all_ratios:
        med_r = st.median(all_ratios)
        print(f"  ratio estimé/réel : médian {med_r:.3f}  "
              f"(min {min(all_ratios):.2f}, max {max(all_ratios):.2f})")
        print(f"  |erreur| médiane : {st.median(all_abs):.1f} m")
        verdict = ("JUSTE (biais <5%)" if 0.95 <= med_r <= 1.05
                   else f"BIAIS : correction ×{1/med_r:.3f} (fov ou cam-height)")
        print(f"  verdict : {verdict}")
    print("===============================================================")

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
