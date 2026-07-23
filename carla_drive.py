"""carla_drive — AegisDrive PILOTE l'ego dans CARLA (étape C).

Boucle fermée : perception (caméra+YOLO) + radar virtuel (vérité terrain) + carte
(waypoints) -> Controller -> throttle/brake/steer appliqués à chaque tick. L'autopilot
CARLA est COUPÉ : c'est notre code qui conduit (suivi de voie, virages, freinage,
changements de voie). Produit une vidéo annotée + télémétrie (collisions, vitesse…).

    .venv-carla\\Scripts\\python.exe carla_drive.py --frames 800 --output recordings/drive.mp4
"""
from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import carla
import cv2

from aegisdrive.io.carla_source import CarlaSource
from aegisdrive.io.video_sink import VideoSink
from aegisdrive.perception.yolo_detector import YoloDetector
from aegisdrive.tracking.kalman_tracker import KalmanTracker
from aegisdrive.estimation.kinematics import KinematicsEstimator
from aegisdrive.scene.carla_lanes import CarlaLaneProvider
from aegisdrive.understanding.object_state import StateEstimator
from aegisdrive.understanding.behavior import BehaviorEngine
from aegisdrive.risk.rule_engine import RuleRiskEngine
from aegisdrive.preprocess.conditions import SceneAnalyzer
from aegisdrive.control.controller import Controller
from aegisdrive.schemas import WorldState


def main() -> int:
    ap = argparse.ArgumentParser(description="AegisDrive pilote l'ego dans CARLA (étape C)")
    ap.add_argument("--frames", type=int, default=800)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--traffic", type=int, default=40)
    ap.add_argument("--target-kmh", type=float, default=30.0)
    ap.add_argument("--proc-width", type=int, default=960)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--output", default="recordings/drive.mp4")
    args = ap.parse_args()

    w = args.proc_width
    h = int(round(w * 9 / 16))
    src = CarlaSource(host=args.host, port=args.port, fps=20.0, width=w, height=h,
                      fov=60.0, cam_height=1.2, autopilot=False, n_traffic=args.traffic,
                      max_frames=args.frames, seed=args.seed, spectator_follow=True)
    ego = src._ego

    # capteur de collision (filet de sécurité + futur signal RL)
    col_bp = src._world.get_blueprint_library().find("sensor.other.collision")
    col = src._world.spawn_actor(col_bp, carla.Transform(), attach_to=ego)
    collisions = {"n": 0}
    col.listen(lambda e: collisions.__setitem__("n", collisions["n"] + 1))

    detector = YoloDetector()
    tracker = KalmanTracker()
    estimator = KinematicsEstimator(image_width_px=w, image_height_px=h,
                                    fov_deg=60.0, cam_height_m=1.2)
    lanes = CarlaLaneProvider(src, drive_side="right")
    state = StateEstimator()
    behavior = BehaviorEngine()
    risk = RuleRiskEngine(frame_size=src.size)
    analyzer = SceneAnalyzer(enhance=False)
    sink = VideoSink(args.output, fps=src.fps, size=src.size,
                     log_path="recordings/drive.jsonl", panel=True)
    ctrl_mod = Controller(src, lanes, target_kmh=args.target_kmh)

    print(f"[drive] pilote AUTONOME — cible {args.target_kmh:.0f} km/h, "
          f"{args.frames} frames, trafic {args.traffic}, seed {args.seed}", flush=True)

    prev_loc = None        # initialisé à la 1re frame (avant le 1er tick, la pose est fausse)
    dist_driven = 0.0
    speeds, actions = [], {}
    min_gap = 1e9

    for frame in src.frames():
        det_frame, cond = analyzer.process(frame)
        detections = detector.detect(det_frame)
        tracks = tracker.update(frame, detections)
        world = WorldState(frame.index, frame.timestamp, tracks, conditions=cond)
        ctx = lanes.estimate(frame)
        lanes.assign(world, ctx)
        world.lane_ctx = ctx
        world.ego_speed_mps = ctrl_mod._speed()
        world.ego_speed_measured = True
        estimator.update(world)
        state.update(world)
        behavior.update(world)
        risk.assess(world)

        # --- DÉCISION + application ---
        control = ctrl_mod.compute(frame.timestamp)

        # incruste l'état du pilote en haut à gauche (avant l'overlay du sink)
        spd = ctrl_mod._speed() * 3.6
        gap = f"{ctrl_mod.lead_dist:.0f}m" if ctrl_mod.lead_dist is not None else "libre"
        txt = f"PILOTE: {ctrl_mod.action}  |  {spd:4.1f} km/h  |  devant: {gap}"
        cv2.putText(frame.image, txt, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame.image, txt, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 255), 1, cv2.LINE_AA)

        sink.consume(frame, world)
        ego.apply_control(control)

        # télémétrie
        loc = ego.get_location()
        if prev_loc is not None:
            dist_driven += ((loc.x - prev_loc.x) ** 2 + (loc.y - prev_loc.y) ** 2) ** 0.5
        prev_loc = loc
        speeds.append(spd)
        actions[ctrl_mod.action] = actions.get(ctrl_mod.action, 0) + 1
        if ctrl_mod.lead_dist is not None:
            min_gap = min(min_gap, ctrl_mod.lead_dist)
        if frame.index % 100 == 0:
            print(f"  f{frame.index:4d}  {spd:4.1f} km/h  {ctrl_mod.action:16s} "
                  f"devant={gap:>6}  collisions={collisions['n']}  parcouru={dist_driven:.0f}m",
                  flush=True)

    sink.close()
    avg = sum(speeds) / len(speeds) if speeds else 0.0
    print("\n==================== BILAN PILOTE AUTONOME ====================")
    print(f"  distance parcourue : {dist_driven:.0f} m")
    print(f"  vitesse moyenne    : {avg:.1f} km/h")
    print(f"  collisions         : {collisions['n']}")
    print(f"  écart mini au lead : {min_gap:.1f} m" if min_gap < 1e8 else "  écart mini : voie libre")
    print(f"  actions            : " + ", ".join(f"{k}={v}" for k, v in
                                                  sorted(actions.items(), key=lambda x: -x[1])))
    print(f"  vidéo              : {args.output}")
    print("==============================================================")

    import subprocess
    try:
        subprocess.run([sys.executable, "carla_reset.py", "--quiet"], timeout=40)
    except Exception:
        pass
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
