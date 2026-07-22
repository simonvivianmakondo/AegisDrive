"""carla_eval — évaluation quantifiée de la perception AegisDrive contre la vérité terrain CARLA.

Fait tourner la perception complète (YOLO -> tracking -> distance/TTC -> danger) sur N
frames CARLA, et à CHAQUE frame compare :
  - la distance ESTIMÉE de l'objet le plus proche devant (perception),
  - la distance RÉELLE de l'objet le plus proche devant dans le champ (vérité terrain).

Sort un résumé JSON : erreur de distance (biais/MAE), taux de détection, FPS, danger.
C'est l'étape B : mesurer pour savoir quoi améliorer, AVANT de piloter (étape C).

    .venv-carla\\Scripts\\python.exe carla_eval.py --frames 1000 --seed 0 --out eval_run0.json

Le correctif OpenMP est posé par main pour le pipeline ; ici on l'active aussi.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from aegisdrive.io.carla_source import CarlaSource
from aegisdrive.perception.yolo_detector import YoloDetector
from aegisdrive.tracking.kalman_tracker import KalmanTracker
from aegisdrive.estimation.kinematics import KinematicsEstimator
from aegisdrive.scene.lanes import LaneEstimator
from aegisdrive.scene.road_segmenter import RoadSegmenter
from aegisdrive.understanding.object_state import StateEstimator
from aegisdrive.understanding.behavior import BehaviorEngine
from aegisdrive.risk.rule_engine import RuleRiskEngine
from aegisdrive.schemas import WorldState, LaneZone


def _match_and_error(tracks, gts):
    """Apparie chaque OBJET RÉEL (projeté) à la détection qui le contient le mieux.

    Piloté par la vérité terrain (pas par les boîtes) : pour chaque acteur réel visible,
    on prend la boîte la plus SERRÉE qui contient sa projection, et on exige que le point
    tombe près du CENTRE de la boîte (évite d'apparier un objet lointain à la grande boîte
    d'un objet proche devant lui). Renvoie les paires (distance_estimée, profondeur_réelle)
    pour le MÊME objet. Note : les décors statiques (voitures « peintes ») n'ont pas
    d'acteur réel -> ils ne sont volontairement pas comptés ici.
    """
    pairs = []
    for g in gts:
        u, v = g["u"], g["v"]
        cont = [t for t in tracks if t.distance_m is not None
                and t.bbox.x1 <= u <= t.bbox.x2 and t.bbox.y1 <= v <= t.bbox.y2]
        if not cont:
            continue
        t = min(cont, key=lambda x: x.bbox.area)     # boîte la plus serrée = objet spécifique
        cx, cy = t.bbox.center
        if abs(u - cx) > 0.45 * t.bbox.width or abs(v - cy) > 0.45 * t.bbox.height:
            continue                                  # projection trop excentrée -> occlusion probable
        pairs.append((t.distance_m, g["depth_m"]))
    return pairs


def main() -> int:
    ap = argparse.ArgumentParser(description="Évaluation perception AegisDrive vs vérité terrain CARLA")
    ap.add_argument("--frames", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--traffic", type=int, default=40)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--fov", type=float, default=60.0)
    ap.add_argument("--cam-height", type=float, default=1.2)
    ap.add_argument("--proc-width", type=int, default=960)
    ap.add_argument("--road", default="carla", choices=["classic", "seg", "carla"],
                    help="voies : 'classic' (Hough), 'seg' (YOLOPv2), 'carla' (waypoints carte)")
    ap.add_argument("--road-model", default="models/YOLOPv2.onnx")
    ap.add_argument("--out", default="eval.json")
    args = ap.parse_args()

    w = args.proc_width
    h = int(round(w * 9 / 16))
    src = CarlaSource(host=args.host, port=args.port, fps=20.0, width=w, height=h,
                      fov=args.fov, cam_height=args.cam_height, n_traffic=args.traffic,
                      max_frames=args.frames, seed=args.seed, spectator_follow=False)
    detector = YoloDetector()
    tracker = KalmanTracker()
    estimator = KinematicsEstimator(image_width_px=src.size[0], image_height_px=src.size[1],
                                    fov_deg=args.fov, cam_height_m=args.cam_height)
    if args.road == "carla":
        from aegisdrive.scene.carla_lanes import CarlaLaneProvider
        lanes = CarlaLaneProvider(src, drive_side="right")     # vérité carte (waypoints)
    elif args.road == "seg":
        # YOLOPv2 seul (pas d'UFLD) : robuste ET léger pour l'assignation de voie.
        lanes = RoadSegmenter(args.road_model, drive_side="right", lane_detector=None)
    else:
        lanes = LaneEstimator(drive_side="right")
    state = StateEstimator()
    behavior = BehaviorEngine()
    risk = RuleRiskEngine(frame_size=src.size)

    print(f"[eval] seed={args.seed} frames={args.frames} trafic={args.traffic} "
          f"{src.size[0]}x{src.size[1]} fov={args.fov:g}°  device={getattr(detector,'_device','?')}",
          flush=True)

    n = 0
    ego_lane_frames = 0       # frames avec >=1 objet dans la voie ego
    danger_frames = 0         # frames avec un danger élevé
    total_det = 0
    errors: list[float] = []  # est - depth_réelle (m), signé, sur le MÊME objet
    rel_errors: list[float] = []
    est_vals: list[float] = []
    gt_vals: list[float] = []
    ttc_mins: list[float] = []
    vis_near_total = 0        # objets réels visibles à <60 m (dénominateur du rappel)
    vis_near_matched = 0      # dont détectés (numérateur du rappel)

    t0 = time.perf_counter()
    for frame in src.frames():
        detections = detector.detect(frame)
        tracks = tracker.update(frame, detections)
        world = WorldState(frame.index, frame.timestamp, tracks)
        ctx = lanes.estimate(frame)
        lanes.assign(world, ctx)
        world.lane_ctx = ctx
        estimator.update(world)
        state.update(world)
        behavior.update(world)
        risk.assess(world)

        n += 1
        total_det += len(detections)
        if any(t.lane_zone == LaneZone.EGO for t in tracks):
            ego_lane_frames += 1
        if tracks and max(t.danger_score for t in tracks) >= 60.0:
            danger_frames += 1
        ttcs = [t.ttc_s for t in tracks if t.ttc_s is not None and t.ttc_s > 0]
        if ttcs:
            ttc_mins.append(min(ttcs))

        # --- appariement rigoureux perception <-> vérité terrain projetée ---
        gts = src.visible_gt()
        for est_d, gt_d in _match_and_error(tracks, gts):
            if gt_d > 0.5:
                errors.append(est_d - gt_d)
                rel_errors.append((est_d - gt_d) / gt_d)
                est_vals.append(est_d)
                gt_vals.append(gt_d)
        # rappel de détection sur les objets réellement visibles et proches (<60 m)
        near = [g for g in gts if g["depth_m"] < 60.0]
        vis_near_total += len(near)
        for g in near:
            if any(t.bbox.x1 <= g["u"] <= t.bbox.x2 and t.bbox.y1 <= g["v"] <= t.bbox.y2
                   for t in tracks):
                vis_near_matched += 1

        if n % 100 == 0:
            fps = n / (time.perf_counter() - t0)
            print(f"  {n}/{args.frames}  {fps:4.1f} FPS  paires_mesurees={len(errors)}", flush=True)

    elapsed = time.perf_counter() - t0

    def _stats(xs):
        if not xs:
            return None
        return {"n": len(xs), "mean": round(st.mean(xs), 3),
                "median": round(st.median(xs), 3),
                "stdev": round(st.pstdev(xs), 3) if len(xs) > 1 else 0.0}

    summary = {
        "seed": args.seed,
        "frames": n,
        "elapsed_s": round(elapsed, 1),
        "fps": round(n / elapsed, 2) if elapsed > 0 else 0.0,
        "avg_detections_per_frame": round(total_det / n, 2) if n else 0.0,
        "recall_near_60m": round(vis_near_matched / vis_near_total, 3) if vis_near_total else None,
        "ego_lane_rate": round(ego_lane_frames / n, 3) if n else 0.0,
        "danger_frame_rate": round(danger_frames / n, 3) if n else 0.0,
        "ttc_min_overall": round(min(ttc_mins), 2) if ttc_mins else None,
        # coeur de l'évaluation : erreur de distance sur le MÊME objet (perception vs réel)
        "dist_pairs": len(errors),
        "dist_bias_m": _stats(errors),          # signé : + = surestime, - = sous-estime
        "dist_abs_err_m": _stats([abs(e) for e in errors]),
        "dist_rel_err": _stats(rel_errors),     # erreur relative (fraction)
        "est_mean_m": round(st.mean(est_vals), 2) if est_vals else None,
        "gt_mean_m": round(st.mean(gt_vals), 2) if gt_vals else None,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("\n" + json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    print(f"[eval] résumé -> {args.out}", flush=True)

    # nettoyage serveur (process frais) + sortie dure (crash natif CARLA au teardown)
    import subprocess
    try:
        subprocess.run([sys.executable, "carla_reset.py", "--quiet"], timeout=40)
    except Exception:
        pass
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
