"""Point d'entrée : input.mp4 -> output.mp4 annoté + log JSONL.

    python -m aegisdrive.main --input input.mp4 --output output.mp4
    python -m aegisdrive.main --input input.mp4 --detector yolo   # nécessite ultralytics
"""
from __future__ import annotations

import argparse

from .estimation.kinematics import KinematicsEstimator
from .io.video_sink import VideoSink
from .io.video_source import VideoFileSource
from .pipeline.runner import run_video
from .preprocess.conditions import SceneAnalyzer
from .risk.rule_engine import RuleRiskEngine
from .scene.lanes import LaneEstimator
from .scene.road_segmenter import RoadSegmenter
from .tracking.kalman_tracker import KalmanTracker
from .tracking.simple_tracker import SimpleIoUTracker
from .ego.ego_motion import EgoMotionEstimator
from .understanding.behavior import BehaviorEngine
from .understanding.object_state import StateEstimator


def _build_tracker(name: str):
    if name == "kalman":
        return KalmanTracker()
    if name == "iou":
        return SimpleIoUTracker()
    raise ValueError(f"Tracker inconnu : {name!r} (kalman|iou)")


def _build_lane_detector(args):
    """Détecteur de lignes pour le mode seg. 'auto' privilégie UFLD (bien plus fidèle à
    la voie ego) si son modèle est présent, avec repli propre sur les fenêtres glissantes."""
    import os
    mode = args.lanes_model
    if mode == "classic":
        return None
    if mode == "auto" and not os.path.exists(args.ufld_model):
        print("  Lignes : UFLD indisponible -> fenêtres glissantes (mode classic)")
        return None
    try:
        from .scene.ufld_lanes import UFLDLaneDetector
        det = UFLDLaneDetector(args.ufld_model)
        print("  Lignes : UFLD (réseau dédié)")
        return det
    except Exception as exc:   # onnxruntime/modèle absent -> repli non bloquant
        if mode == "ufld":
            raise
        print(f"  Lignes : UFLD a échoué ({type(exc).__name__}) -> fenêtres glissantes")
        return None


def _build_detector(name: str):
    if name == "fake":
        from .perception.fake_detector import FakeDetector
        return FakeDetector()
    if name == "yolo":
        from .perception.yolo_detector import YoloDetector
        return YoloDetector()
    raise ValueError(f"Détecteur inconnu : {name!r} (fake|yolo)")


def main() -> None:
    p = argparse.ArgumentParser(description="AegisDrive — analyse vidéo de scène routière")
    p.add_argument("--source", default="video", choices=["video", "carla"],
                   help="source des frames : 'video' (fichier mp4) ou 'carla' (simulateur temps réel)")
    p.add_argument("--input", help="vidéo d'entrée (.mp4) — requis si --source video")
    p.add_argument("--output", default="output.mp4", help="vidéo annotée de sortie")
    # --- options CARLA (utilisées seulement si --source carla) ---
    p.add_argument("--carla-host", default="127.0.0.1", help="hôte du serveur CARLA")
    p.add_argument("--carla-port", type=int, default=2000, help="port RPC du serveur CARLA")
    p.add_argument("--carla-frames", type=int, default=400,
                   help="nb de frames à capturer depuis CARLA (source infinie sinon)")
    p.add_argument("--carla-traffic", type=int, default=40, help="nb de véhicules PNJ dans la scène")
    p.add_argument("--carla-town", default=None, help="carte CARLA à charger (ex. Town10HD)")
    p.add_argument("--carla-cam-x", type=float, default=2.3,
                   help="avancée de la caméra sur l'ego en mètres (hors du capot)")
    p.add_argument("--carla-seed", type=int, default=0,
                   help="graine (spawn ego + trafic) — varie la route entre deux runs")
    p.add_argument("--detector", default="fake", choices=["fake", "yolo"])
    p.add_argument("--tracker", default="kalman", choices=["kalman", "iou"])
    p.add_argument("--log", default="replay.jsonl", help="log JSONL de l'état monde")
    p.add_argument("--fov", type=float, default=60.0,
                   help="champ de vision horizontal caméra en degrés (pour la distance)")
    p.add_argument("--drive-side", default="right", choices=["right", "left"],
                   help="sens de circulation (right = venant d'en face à gauche)")
    p.add_argument("--no-lanes", action="store_true", help="désactive la détection de voies")
    p.add_argument("--road", default="classic", choices=["classic", "seg", "carla"],
                   help="voies : 'classic' (Hough), 'seg' (YOLOPv2), 'carla' (waypoints carte — "
                        "défaut auto en --source carla, bien plus fiable sur le simulateur)")
    p.add_argument("--road-model", default="models/YOLOPv2.onnx",
                   help="chemin du modèle ONNX de segmentation (mode --road seg)")
    p.add_argument("--lanes-model", default="auto", choices=["auto", "classic", "ufld"],
                   help="lignes : 'auto' (UFLD si le modèle est présent, sinon fenêtres "
                        "glissantes), 'classic' (fenêtres glissantes), 'ufld' (réseau dédié). "
                        "UFLD suit bien mieux la voie ego (pointillés inclus).")
    p.add_argument("--ufld-model", default="models/ufldv2_culane_res34_320x1600.onnx",
                   help="chemin du modèle ONNX UFLD (détection de lignes dédiée)")
    p.add_argument("--no-enhance", action="store_true",
                   help="désactive le rehaussement nuit/météo (garde l'analyse des conditions)")
    p.add_argument("--no-panel", action="store_true",
                   help="désactive le panneau latéral (vidéo seule, sans synthèse à droite)")
    p.add_argument("--no-ego", action="store_true",
                   help="désactive l'estimation du mouvement propre (ego-motion)")
    p.add_argument("--cam-height", type=float, default=1.2,
                   help="hauteur caméra en mètres (échelle de la vitesse ego estimée)")
    p.add_argument("--proc-width", type=int, default=960,
                   help="largeur de traitement : les vidéos plus larges sont réduites "
                        "(gros gain de vitesse en 1080p/4K). 0 = pleine résolution.")
    p.add_argument("--frame-stride", type=int, default=1,
                   help="analyse 1 frame sur N (ex. 2 pour une vidéo 60fps -> 30fps analysés).")
    p.add_argument("--start", type=float, default=0.0,
                   help="début du segment à analyser, en secondes (défaut 0 = début).")
    p.add_argument("--end", type=float, default=None,
                   help="fin du segment à analyser, en secondes (défaut = fin de la vidéo). "
                        "Ex. --start 50 --end 90 n'analyse que les secondes 50 à 90.")
    p.add_argument("--report", default="report.json", help="chemin du rapport de fin (JSON + .txt)")
    p.add_argument("--no-report", action="store_true", help="ne pas générer le rapport de fin")
    args = p.parse_args()

    if args.source == "carla":
        # CARLA (client natif) + torch CUDA dans le même process -> fast-fail 0xC0000409
        # (conflit de runtime OpenMP sous Windows). On neutralise AVANT tout import torch.
        import os
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
        from .io.carla_source import CarlaSource
        cam_w = args.proc_width if args.proc_width and args.proc_width > 0 else 1280
        cam_h = int(round(cam_w * 9 / 16))
        # fov et cam_height sont partagés avec l'estimateur -> distances cohérentes.
        source = CarlaSource(host=args.carla_host, port=args.carla_port, fps=20.0,
                             width=cam_w, height=cam_h, fov=args.fov,
                             cam_height=args.cam_height, cam_forward=args.carla_cam_x,
                             n_traffic=args.carla_traffic, max_frames=args.carla_frames,
                             town=args.carla_town, seed=args.carla_seed)
        # Sur CARLA, la détection de voies par image échoue (réseaux entraînés sur du
        # réel) : on passe par la vérité carte (waypoints) par défaut.
        if args.road == "classic":
            args.road = "carla"
    else:
        if not args.input:
            p.error("--input est requis avec --source video")
        source = VideoFileSource(args.input, proc_width=args.proc_width or None,
                                 stride=args.frame_stride, start_s=args.start, end_s=args.end)
    detector = _build_detector(args.detector)
    tracker = _build_tracker(args.tracker)
    estimator = KinematicsEstimator(image_width_px=source.size[0],
                                    image_height_px=source.size[1],
                                    fov_deg=args.fov, cam_height_m=args.cam_height)
    risk = RuleRiskEngine(frame_size=source.size)
    analyzer = SceneAnalyzer(enhance=not args.no_enhance)
    if args.no_lanes:
        lanes = None
    elif args.road == "carla":
        from .scene.carla_lanes import CarlaLaneProvider
        lanes = CarlaLaneProvider(source, drive_side=args.drive_side)
    elif args.road == "seg":
        lane_det = _build_lane_detector(args)
        lanes = RoadSegmenter(args.road_model, drive_side=args.drive_side,
                              lane_detector=lane_det)
    else:
        lanes = LaneEstimator(drive_side=args.drive_side)
    state = StateEstimator()
    behavior = BehaviorEngine()
    ego = None if args.no_ego else EgoMotionEstimator(fov_deg=args.fov,
                                                      cam_height_m=args.cam_height)
    sink = VideoSink(args.output, fps=source.fps, size=source.size, log_path=args.log,
                     panel=not args.no_panel)

    dev = getattr(detector, "_device", "?")
    dev_str = "GPU (CUDA)" if dev == 0 else "CPU"
    w, h = source.size
    if args.source == "carla":
        print(f"[AegisDrive] CARLA {args.carla_host}:{args.carla_port} -> {args.output}")
        print(f"  Simulateur : {w}x{h}  {source.fps:.0f} fps  {args.carla_frames} frames  "
              f"trafic={args.carla_traffic}  fov={args.fov:.0f}°")
    else:
        print(f"[AegisDrive] {args.input} -> {args.output}")
        seg = ""
        if args.start > 0 or args.end is not None:
            seg = f"  segment {args.start:.0f}s -> {args.end if args.end is not None else 'fin'}s"
        print(f"  Vidéo : {w}x{h}  {source.fps:.0f} fps  {source.frame_count} frames{seg}  "
              f"codec={source.codec}")
    print(f"  Calcul : {dev_str}  |  détecteur={args.detector}  tracker={args.tracker}  "
          f"voies={not args.no_lanes}")
    stats = run_video(source, detector, tracker, estimator, risk, sink,
                      analyzer=analyzer, lanes=lanes, state=state, behavior=behavior, ego=ego)
    print(f"[AegisDrive] {stats.frames} frames, "
          f"{stats.avg_fps:.1f} FPS moyen, "
          f"{stats.total_tracks} tracks cumulés. Log: {args.log}")

    # Rapport automatique de fin de vidéo (Phase 5).
    if not args.no_report and args.log:
        from .analytics.report import generate, format_text
        r = generate(args.log, args.report,
                     extra={"processing_fps": round(stats.avg_fps, 1),
                            "elapsed_s": round(stats.elapsed_s, 1)})
        print("\n" + format_text(r))
        print(f"\nRapport écrit : {args.report} (+ .txt)")

    # Source CARLA : la lib native crashe à l'extinction de Python. On nettoie le
    # serveur dans un process frais puis on sort en dur (voir io/carla_source.py).
    if args.source == "carla":
        import os
        import subprocess
        import sys
        try:
            subprocess.run([sys.executable, "carla_reset.py", "--quiet"], timeout=40)
        except Exception:
            pass
        sys.stdout.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
