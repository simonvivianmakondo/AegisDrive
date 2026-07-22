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
    p.add_argument("--input", required=True, help="vidéo d'entrée (.mp4)")
    p.add_argument("--output", default="output.mp4", help="vidéo annotée de sortie")
    p.add_argument("--detector", default="fake", choices=["fake", "yolo"])
    p.add_argument("--tracker", default="kalman", choices=["kalman", "iou"])
    p.add_argument("--log", default="replay.jsonl", help="log JSONL de l'état monde")
    p.add_argument("--fov", type=float, default=60.0,
                   help="champ de vision horizontal caméra en degrés (pour la distance)")
    p.add_argument("--drive-side", default="right", choices=["right", "left"],
                   help="sens de circulation (right = venant d'en face à gauche)")
    p.add_argument("--no-lanes", action="store_true", help="désactive la détection de voies")
    p.add_argument("--road", default="classic", choices=["classic", "seg"],
                   help="voies : 'classic' (Hough) ou 'seg' (segmentation IA YOLOPv2, robuste)")
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


if __name__ == "__main__":
    main()
