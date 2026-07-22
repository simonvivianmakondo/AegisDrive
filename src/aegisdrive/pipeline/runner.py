"""Orchestrateur — mode vidéo séquentiel (déterministe).

La logique est ici indépendante du threading. Le mode temps réel (futur) réutilisera
EXACTEMENT ces mêmes composants (Detector/Tracker/RiskEngine/Sink) répartis sur des
threads reliés par des files — sans réécrire leur logique.
"""
from __future__ import annotations

import time

from ..interfaces import (BehaviorEngine, Detector, Estimator, LaneEstimator,
                          RiskEngine, SceneAnalyzer, Sink, Source,
                          StateEstimator, Tracker)
from ..schemas import WorldState


class PipelineStats:
    def __init__(self) -> None:
        self.frames = 0
        self.total_tracks = 0
        self.elapsed_s = 0.0

    @property
    def avg_fps(self) -> float:
        return self.frames / self.elapsed_s if self.elapsed_s > 0 else 0.0


def run_video(source: Source, detector: Detector, tracker: Tracker,
              estimator: Estimator, risk: RiskEngine, sink: Sink,
              analyzer: SceneAnalyzer | None = None,
              lanes: LaneEstimator | None = None,
              state: StateEstimator | None = None,
              behavior: BehaviorEngine | None = None,
              ego=None) -> PipelineStats:
    stats = PipelineStats()
    total = getattr(source, "frame_count", 0)
    t0 = time.perf_counter()
    win_t, win_f = t0, 0          # fenêtre glissante pour un FPS/ETA réaliste
    win_fps = 0.0
    for frame in source.frames():
        # Conditions (nuit/météo) + frame rehaussée pour la détection.
        cond = None
        det_frame = frame
        if analyzer is not None:
            det_frame, cond = analyzer.process(frame)

        detections = detector.detect(det_frame)
        tracks = tracker.update(frame, detections)
        world = WorldState(frame.index, frame.timestamp, tracks, conditions=cond)

        # Voies : détection + assignation de zone à chaque track.
        if lanes is not None:
            ctx = lanes.estimate(frame)
            lanes.assign(world, ctx)
            world.lane_ctx = ctx

        # Mouvement propre (ego-motion) -> vitesse ego, avant l'état des objets.
        if ego is not None:
            world.ego_speed_mps = ego.update(frame, world.lane_ctx)

        estimator.update(world)   # distance / vitesse / accel / ttc
        if state is not None:
            state.update(world)   # motion_state / lateral_state (Étape 1)
        if behavior is not None:
            behavior.update(world)  # comportements (Étape 2)
        risk.assess(world)        # exploite conditions + zone + comportements
        sink.consume(frame, world)

        stats.frames += 1
        stats.total_tracks += len(tracks)

        # Progression (mise à jour sur la même ligne, toutes les ~10 frames).
        if stats.frames % 10 == 0 or stats.frames == total:
            now = time.perf_counter()
            # FPS sur la FENÊTRE récente (pas depuis le début) -> ETA réaliste.
            dw = now - win_t
            if dw > 0:
                win_fps = (stats.frames - win_f) / dw
            win_t, win_f = now, stats.frames
            if total > 0:
                pct = 100.0 * stats.frames / total
                eta = (total - stats.frames) / win_fps if win_fps > 0 else 0.0
                print(f"\r  Analyse : {pct:5.1f}%  ({stats.frames}/{total})  "
                      f"{win_fps:.0f} FPS  ETA {eta:4.0f}s   ", end="", flush=True)
            else:
                print(f"\r  Analyse : {stats.frames} frames  {win_fps:.0f} FPS",
                      end="", flush=True)
    print()  # saut de ligne final après la barre de progression
    stats.elapsed_s = time.perf_counter() - t0
    sink.close()
    return stats
