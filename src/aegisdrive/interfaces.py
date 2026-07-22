"""Interfaces des modules — des Protocols, pas des classes de base.

Chaque étape du pipeline dépend d'une interface, jamais d'une implémentation concrète.
Cela permet de brancher un faux détecteur en test, un YOLO en dev, un TensorRT en prod,
sans toucher au reste du code (principe d'inversion de dépendance).
"""
from __future__ import annotations

from typing import Iterator, Protocol, runtime_checkable

from .schemas import Detection, Frame, SceneConditions, Track, WorldState


@runtime_checkable
class SceneAnalyzer(Protocol):
    """Estime les conditions (nuit/météo) et renvoie une frame rehaussée pour la détection."""
    def process(self, frame: Frame) -> tuple[Frame, SceneConditions]: ...


@runtime_checkable
class LaneEstimator(Protocol):
    """Détecte les voies et assigne une zone de voie à chaque track (en place)."""
    def estimate(self, frame: Frame): ...            # -> LaneContext (objet interne)
    def assign(self, world: WorldState, lane_ctx) -> None: ...


@runtime_checkable
class Source(Protocol):
    """Produit des frames (fichier vidéo aujourd'hui, caméra temps réel demain)."""
    def frames(self) -> Iterator[Frame]: ...
    @property
    def fps(self) -> float: ...
    @property
    def size(self) -> tuple[int, int]: ...  # (largeur, hauteur)


@runtime_checkable
class Detector(Protocol):
    """Frame -> détections brutes."""
    def detect(self, frame: Frame) -> list[Detection]: ...


@runtime_checkable
class Tracker(Protocol):
    """Associe les détections dans le temps -> tracks persistants."""
    def update(self, frame: Frame, detections: list[Detection]) -> list[Track]: ...


@runtime_checkable
class Estimator(Protocol):
    """Remplit la cinématique des tracks (distance/vitesse/accel/ttc), en place."""
    def update(self, world: WorldState) -> None: ...


@runtime_checkable
class StateEstimator(Protocol):
    """Enrichit l'état de chaque track (mouvement, dérive latérale), en place."""
    def update(self, world: WorldState) -> None: ...


@runtime_checkable
class BehaviorEngine(Protocol):
    """Reconnaît des comportements sur l'historique des tracks (mute Track.behaviors)."""
    def update(self, world: WorldState) -> None: ...


@runtime_checkable
class RiskEngine(Protocol):
    """Calcule un score de danger explicable par track (mutation en place)."""
    def assess(self, world: WorldState) -> None: ...


@runtime_checkable
class Sink(Protocol):
    """Consomme l'état final (écrit une vidéo annotée, un log, etc.)."""
    def consume(self, frame: Frame, world: WorldState) -> None: ...
    def close(self) -> None: ...
