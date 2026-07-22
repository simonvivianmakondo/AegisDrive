"""Contrats de données — la source de vérité du système.

Tous les modules communiquent *uniquement* via ces structures. Un module peut être
réécrit (Python -> C++, YOLO -> TensorRT) tant qu'il respecte ces schémas.

Volontairement en dataclasses stdlib : zéro dépendance, sérialisable en JSON pour le
log de replay (Phase 5). On pourra passer à Pydantic/protobuf sans changer les champs.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional
import numpy as np


class MotionState(str, Enum):
    """État longitudinal d'un objet.

    Par défaut relatif à l'ego (dashcam sans vitesse propre). Les états ABSOLUS
    (STOPPED/ACCELERATING/...) ne sont produits que si la vitesse ego est fournie.
    """
    UNKNOWN = "unknown"
    # --- relatif à l'ego (défaut) ---
    CLOSING = "closing"          # se rapproche
    CLOSING_FAST = "closing_fast"  # se rapproche vite
    MATCHING = "matching"        # roule à notre allure
    RECEDING = "receding"        # s'éloigne
    # --- absolu (si ego_speed connu) ---
    STOPPED = "stopped"
    CRUISING = "cruising"
    ACCELERATING = "accelerating"
    DECELERATING = "decelerating"
    BRAKING_HARD = "braking_hard"


class LateralState(str, Enum):
    """Dérive latérale d'un objet (indépendante de l'ego) — amorce de changement de voie."""
    UNKNOWN = "unknown"
    KEEPING = "keeping"          # tient sa trajectoire
    TO_LEFT = "to_left"          # dérive vers la gauche (image)
    TO_RIGHT = "to_right"        # dérive vers la droite (image)


class Behavior(str, Enum):
    """Comportements reconnus sur l'historique d'un track — Étape 2."""
    LANE_CHANGE_LEFT = "lane_change_left"
    LANE_CHANGE_RIGHT = "lane_change_right"
    CUT_IN = "cut_in"                 # se rabat vers ma voie
    OVERTAKING = "overtaking"         # nous dépasse
    HARD_BRAKING = "hard_braking"     # freinage brusque devant
    STOPPED = "stopped"               # immobilisé (si vitesse ego connue)
    APPROACHING_FAST = "approaching_fast"
    PED_CROSSING = "ped_crossing"     # piéton qui traverse
    PED_WAITING = "ped_waiting"       # piéton qui attend


class LaneZone(str, Enum):
    """Zone d'un objet par rapport à la chaussée de l'ego-véhicule."""
    EGO = "ego"                 # dans ma voie
    ADJACENT_LEFT = "adj_left"  # voie adjacente à gauche
    ADJACENT_RIGHT = "adj_right"  # voie adjacente à droite
    OPPOSITE = "opposite"       # chaussée opposée (séparée / terre-plein)
    UNKNOWN = "unknown"


class ObjectClass(str, Enum):
    """Catégories perçues (le PRD en liste ~10)."""
    CAR = "car"
    TRUCK = "truck"
    BUS = "bus"
    MOTORCYCLE = "motorcycle"
    BICYCLE = "bicycle"
    PEDESTRIAN = "pedestrian"
    TRAFFIC_SIGN = "traffic_sign"
    TRAFFIC_LIGHT = "traffic_light"
    ANIMAL = "animal"
    OBSTACLE = "obstacle"
    UNKNOWN = "unknown"


@dataclass
class BBox:
    """Boîte englobante en pixels image (coin haut-gauche / bas-droit)."""
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    def iou(self, other: "BBox") -> float:
        ix1, iy1 = max(self.x1, other.x1), max(self.y1, other.y1)
        ix2, iy2 = min(self.x2, other.x2), min(self.y2, other.y2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0


@dataclass
class Frame:
    """Une image + son horodatage. `image` est HxWx3 BGR (convention OpenCV)."""
    index: int
    timestamp: float          # secondes depuis le début de la vidéo
    image: np.ndarray

    @property
    def shape(self) -> tuple[int, int]:
        h, w = self.image.shape[:2]
        return w, h


@dataclass
class Detection:
    """Sortie brute d'un détecteur pour une frame. Sans identité temporelle."""
    bbox: BBox
    cls: ObjectClass
    confidence: float


@dataclass
class Track:
    """Un objet suivi dans le temps — l'« entité intelligente » du PRD.

    C'est la source de vérité (pas la détection). Il accumule un état estimé
    (position/vitesse) que la Phase 2 remplira via un filtre de Kalman.
    """
    id: int
    cls: ObjectClass
    bbox: BBox
    confidence: float
    age: int = 0                       # nb de frames depuis la création
    missed: int = 0                    # nb de frames consécutives sans association
    # État estimé (rempli progressivement — Phase 2+)
    distance_m: Optional[float] = None
    speed_mps: Optional[float] = None
    accel_mps2: Optional[float] = None
    ttc_s: Optional[float] = None      # time-to-collision
    lane_zone: "LaneZone" = None       # rempli par le module scene (défaut UNKNOWN)
    # État enrichi (module understanding — Étape 1)
    motion_state: "MotionState" = None
    lateral_state: "LateralState" = None
    speed_abs_mps: Optional[float] = None   # vitesse absolue si ego_speed connu
    behaviors: list = field(default_factory=list)   # list[Behavior] — Étape 2
    danger_score: float = 0.0          # 0..100
    explanations: list[str] = field(default_factory=list)

    def __post_init__(self):
        if self.lane_zone is None:
            self.lane_zone = LaneZone.UNKNOWN
        if self.motion_state is None:
            self.motion_state = MotionState.UNKNOWN
        if self.lateral_state is None:
            self.lateral_state = LateralState.UNKNOWN


@dataclass
class SceneConditions:
    """Conditions perçues de la scène (lumière / visibilité) — Phase météo/nuit."""
    brightness: float          # luminance moyenne 0..255
    contrast: float            # écart-type de la luminance (proxy de netteté/brume)
    is_night: bool
    visibility: float          # 0 (nul) .. 1 (dégagé)
    label: str                 # ex. "jour clair", "nuit", "faible visibilité"


@dataclass
class WorldState:
    """État complet de la scène à un instant t. C'est CE qu'on logge pour le replay.

    Sérialiser un WorldState par frame en JSONL rend le replay (Phase 5) et la
    reconstruction 3D (Phase 4) quasi gratuits : ce sont des lecteurs de ce log.
    """
    frame_index: int
    timestamp: float
    tracks: list[Track] = field(default_factory=list)
    conditions: Optional[SceneConditions] = None
    ego_speed_mps: Optional[float] = None   # vitesse propre (None en dashcam)
    ego_speed_measured: bool = False         # True si capteur (CAN/GPS), False si estimée
    # Contexte de voies pour le dessin — NON sérialisé (objet géométrique volatil).
    lane_ctx: object = None

    def snapshot(self) -> "WorldState":
        """Copie figée pour la consommation asynchrone (thread d'écriture).

        Les `Track` sont des objets PERSISTANTS que les étages suivantes réécrivent
        en place à la frame suivante. Le thread du `VideoSink` lit ces mêmes objets
        (dessin + sérialisation JSON) : sans figer une copie, il verrait un mélange
        de deux frames (bbox de l'une, score de l'autre). On copie donc chaque track
        (copie superficielle suffisante : les étages RÉ-ASSIGNENT les attributs, ne
        les mutent pas sur place). `conditions` et `lane_ctx` sont recréés à chaque
        frame et jamais mutés après coup -> partage de référence sûr.
        """
        return WorldState(
            frame_index=self.frame_index,
            timestamp=self.timestamp,
            tracks=[copy.copy(t) for t in self.tracks],
            conditions=self.conditions,
            ego_speed_mps=self.ego_speed_mps,
            ego_speed_measured=self.ego_speed_measured,
            lane_ctx=self.lane_ctx,
        )

    def to_json_dict(self) -> dict:
        """Représentation JSON-safe (sans l'image ni la géométrie, enums en str)."""
        out = {
            "frame_index": self.frame_index,
            "timestamp": self.timestamp,
            "tracks": [
                {**{k: v for k, v in asdict(t).items() if k not in ("bbox",)},
                 "cls": t.cls.value, "lane_zone": t.lane_zone.value,
                 "motion_state": t.motion_state.value,
                 "lateral_state": t.lateral_state.value,
                 "behaviors": [b.value for b in t.behaviors],
                 "bbox": asdict(t.bbox)}
                for t in self.tracks
            ],
        }
        if self.conditions is not None:
            out["conditions"] = asdict(self.conditions)
        return out
