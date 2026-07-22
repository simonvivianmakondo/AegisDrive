"""Tests unitaires — tournent sans GPU, sans modèle, sans fichier vidéo."""
import numpy as np

from aegisdrive.schemas import BBox, Detection, Frame, ObjectClass, WorldState
from aegisdrive.perception.fake_detector import FakeDetector
from aegisdrive.tracking.simple_tracker import SimpleIoUTracker
from aegisdrive.risk.rule_engine import RuleRiskEngine


def _frame(idx=0, w=640, h=480):
    return Frame(index=idx, timestamp=idx / 30.0, image=np.zeros((h, w, 3), np.uint8))


def test_bbox_iou():
    a = BBox(0, 0, 10, 10)
    assert a.iou(BBox(0, 0, 10, 10)) == 1.0
    assert a.iou(BBox(100, 100, 110, 110)) == 0.0


def test_fake_detector_produces_objects():
    dets = FakeDetector().detect(_frame())
    assert len(dets) == 2
    assert any(d.cls is ObjectClass.PEDESTRIAN for d in dets)


def test_tracker_keeps_stable_id_across_frames():
    tracker = SimpleIoUTracker()
    det = FakeDetector()
    ids_over_time = []
    for i in range(5):
        tracks = tracker.update(_frame(i), det.detect(_frame(i)))
        peds = [t for t in tracks if t.cls is ObjectClass.PEDESTRIAN]
        assert peds, "le piéton fixe doit rester suivi"
        ids_over_time.append(peds[0].id)
    assert len(set(ids_over_time)) == 1, "l'id du piéton doit rester stable"


def test_risk_engine_scores_and_explains():
    engine = RuleRiskEngine(frame_size=(640, 480))
    ped = Detection(BBox(300, 200, 340, 300), ObjectClass.PEDESTRIAN, 0.9)
    tracks = SimpleIoUTracker().update(_frame(), [ped])
    world = WorldState(0, 0.0, tracks)
    engine.assess(world)
    t = world.tracks[0]
    assert 0.0 <= t.danger_score <= 100.0
    assert t.explanations, "un piéton doit produire au moins une explication"


def test_worldstate_json_is_serializable():
    import json
    tracks = SimpleIoUTracker().update(
        _frame(), [Detection(BBox(0, 0, 10, 10), ObjectClass.CAR, 0.5)])
    payload = WorldState(0, 0.0, tracks).to_json_dict()
    json.dumps(payload)  # ne doit pas lever


def test_worldstate_snapshot_is_isolated():
    """Le snapshot fige les tracks : muter l'original ne doit pas l'affecter (anti-race)."""
    tracks = SimpleIoUTracker().update(
        _frame(), [Detection(BBox(0, 0, 10, 10), ObjectClass.CAR, 0.5)])
    world = WorldState(0, 0.0, tracks)
    snap = world.snapshot()
    world.tracks[0].danger_score = 99.0
    world.tracks[0].bbox = BBox(5, 5, 15, 15)
    world.tracks[0].behaviors.append("mutation")
    assert snap.tracks[0] is not world.tracks[0]
    assert snap.tracks[0].danger_score == 0.0
    assert snap.tracks[0].bbox.x1 == 0.0


def test_lane_fit_rejects_outlier():
    """L'ajustement robuste (RANSAC allégé) ignore un point aberrant isolé."""
    from aegisdrive.scene.lanes import LaneLine
    ys = [100, 150, 200, 250, 300, 350]
    xs = [100, 110, 120, 130, 140, 150]          # droite propre
    xs_out = xs[:]; xs_out[2] = 400               # 1 outlier
    clean = LaneLine.fit(ys, xs, degree=1, y_scale=480)
    robust = LaneLine.fit(ys, xs_out, degree=1, y_scale=480)
    assert abs(robust.x_at(400) - clean.x_at(400)) < 12


def test_lane_curvature_direction_and_straight():
    """Voie droite -> pas de rayon ; voie courbée -> rayon fini + sens gauche/droite."""
    from aegisdrive.scene.lanes import LaneLine, lane_curvature
    straight_l = LaneLine(0.0, -0.3, 120, 100, 470)
    straight_r = LaneLine(0.0, -0.3, 320, 100, 470)
    assert lane_curvature(straight_l, straight_r, 480, 200)[0] is None
    cl = LaneLine.fit([120, 240, 360, 470], [140, 120, 108, 100], degree=2, y_scale=480)
    cr = LaneLine.fit([120, 240, 360, 470], [340, 320, 308, 300], degree=2, y_scale=480)
    radius, direction = lane_curvature(cl, cr, 480, 200)
    assert radius is not None and radius > 0
    assert direction in ("left", "right")
