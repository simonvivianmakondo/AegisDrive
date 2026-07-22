"""Segmentation de la route par IA (YOLOPv2 ONNX) — Étape 3.

Remplace la détection de voies classique (Canny+Hough, fragile) par un modèle entraîné
sur des scènes de conduite. Une passe fournit :
  - la ZONE ROULABLE (drivable area) -> on en déduit les *vraies* limites de la route ;
  - les LIGNES DE VOIE.

Se branche DERRIÈRE la même interface `LaneEstimator` (estimate/assign) : il produit un
`LaneContext` (avec en plus les masques pour le dessin), donc l'assignation de zone,
l'affichage et le reste du pipeline fonctionnent sans modification. Fallback classique
conservé si le modèle est absent.

Exécution sur GPU via onnxruntime-gpu, en réutilisant les DLL CUDA/cuDNN de torch
(aucune installation CUDA système requise).
"""
from __future__ import annotations

import os

import cv2
import numpy as np

from ..learning.profile import SceneProfile
from ..schemas import LaneZone, WorldState
from .lanes import (LaneContext, LaneEstimator, LaneLine, corridor_horizon,
                   lane_curvature)
from .lane_tracker import LaneLineTracker

_INPUT = 640   # résolution carrée d'entrée du modèle


def _register_torch_cuda_dlls() -> None:
    """Rend les DLL CUDA 13 / cuDNN 9 embarquées par torch visibles à onnxruntime."""
    try:
        import torch
        lib = os.path.join(os.path.dirname(torch.__file__), "lib")
        if os.path.isdir(lib):
            os.add_dll_directory(lib)
    except Exception:
        pass


class RoadSegmenter:
    def __init__(self, model_path: str = "models/YOLOPv2.onnx",
                 drive_side: str = "right", lane_thr: float = 0.5,
                 lane_detector=None):
        _register_torch_cuda_dlls()
        import onnxruntime as ort
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self._sess = ort.InferenceSession(model_path, providers=providers)
        self._on_gpu = self._sess.get_providers()[0] == "CUDAExecutionProvider"
        self._input = self._sess.get_inputs()[0].name
        self._lane_thr = lane_thr
        # Réutilise l'assignation de zone du module classique (composition).
        self._assigner = LaneEstimator(drive_side=drive_side)
        # Suivi temporel des lignes (mémoire entre frames) : lignes stables + longues.
        self._tracker = LaneLineTracker()
        # Corridor AFFICHÉ persistant : glisse vers la cible (serpente lors d'un
        # changement de voie) et survit aux pertes brèves -> AUCUNE interruption.
        self._cur_left = None
        self._cur_right = None
        self._cur_miss = 0
        self._curv_ema = None       # rayon de courbure lissé (anti « virage serré » bidon)
        # Profil APPRIS (persistant entre les runs) : largeur de voie + horizon
        # typiques, utilisés comme priors pour rejeter les conjectures aberrantes.
        self._profile = SceneProfile()
        # Détecteur de lignes DÉDIÉ (UFLD) optionnel : remplace les fenêtres glissantes.
        self._lane_detector = lane_detector
        # OPTIM : segmentation lourde 1 frame sur 2 (masques réutilisés entre deux).
        self._seg_stride = 2
        self._seg_i = 0
        self._seg_cache = None
        # Persistance du masque roulable : évite les « trous » (route qui disparaît) quand
        # une frame renvoie un masque dégénéré -> on conserve le dernier masque valide.
        self._last_drivable = None
        self._last_area = 0

    @property
    def on_gpu(self) -> bool:
        return self._on_gpu

    # ------------------------------------------------------------------ #
    def estimate(self, frame) -> LaneContext:
        img = frame.image
        h, w = img.shape[:2]

        # OPTIM GPU : la segmentation lourde (YOLOPv2) ne tourne qu'1 frame sur
        # `_seg_stride` — la route change lentement, on réutilise les masques entre
        # deux. Le reste (lignes/tracker/corridor) tourne à CHAQUE frame -> aucune
        # perte de qualité visible, ~2× moins de charge GPU sur la segmentation.
        self._seg_i += 1
        if self._seg_cache is not None and (self._seg_i % self._seg_stride) != 0:
            drivable, lanes = self._seg_cache
        else:
            blob = cv2.resize(img, (_INPUT, _INPUT), interpolation=cv2.INTER_LINEAR)
            blob = cv2.cvtColor(blob, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            blob = np.transpose(blob, (2, 0, 1))[None]
            outs = self._sess.run(None, {self._input: blob})
            drivable = np.argmax(outs[4][0], axis=0).astype(np.uint8)
            lanes = (outs[5][0, 0] > self._lane_thr).astype(np.uint8)
            drivable = cv2.resize(drivable, (w, h), cv2.INTER_NEAREST).astype(bool)
            lanes = cv2.resize(lanes, (w, h), cv2.INTER_NEAREST).astype(bool)
            lanes = self._remove_horizontal(lanes)
            # Persistance : si le masque roulable s'effondre brutalement (frame dégénérée),
            # on garde le dernier masque valide plutôt que d'afficher un trou.
            area = int(drivable.sum())
            if self._last_drivable is not None and area < 0.3 * self._last_area:
                drivable = self._last_drivable
            else:
                self._last_drivable, self._last_area = drivable, area
            self._seg_cache = (drivable, lanes)

        # Détection des lignes : réseau dédié UFLD si fourni, sinon fenêtres glissantes
        # classiques sur le masque YOLOPv2. Puis suivi temporel (stable + long).
        if self._lane_detector is not None:
            raw_lines = self._lane_detector.detect(frame)
        else:
            raw_lines = self._detect_lane_lines(lanes, h, w)
        lane_lines = self._tracker.update(raw_lines, h, w)
        ctx = self._build_context(lane_lines, drivable, lanes, h, w)
        ctx.drivable_mask = drivable
        ctx.lane_mask = lanes
        # Changement de voie : signal FIABLE du tracker (ligne suivie qui traverse le centre).
        ctx.lane_change = self._tracker.lane_change
        return ctx

    @staticmethod
    def _remove_horizontal(lanes: np.ndarray) -> np.ndarray:
        """Retire les longues structures horizontales (passages piétons, stop) par morpho."""
        m = lanes.astype(np.uint8)
        w = m.shape[1]
        kw = max(15, int(w * 0.03))            # largeur mini d'une structure "horizontale"
        hk = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, 1))
        horiz = cv2.morphologyEx(m, cv2.MORPH_OPEN, hk)
        return cv2.subtract(m, horiz).astype(bool)

    # ---------------- Détection des lignes individuelles ---------------- #
    def _detect_lane_lines(self, lanes: np.ndarray, h: int, w: int) -> list:
        """Sépare chaque ligne de marquage par fenêtres glissantes (sliding windows)."""
        roi_top = int(0.55 * h)
        # 1) Histogramme des colonnes sur le bas de l'image -> bases des lignes.
        #    Bande assez haute pour capter les lignes qui n'atteignent pas le tout-bas.
        band = lanes[int(0.68 * h):h, :].sum(axis=0).astype(np.float32)
        if band.max() < 3:
            return []
        k = max(3, int(w * 0.01) | 1)
        band = cv2.GaussianBlur(band.reshape(1, -1), (k, 1), 0).ravel()
        bases = self._peaks(band, thr=max(3.0, band.max() * 0.25),
                            min_dist=int(w * 0.05))

        # 2) Pour chaque base, remonter fenêtre par fenêtre (jusqu'assez haut).
        roi_top = int(0.50 * h)
        nwin = 14
        wh = max(1, (h - roi_top) // nwin)
        margin = int(w * 0.045)
        lines = []
        for base in bases:
            cx = float(base)
            vx = 0.0                      # tendance horizontale (suit la convergence)
            px, py = [], []
            for wi in range(nwin):
                y_hi = h - wi * wh
                y_lo = max(roi_top, y_hi - wh)
                x_lo = int(max(0, cx - margin)); x_hi = int(min(w, cx + margin))
                ys_i, xs_i = np.where(lanes[y_lo:y_hi, x_lo:x_hi])
                if xs_i.size > 6:
                    new_cx = x_lo + float(xs_i.mean())
                    if px:
                        vx = 0.5 * vx + 0.5 * (new_cx - cx)
                    cx = new_cx
                    px.append(cx); py.append((y_lo + y_hi) // 2)
                else:
                    cx = cx + vx          # traverse les trous (pointillés) via la tendance
            if len(px) >= 4:
                line = LaneLine.fit(py, px, degree=2, y_scale=h)
                # FILTRE DE SANTÉ (anti-artefacts) : rejette les segments trop courts
                # (stubs sans signification) et les courbures physiquement aberrantes.
                if (line is not None
                        and (line.y_hi - line.y_lo) >= 0.10 * h
                        and abs(line.a) * h * h <= 0.45 * w):
                    lines.append(line)
        # Ordonne gauche -> droite (par x en bas de l'image).
        lines.sort(key=lambda l: l.x_at(h - 1))
        # Fusionne les lignes trop proches (doublons).
        merged = []
        for l in lines:
            if merged and abs(l.x_at(h - 1) - merged[-1].x_at(h - 1)) < w * 0.04:
                continue
            merged.append(l)
        return merged

    @staticmethod
    def _peaks(sig: np.ndarray, thr: float, min_dist: int) -> list:
        idx = []
        i = 1
        n = len(sig)
        while i < n - 1:
            if sig[i] >= thr and sig[i] >= sig[i - 1] and sig[i] >= sig[i + 1]:
                if not idx or (i - idx[-1]) >= min_dist:
                    idx.append(i)
                elif sig[i] > sig[idx[-1]]:
                    idx[-1] = i
            i += 1
        return idx

    def _build_context(self, lines: list, drivable, lanes, h, w) -> LaneContext:
        """Voie ego = la PAIRE DE LIGNES CONSÉCUTIVES qui contient réellement l'ego, avec
        une largeur de voie plausible. On balaie toutes les voies (paires successives)
        plutôt que de prendre bêtement les 2 lignes autour du centre image (qui donnait
        souvent une paire trop étroite au centre -> mauvais cadrage).
        Si aucune paire fiable -> PAS de corridor (mieux que du faux)."""
        num = max(0, len(lines) - 1)
        ego_x = self._ego_x(drivable, w, h)
        # 1) CHOIX DE LA VOIE : parmi les paires consécutives (= voies), on garde celle
        #    dont la largeur est plausible et qui contient l'ego (sinon la plus proche).
        target_l = target_r = None
        ego_index = 0
        best = None
        for i in range(len(lines) - 1):
            a, b = lines[i], lines[i + 1]
            yb = min(a.y_hi, b.y_hi, h - 1)
            xa, xb = a.x_at(yb), b.x_at(yb)
            if xb - xa <= 1:
                continue
            width_frac = (xb - xa) / w
            if not (0.16 < width_frac < 0.75):   # une vraie voie n'est ni minuscule ni géante
                continue
            center = (xa + xb) / 2.0
            contains = (xa - 0.05 * w) <= ego_x <= (xb + 0.05 * w)
            score = (0 if contains else 1, abs(center - ego_x))
            if best is None or score < best[0]:
                best = (score, i, a, b)
        if best is not None:
            _, ego_index, target_l, target_r = best

        # 2) GLISSEMENT du corridor affiché vers la cible : lors d'un changement de
        # voie la cible saute d'une voie -> le corridor SERPENTE vers elle au lieu de
        # sauter ; cible perdue -> on garde le dernier corridor (pas d'interruption).
        if target_l is not None:
            if self._cur_left is None:
                self._cur_left, self._cur_right = target_l, target_r
            else:
                # Glissement ADAPTATIF : rapide en changement de voie (serpentage marqué
                # vers la nouvelle voie) ou en virage (courbure forte), doux sinon.
                turning = abs(target_l.a) + abs(target_r.a) > 1.5e-3
                a = 0.75 if (self._tracker.lane_change or turning) else 0.6
                self._cur_left = self._glide(self._cur_left, target_l, a)
                self._cur_right = self._glide(self._cur_right, target_r, a)
            self._cur_miss = 0
        else:
            self._cur_miss += 1
            if self._cur_miss > 45:            # ~3 s sans cible -> on efface enfin
                self._cur_left = self._cur_right = None

        if self._cur_left is not None:
            left, right = self._cur_left, self._cur_right
            yb = min(left.y_hi, right.y_hi, h - 1)
            xl_b, xr_b = left.x_at(yb), right.x_at(yb)
            lane_w = max(1.0, xr_b - xl_b)
            # Confiance : 1.0 tant qu'une cible est détectée ; décroît en « glissement »
            # aveugle (cible perdue) -> pilote l'opacité du tracé jaune.
            conf = max(0.15, 1.0 - self._cur_miss / 30.0)
            r_m, r_dir = lane_curvature(left, right, h, lane_w)
            r_m = self._smooth_curvature(r_m)
            return LaneContext(True, h, w, left, right,
                               center_x_bottom=(xl_b + xr_b) / 2.0,
                               lane_width_px=lane_w,
                               y_top=min(left.y_lo, right.y_lo),
                               lane_lines=lines, num_lanes=num,
                               ego_index=ego_index,
                               horizon_y=corridor_horizon(left, right, h) or 0.0,
                               corridor_confidence=conf,
                               curvature_radius_m=r_m, curvature_dir=r_dir)
        return LaneContext(False, h, w, lane_lines=lines, num_lanes=num)

    @staticmethod
    def _ego_x(drivable, w: int, h: int) -> float:
        """Position latérale de l'ego = centre de la zone roulable au bas de l'image
        (robuste au décalage caméra). Repli sur le centre image si pas de masque."""
        if drivable is None:
            return w / 2.0
        band = drivable[int(0.85 * h):, :]
        cols = np.where(band.any(axis=0))[0]
        if cols.size == 0:
            return w / 2.0
        # Médiane des colonnes actives -> centre robuste (insensible à un trottoir isolé).
        return float(np.median(cols))

    def _smooth_curvature(self, r_m):
        """Lisse fortement le rayon (les lignes courtes donnent un `a` très bruité)."""
        if r_m is None:
            self._curv_ema = None
            return None
        self._curv_ema = r_m if self._curv_ema is None else 0.15 * r_m + 0.85 * self._curv_ema
        return round(self._curv_ema, 0)

    @staticmethod
    def _glide(old: LaneLine, new: LaneLine, a: float = 0.3) -> LaneLine:
        """Interpolation vers la cible (y compris la plage verticale)."""
        return LaneLine(a * new.a + (1 - a) * old.a,
                        a * new.b + (1 - a) * old.b,
                        a * new.c + (1 - a) * old.c,
                        a * new.y_lo + (1 - a) * old.y_lo,
                        a * new.y_hi + (1 - a) * old.y_hi)

    def assign(self, world: WorldState, ctx: LaneContext) -> None:
        lines = getattr(ctx, "lane_lines", None)
        if not lines or len(lines) < 2:
            # Pas assez de lignes -> assignation par corridor (classique).
            self._assigner.assign(world, ctx)
            return
        # Assignation par INDEX de voie : on compte les lignes à gauche de l'objet.
        ego_i = ctx.ego_index
        opp = -1 if self._assigner._drive_side == "right" else 1
        for t in world.tracks:
            gx, gy = t.bbox.center[0], t.bbox.y2
            obj_i = sum(1 for l in lines if l.x_at(gy) < gx) - 1
            rel = obj_i - ego_i          # 0 = même voie que l'ego
            if rel == 0:
                t.lane_zone = LaneZone.EGO
            elif rel == -1:
                t.lane_zone = LaneZone.ADJACENT_LEFT
            elif rel == 1:
                t.lane_zone = LaneZone.ADJACENT_RIGHT
            elif (rel < 0) == (opp < 0):     # côté opposé au sens de circulation
                t.lane_zone = LaneZone.OPPOSITE
            else:
                t.lane_zone = (LaneZone.ADJACENT_LEFT if rel < 0
                               else LaneZone.ADJACENT_RIGHT)
