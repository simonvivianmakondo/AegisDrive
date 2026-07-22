"""Rapport automatique de fin de vidéo — Phase 5.

Relit le log `replay.jsonl` (un WorldState par ligne) en STREAMING et en agrège des
statistiques de scène. Aucune dépendance au reste du pipeline : le rapport est un pur
lecteur du log, donc rejouable/exportable à volonté (même philosophie que le replay).

Sortie : un dictionnaire JSON-sérialisable + un résumé texte lisible.
"""
from __future__ import annotations

import json

_DANGER_THRESHOLD = 70.0   # seuil "situation dangereuse"
_VEHICLES = {"car", "truck", "bus", "motorcycle"}


def build_report(log_path: str, extra: dict | None = None) -> dict:
    """Agrège les statistiques du log en une passe (streaming)."""
    frames = 0
    last_ts = 0.0
    id_to_class: dict[int, str] = {}          # objets uniques -> catégorie
    min_distance = None
    ttc_values: list[float] = []
    dangerous_ids: set[int] = set()           # tracks ayant atteint le seuil
    dangerous_frames = 0
    behavior_ids: dict[str, set] = {}         # comportement -> ids distincts
    condition_counts: dict[str, int] = {}

    with open(log_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            frames += 1
            last_ts = d.get("timestamp", last_ts)

            cond = d.get("conditions")
            if cond:
                condition_counts[cond["label"]] = condition_counts.get(cond["label"], 0) + 1

            frame_has_danger = False
            for t in d.get("tracks", []):
                id_to_class.setdefault(t["id"], t["cls"])
                if t.get("distance_m") is not None:
                    if min_distance is None or t["distance_m"] < min_distance:
                        min_distance = t["distance_m"]
                if t.get("ttc_s") is not None:
                    ttc_values.append(t["ttc_s"])
                if t.get("danger_score", 0) >= _DANGER_THRESHOLD:
                    dangerous_ids.add(t["id"])
                    frame_has_danger = True
                for b in t.get("behaviors", []):
                    behavior_ids.setdefault(b, set()).add(t["id"])
            if frame_has_danger:
                dangerous_frames += 1

    # Répartition par catégorie (objets uniques).
    by_class: dict[str, int] = {}
    for cls in id_to_class.values():
        by_class[cls] = by_class.get(cls, 0) + 1
    n_pedestrians = by_class.get("pedestrian", 0)
    n_vehicles = sum(v for k, v in by_class.items() if k in _VEHICLES)

    predominant_cond = (max(condition_counts, key=condition_counts.get)
                        if condition_counts else None)

    report = {
        "duree_s": round(last_ts, 1),
        "frames_analysees": frames,
        "objets_uniques": len(id_to_class),
        "objets_par_categorie": dict(sorted(by_class.items(), key=lambda x: -x[1])),
        "pietons": n_pedestrians,
        "vehicules": n_vehicles,
        "distance_min_m": min_distance,
        "ttc_moyen_s": round(sum(ttc_values) / len(ttc_values), 2) if ttc_values else None,
        "ttc_min_s": round(min(ttc_values), 2) if ttc_values else None,
        "situations_dangereuses": len(dangerous_ids),
        "frames_dangereuses": dangerous_frames,
        "comportements": {k: len(v) for k, v in
                          sorted(behavior_ids.items(), key=lambda x: -len(x[1]))},
        "condition_dominante": predominant_cond,
    }
    if extra:
        report.update(extra)
    return report


def format_text(r: dict) -> str:
    """Résumé lisible (console / fichier .txt)."""
    L = ["==================  RAPPORT AEGISDRIVE  =================="]
    L.append(f"Durée analysée      : {r['duree_s']} s  ({r['frames_analysees']} frames)")
    if r.get("processing_fps"):
        L.append(f"Vitesse de calcul   : {r['processing_fps']:.1f} FPS "
                 f"({r.get('elapsed_s', 0):.0f}s de traitement)")
    if r.get("condition_dominante"):
        L.append(f"Conditions          : {r['condition_dominante']}")
    L.append("")
    L.append(f"Objets détectés     : {r['objets_uniques']}  "
             f"(véhicules: {r['vehicules']}, piétons: {r['pietons']})")
    for cls, n in r["objets_par_categorie"].items():
        L.append(f"    - {cls:12} {n}")
    L.append("")
    L.append(f"Distance minimale   : {r['distance_min_m']} m")
    L.append(f"TTC moyen / min     : {r['ttc_moyen_s']} s / {r['ttc_min_s']} s")
    L.append(f"Situations dangereuses : {r['situations_dangereuses']} objets  "
             f"({r['frames_dangereuses']} frames)")
    if r["comportements"]:
        L.append("")
        L.append("Comportements observés (objets distincts) :")
        for b, n in r["comportements"].items():
            L.append(f"    - {b:20} {n}")
    L.append("=========================================================")
    return "\n".join(L)


def generate(log_path: str, json_path: str, extra: dict | None = None) -> dict:
    """Construit le rapport, l'écrit en JSON + .txt, et renvoie le dict."""
    r = build_report(log_path, extra)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(r, f, ensure_ascii=False, indent=2)
    txt_path = json_path.rsplit(".", 1)[0] + ".txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(format_text(r))
    return r
