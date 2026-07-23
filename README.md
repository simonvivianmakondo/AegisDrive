<h1 align="center">🛡️ AegisDrive</h1>

<p align="center">
  <b>Perception routière embarquée par IA</b> — d'une vidéo de conduite à une analyse
  de scène complète : détection, tracking, distance/vitesse, TTC et
  <b>score de danger explicable</b>.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white"/>
  <img src="https://img.shields.io/badge/YOLO-00FFFF?style=for-the-badge&logo=&logoColor=black"/>
  <img src="https://img.shields.io/badge/statut-en%20développement-orange?style=for-the-badge"/>
</p>

> ⚠️ **Le système observe et analyse. Il ne pilote jamais le véhicule.**
> AegisDrive est un outil de **perception passive** (analyse hors-ligne et, à terme, temps réel embarqué), pas un système de contrôle.

---

## ✨ Ce que ça fait

À partir d'une simple vidéo de conduite (dashcam, smartphone…), AegisDrive produit une **vidéo annotée** + un **log rejouable** + un **rapport de synthèse** :

- 🎯 **Détection d'objets** — véhicules, piétons, feux, camions… (YOLO, ou `FakeDetector` pour tests/CI)
- 🔗 **Tracking multi-objets** — identités stables dans le temps (filtre de **Kalman** ou IoU simple)
- 🛣️ **Détection de voies** — voie ego et lignes (Hough classique, segmentation **YOLOPv2**, ou réseau dédié **UFLD**)
- 📏 **Cinématique** — distance (modèle pinhole + FOV caméra), vitesse relative, **mouvement propre** (ego-motion)
- ⏱️ **Time-To-Collision (TTC)** et **score de danger explicable** (moteur de règles, pas une boîte noire)
- 🚦 **Analyse de comportements** — freinage d'urgence, changement de voie, dépassement, *cut-in*, piéton traversant…
- 🌗 **Conditions de scène** — jour/nuit/météo, avec rehaussement optionnel de l'image
- 📊 **Dashboard** intégré (panneau latéral) + **rapport** de fin (`.json` + `.txt`)

---

## 🧱 Principe d'architecture

On ne code pas un module contre un autre — on code contre des **schémas de données**
(`schemas.py`) et des **interfaces** (`interfaces.py` : `Source`, `Detector`,
`Tracker`, `RiskEngine`, `Sink`…). Chaque étage est **remplaçable** sans toucher au reste.

```
Frame
  └─▶ Detection[]        (perception : yolo | fake)
      └─▶ Track[]        (tracking : kalman | iou)
          └─▶ WorldState (cinématique + ego-motion + voies + comportements)
              └─▶ RiskAssessment (TTC + score de danger explicable)
                  └─▶ vidéo annotée + replay.jsonl + rapport
```

Le **même code** sert **deux modes** :
- **Vidéo** (objectif principal) — étages chaînés séquentiellement, déterministe.
- **Temps réel** (futur, cible **Jetson**) — mêmes étages, un thread chacun, reliés par des files.

---

## 🚀 Démarrage rapide

```bash
python -m venv .venv && .venv\Scripts\activate      # Windows
pip install -r requirements.txt

# Analyse de base (FakeDetector — aucun modèle requis, idéal pour tester le pipeline)
python -m aegisdrive.main --input input.mp4 --output output.mp4

# Analyse réelle avec YOLO (nécessite `pip install ultralytics`)
python -m aegisdrive.main --input input.mp4 --detector yolo --tracker kalman
```

Sans YOLO installé, le pipeline tourne avec un **`FakeDetector`** (utile pour les tests et la CI, sans GPU ni modèle).

> 📦 Les **poids de modèles** (`.pt`, `.onnx`), les **vidéos** et les **sorties générées**
> (log, rapport, frames) ne sont **pas** versionnés — voir `.gitignore`. Télécharge/place
> tes modèles dans `models/` selon les chemins par défaut ci-dessous.

---

## 🕹️ Mode simulateur (CARLA)

La **même perception** peut analyser un monde 3D vivant au lieu d'une vidéo figée, via le
simulateur **[CARLA](https://carla.org/) 0.10** (Unreal Engine 5). On remplace simplement
la source : `--source carla`. Le reste du pipeline (détection → tracking → risque) est
**inchangé**.

```bash
# 1) lancer le serveur CARLA (fenêtre 3D)
CarlaUnreal.exe -carla-rpc-port=2000

# 2) faire tourner AegisDrive sur le simulateur (YOLO + voies carte)
python -m aegisdrive.main --source carla --detector yolo --carla-frames 700 --output out.mp4
```

Points clés de l'intégration :

- **`CarlaSource`** — une caméra virtuelle embarquée exposée via l'interface `Source`
  (mode synchrone → déterministe), + la **vérité terrain** du simulateur (positions
  réelles, projetées dans l'image) pour *valider* la perception.
- **`CarlaLaneProvider`** — les voies sont lues dans la **carte HD** du simulateur
  (waypoints), pas détectées à l'image : sur le rendu synthétique de CARLA les réseaux de
  voies (entraînés sur du réel) échouent. Résultat : corridor + lignes exacts, assignation
  de voie ~90 %, et infos de conduite (virage, voies adjacentes, changement autorisé).
- **Outils d'évaluation** : `carla_eval.py` (perception vs vérité terrain — détection,
  distance, TTC), `carla_calib.py` (calibration distance en scénario contrôlé — **biais
  < 5 %**), `carla_reset.py` (remise à zéro du serveur), `hello_carla.py` (test de
  connexion).

> ⚠️ La lib native CARLA plante à l'extinction de Python (mode synchrone) : les scripts
> terminent par `os._exit()` + un reset serveur dans un process séparé. Détail dans
> `io/carla_source.py`.

---

## ⚙️ Options principales (CLI)

| Option | Défaut | Rôle |
|---|---|---|
| `--source` | `video` | Source des frames : `video` (mp4) \| `carla` (simulateur) |
| `--input` | — | Vidéo d'entrée (`.mp4`) — **requis si `--source video`** |
| `--output` | `output.mp4` | Vidéo annotée de sortie |
| `--detector` | `fake` | `fake` \| `yolo` |
| `--tracker` | `kalman` | `kalman` \| `iou` |
| `--road` | `classic` | Voies : `classic` (Hough) \| `seg` (YOLOPv2) \| `carla` (waypoints — défaut auto en `--source carla`) |
| `--lanes-model` | `auto` | Lignes : `auto` \| `classic` \| `ufld` (réseau dédié) |
| `--fov` | `60` | Champ de vision horizontal caméra (°) — sert au calcul de distance |
| `--cam-height` | `1.2` | Hauteur caméra (m) — échelle de la vitesse ego |
| `--proc-width` | `960` | Largeur de traitement (réduit 1080p/4K → gros gain de vitesse) |
| `--frame-stride` | `1` | Analyse 1 frame sur N |
| `--start` / `--end` | `0` / fin | Analyser seulement un segment (secondes) |
| `--report` / `--no-report` | `report.json` | Rapport de fin (JSON + `.txt`) |

**Options CARLA** (avec `--source carla`) : `--carla-frames` (nb de frames), `--carla-traffic`
(véhicules PNJ), `--carla-town` (carte), `--carla-seed` (varie la route), `--carla-host`/`--carla-port`.

Exemples : `--input drive.mp4 --detector yolo --road seg --start 50 --end 90`
· `--source carla --detector yolo --carla-frames 1400 --carla-seed 2`

---

## 📋 Exemple de rapport

```
==================  RAPPORT AEGISDRIVE  ==================
Durée analysée      : 150.0 s  (3000 frames)
Vitesse de calcul   : 10.7 FPS
Conditions          : jour clair

Objets détectés     : 420  (véhicules: 344, piétons: 20)
Distance minimale   : 2.7 m
TTC moyen / min     : 7.32 s / 0.39 s
Situations dangereuses : 108 objets  (816 frames)

Comportements observés : hard_braking, lane_change, overtaking,
                         cut_in, ped_crossing, stopped, ...
=========================================================
```

---

## 🗺️ Feuille de route

| Phase | Contenu | État |
|-------|---------|------|
| 1 | Lecture → détection → tracking → risque → vidéo annotée + log JSONL | ✅ |
| 2 | Kalman + distance (pinhole) + vitesse relative | ✅ |
| 3 | TTC + score de danger explicable | ✅ |
| 4 | Voies (UFLD / YOLOPv2) + ego-motion + comportements | ✅ |
| 5 | Rapport de synthèse (relit le log JSONL) | ✅ |
| 6 | **Intégration CARLA** : source simulateur + voies carte + validation vs vérité terrain | ✅ |
| 7 | Profondeur mono (Depth Anything) + vue BEV | 🔜 |
| 8 | Contrôle en boucle fermée (suivi de voie, virages, freinage) puis agent RL | 🔜 |
| 9 | Multi-cam, optim GPU (TensorRT), temps réel passif embarqué (Jetson) | 🔜 |

---

## 📁 Structure

```
src/aegisdrive/
  schemas.py        contrats de données (source de vérité)
  interfaces.py     Protocols : Source, Detector, Tracker, RiskEngine, Sink
  io/               sources/sinks : vidéo (OpenCV) + CARLA (carla_source.py)
  perception/       détecteurs (fake, yolo)
  tracking/         suivi (IoU simple, Kalman)
  estimation/       cinématique (distance, vitesse)
  ego/              estimation du mouvement propre (ego-motion)
  scene/            voies & route (Hough, YOLOPv2, UFLD, carla_lanes.py)
  understanding/    états d'objets & moteur de comportements
  risk/             moteur de règles explicable (TTC, danger)
  preprocess/       conditions de scène (jour/nuit/météo)
  analytics/        rapport de fin (JSON + texte)
  viz/              annotation des frames + dashboard
  pipeline/         orchestrateur (mode vidéo ; temps réel plus tard)
  main.py           input.mp4 / CARLA → output.mp4 + log + rapport
hello_carla.py      test de connexion CARLA (caméra + vérité terrain)
carla_eval.py       évaluation perception vs vérité terrain CARLA
carla_calib.py      calibration distance (scénario contrôlé)
carla_reset.py      remise à zéro du serveur CARLA
tests/              tests unitaires (tournent sans GPU ni modèle)
```

---

<p align="center">
  <sub>Projet personnel — <a href="https://github.com/simonvivianmakondo">Simon Vivian Makondo</a></sub>
</p>
