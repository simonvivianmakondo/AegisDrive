"""hello_carla — test de bout en bout de la connexion CARLA.

Prouve que le socle marche AVANT de brancher tout AegisDrive :
  1. se connecte au serveur CARLA,
  2. fait apparaître un véhicule ego + une caméra + du trafic,
  3. récupère quelques frames (mode synchrone),
  4. affiche la vérité terrain (distances réelles),
  5. sauve une image en PNG pour inspection visuelle.

Prérequis : serveur CARLA lancé (CarlaUE5.exe) + `pip install` du wheel client carla
dans le venv .venv-carla. À lancer depuis la racine du projet :

    .venv-carla\\Scripts\\python.exe hello_carla.py
    .venv-carla\\Scripts\\python.exe hello_carla.py --frames 40 --town Town10HD
"""
from __future__ import annotations

import argparse
import sys

import cv2

# permet d'importer le paquet sans installation (src/ layout)
sys.path.insert(0, "src")
from aegisdrive.io.carla_source import CarlaSource  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Test de connexion CARLA pour AegisDrive")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--frames", type=int, default=20, help="nb de frames à récupérer")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fov", type=float, default=90.0)
    ap.add_argument("--traffic", type=int, default=30, help="nb de véhicules PNJ")
    ap.add_argument("--town", default=None, help="carte à charger (ex. Town10HD)")
    ap.add_argument("--out", default="frame_extracted/carla_hello.png")
    args = ap.parse_args()

    print(f"→ Connexion à CARLA {args.host}:{args.port} …")
    try:
        src = CarlaSource(
            host=args.host, port=args.port,
            width=args.width, height=args.height, fov=args.fov,
            n_traffic=args.traffic, max_frames=args.frames, town=args.town,
        )
    except Exception as e:  # connexion / version / carte
        print(f"✗ Échec de connexion : {e}")
        print("  Vérifie que CarlaUE5.exe tourne et que la version client == serveur.")
        return 1

    print(f"✓ Connecté. Caméra {src.size[0]}x{src.size[1]} @ {src.fps:g} fps, FOV {src.fov:g}°",
          flush=True)
    got = 0
    # On écrit le PNG DANS la boucle (monde + caméra sains), et on ne rappelle JAMAIS
    # CARLA après l'arrêt des ticks : la lib native crashe sinon. Le nettoyage du serveur
    # est délégué à carla_reset.py (process frais) lancé à la fin. Voir carla_source.py.
    for frame in src.frames():
        got += 1
        if frame.index % 5 == 0:
            gt = src.ground_truth()
            nearest = min((g["distance_m"] for g in gt), default=float("nan"))
            print(f"  frame {frame.index:3d}  t={frame.timestamp:5.2f}s  "
                  f"image={frame.image.shape}  acteurs={len(gt):3d}  "
                  f"plus proche={nearest:5.1f} m", flush=True)
        if got >= args.frames:
            cv2.imwrite(args.out, frame.image)     # écrit pendant que tout est sain
            break

    ok = got >= args.frames
    print(f"✓ {got} frames reçues. Dernière image → {args.out}" if ok
          else "✗ Aucune frame reçue (le capteur n'a rien renvoyé).", flush=True)

    _reset_and_exit(0 if ok else 1)


def _reset_and_exit(code: int) -> None:
    """Lance le nettoyage serveur dans un PROCESS FRAIS, puis sortie dure.

    On ne peut pas nettoyer dans CE process (crash natif CARLA après les ticks). On
    délègue à carla_reset.py, on l'attend brièvement, puis os._exit court-circuite le
    destructeur natif qui planterait à l'extinction de Python.
    """
    import os
    import subprocess
    try:
        subprocess.run([sys.executable, "carla_reset.py", "--quiet"], timeout=40)
    except Exception:
        pass
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


if __name__ == "__main__":
    main()
