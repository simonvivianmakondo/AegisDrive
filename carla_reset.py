"""carla_reset — remet le serveur CARLA dans un état propre.

Détruit tous les véhicules / piétons / capteurs et repasse le monde en mode
ASYNCHRONE (sinon, sans client pour le faire avancer, la simulation reste figée).

À exécuter dans un PROCESS FRAIS : c'est la seule façon fiable de dialoguer avec le
serveur après un run synchrone (la lib native crashe si on tente ce nettoyage dans
le process qui vient de piloter les ticks — voir carla_source.py). Les entrypoints
l'appellent automatiquement en sous-processus avant de sortir.

    .venv-carla\\Scripts\\python.exe carla_reset.py
"""
from __future__ import annotations

import argparse
import os
import sys

import carla

_MOBILE = ("vehicle.", "walker.", "sensor.")


def main() -> int:
    ap = argparse.ArgumentParser(description="Réinitialise le serveur CARLA")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    def log(m):
        if not args.quiet:
            print(m, flush=True)

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(30.0)
        world = client.get_world()
        # async d'abord : le serveur redevient autonome
        s = world.get_settings()
        s.synchronous_mode = False
        s.fixed_delta_seconds = None
        world.apply_settings(s)
        try:
            client.get_trafficmanager().set_synchronous_mode(False)
        except Exception:
            pass
        # puis destruction des acteurs mobiles
        kill = [a for a in world.get_actors() if a.type_id.startswith(_MOBILE)]
        client.apply_batch([carla.command.DestroyActor(a.id) for a in kill])
        log(f"✓ reset : {len(kill)} acteurs détruits, mode asynchrone restauré")
    except Exception as e:
        log(f"✗ reset impossible : {e}")
        sys.stdout.flush()
        os._exit(1)

    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
