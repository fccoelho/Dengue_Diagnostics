"""Assiste ao agente aleatório agindo, com renderização (fora do benchmark).

Uso (da raiz do repositório):
    poetry run python agents/random/watch.py
    poetry run python agents/random/watch.py --seed 100 --fps 8   # reproduzir surto
    poetry run python agents/random/watch.py --seed random        # surto aleatorio
    poetry run python agents/random/watch.py --config experiments/configs/env/synthetic_large.yaml

Abre a janela Pygame e mostra o agente decidindo caso a caso (mapa + gráficos de
recompensa e de acurácia multiclasse). Ao final, imprime um resumo do episódio.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Bootstrap: garante que a raiz do repo esteja no sys.path para `import agents`.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from agents.random.agent import RandomAgentRunner
from agents.watch import load_env_config

_DEFAULT_CONFIG = "experiments/configs/env/synthetic_default.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualizar o agente aleatório.")
    parser.add_argument("--config", default=_DEFAULT_CONFIG, help="YAML do ambiente.")
    parser.add_argument(
        "--seed",
        default="random",
        help="Seed do cenario (int) ou 'random' para surto diferente a cada execucao.",
    )
    parser.add_argument("--fps", type=int, default=10, help="Frames por segundo.")
    args = parser.parse_args()

    if args.seed == "random":
        seed = None
    else:
        seed = int(args.seed)

    env_config = load_env_config(args.config)
    RandomAgentRunner().watch(env_config, seed=seed, render_fps=args.fps)


if __name__ == "__main__":
    main()
