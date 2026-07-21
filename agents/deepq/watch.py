"""Assiste a uma policy DQN treinada agindo, com renderização.

Uso (da raiz do repositório):
    poetry run python agents/deepq/watch.py
    poetry run python agents/deepq/watch.py --policy results/dqn/policy_best.pth --seed 100
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from agents.deepq.agent import DEFAULT_CHECKPOINT, load_policy
from agents.watch import load_env_config, run_watch
from tianshou.data import Batch

_DEFAULT_CONFIG = "experiments/configs/env/synthetic_default.yaml"


def _make_setup(policy_path: str, device: str):
    def setup(env):
        policy = load_policy(policy_path, env, device=device)
        print(f"[watch] policy carregada de: {policy_path}")

        def act_fn(obs, _env) -> int:
            batch = Batch(obs=[obs], info={})
            return int(policy(batch).act[0])

        return act_fn

    return setup


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualizar uma policy DQN treinada.")
    parser.add_argument(
        "--policy",
        default=str(DEFAULT_CHECKPOINT),
        help=f"Caminho do .pth (default: {DEFAULT_CHECKPOINT}).",
    )
    parser.add_argument("--config", default=_DEFAULT_CONFIG, help="YAML do ambiente.")
    parser.add_argument(
        "--seed",
        default="random",
        help="Seed do cenario (int) ou 'random'.",
    )
    parser.add_argument("--fps", type=int, default=10, help="Frames por segundo.")
    parser.add_argument(
        "--save-maps",
        action="store_true",
        help="Salvar mapa de confusao e da epidemia.",
    )
    parser.add_argument(
        "--save-confusion-map",
        action="store_true",
        help="Salvar mapa de confusao ao final.",
    )
    parser.add_argument(
        "--save-epidemic-map",
        action="store_true",
        help="Salvar mapa da epidemia ao iniciar.",
    )
    parser.add_argument(
        "--device", default=None, help="cpu ou cuda (default: cuda se disponivel)."
    )
    args = parser.parse_args()

    device = args.device
    if device is None:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"

    seed = None if args.seed == "random" else int(args.seed)

    env_config = load_env_config(args.config)
    run_watch(
        env_config,
        setup=_make_setup(args.policy, device),
        agent_name="dqn",
        seed=seed,
        render_fps=args.fps,
        save_confusion_map=args.save_maps or args.save_confusion_map,
        save_epidemic_map=args.save_maps or args.save_epidemic_map,
    )


if __name__ == "__main__":
    main()
