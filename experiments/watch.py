"""CLI de visualização: roda uma política aleatória com renderização.

Uso:
    poetry run python -m experiments.watch --config experiments/configs/env/synthetic_default.yaml

Requer um display disponível (Pygame). Constrói o ambiente BASE (sem wrappers de
RL) para acompanhar a simulação visualmente.
"""
from __future__ import annotations

import argparse

from experiments.config import build_raw_env, load_config


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Visualização do DengueDiagnostics")
    parser.add_argument("--config", required=True, help="Caminho para o YAML de ambiente")
    parser.add_argument("--steps", type=int, default=60, help="Número máximo de passos")
    parser.add_argument("--render-mode", default="human", help="human | console")
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)

    env = build_raw_env(config, render_mode=args.render_mode)
    env.reset(seed=args.seed)

    terminated = truncated = False
    steps = 0
    while not (terminated or truncated) and steps < args.steps:
        action = env.action_space.sample()
        _obs, _reward, terminated, truncated, _info = env.step(action)
        steps += 1

    close = getattr(env, "close", None)
    if callable(close):
        close()
    print(f"[watch] Simulação encerrada após {steps} passos.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
