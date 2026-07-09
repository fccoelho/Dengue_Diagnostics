"""Assiste ao agente Q-Learning agindo (fora do benchmark).

Uso (da raiz do repositório):
    poetry run python agents/qlearning/watch.py --q-table results/qlearning/q_table.pkl
    poetry run python agents/qlearning/watch.py --seed 100 --fps 8

Requer Q-table treinada (`agents/qlearning/train.py`). Sem arquivo, usa tabela
vazia (comportamento aleatorio).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from agents.qlearning.agent import DEFAULT_CHECKPOINT, QLearningAgent
from agents.watch import load_env_config, run_watch

_DEFAULT_CONFIG = "experiments/configs/env/synthetic_default.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualizar agente Q-Learning.")
    parser.add_argument(
        "--q-table",
        default=str(DEFAULT_CHECKPOINT),
        help="Caminho do q_table.pkl treinado.",
    )
    parser.add_argument("--config", default=_DEFAULT_CONFIG, help="YAML do ambiente.")
    parser.add_argument(
        "--seed",
        default="random",
        help="Seed do cenario (int) ou 'random'.",
    )
    parser.add_argument("--fps", type=int, default=10, help="Frames por segundo.")
    parser.add_argument(
        "--state-version",
        default=None,
        choices=["rich_v1", "compact_v1"],
        help="Forcar versao de estado (default: ler do checkpoint).",
    )
    parser.add_argument(
        "--save-maps",
        action="store_true",
        help="Salvar mapa de confusao e da epidemia (default: nao salvar).",
    )
    parser.add_argument("--save-confusion-map", action="store_true")
    parser.add_argument("--save-epidemic-map", action="store_true")
    args = parser.parse_args()

    if args.seed == "random":
        seed = None
    else:
        seed = int(args.seed)

    q_path = Path(args.q_table)
    if q_path.exists():
        agent = QLearningAgent.load(q_path, epsilon=0.0)
        if args.state_version and agent.encoder.version != args.state_version:
            print(
                f"[qlearning] aviso: checkpoint={agent.encoder.version!r} "
                f"!= --state-version={args.state_version!r}"
            )
    else:
        print(
            f"[qlearning] aviso: Q-table ausente em {q_path.resolve()}. "
            "Usando tabela vazia."
        )
        agent = QLearningAgent(epsilon=0.0)

    env_config = load_env_config(args.config)

    def act_fn(obs, env) -> int:
        return agent.choose_action_from_env(env, explore=False)

    run_watch(
        env_config,
        act_fn=act_fn,
        agent_name="qlearning",
        seed=seed,
        render_fps=args.fps,
        save_confusion_map=args.save_maps or args.save_confusion_map,
        save_epidemic_map=args.save_maps or args.save_epidemic_map,
    )


if __name__ == "__main__":
    main()
