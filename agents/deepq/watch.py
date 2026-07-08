"""Assiste a uma policy DQN treinada agindo, com renderização.

Uso (da raiz do repositório):
    poetry run python agents/deepq/watch.py --policy caminho/para/policy.pth
    poetry run python agents/deepq/watch.py --policy policy.pth --seed 100 --fps 8

Carrega os pesos de uma `DQNPolicy` (Tianshou) e a executa no MESMO ambiente do
benchmark, com `render_mode="human"`. Útil para inspecionar visualmente o que um
checkpoint aprendeu, sem passar pelo treino/benchmark.

A rede (`DengueNet`) e os hiperparâmetros da policy espelham os do treino em
`agents/deepq/files`.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Bootstrap: raiz do repo + pasta `files` (onde vive `fcn_network`).
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from agents.watch import load_env_config, run_watch

_DEFAULT_CONFIG = "experiments/configs/env/synthetic_default.yaml"

# Hiperparâmetros da policy (devem casar com os do treino para carregar os pesos).
_DISCOUNT = 0.99
_N_STEP = 3
_TARGET_UPDATE_FREQ = 500


def _make_setup(policy_path: str, device: str):
    """Fábrica: recebe o env já criado e devolve a função de decisão da policy."""

    def setup(env):
        import torch
        from tianshou.data import Batch
        from tianshou.policy import DQNPolicy

        from agents.deepq.files.fcn_network import DengueNet

        map_shape = env.observation_space.spaces["map"].shape
        action_shape = env.action_space.n

        net = DengueNet(map_shape, action_shape, device=device).to(device)
        policy = DQNPolicy(
            model=net,
            optim=None,
            discount_factor=_DISCOUNT,
            action_space=env.action_space,
            estimation_step=_N_STEP,
            target_update_freq=_TARGET_UPDATE_FREQ,
        )

        path = Path(policy_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Policy não encontrada: {path}. "
                "Treine um DQN e aponte --policy para o .pth gerado."
            )
        policy.load_state_dict(
            torch.load(str(path), map_location=device, weights_only=True)
        )
        policy.eval()
        policy.set_eps(0.0)  # avaliação determinística (sem exploração)
        print(f"[watch] policy carregada de: {path}")

        def act_fn(obs, _env) -> int:
            batch = Batch(obs=[obs], info={})
            return int(policy(batch).act[0].item())

        return act_fn

    return setup


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualizar uma policy DQN treinada.")
    parser.add_argument("--policy", required=True, help="Caminho do .pth da policy.")
    parser.add_argument("--config", default=_DEFAULT_CONFIG, help="YAML do ambiente.")
    parser.add_argument(
        "--seed",
        default="random",
        help="Seed do cenario (int) ou 'random'.",
    )
    parser.add_argument("--fps", type=int, default=10, help="Frames por segundo.")
    parser.add_argument(
        "--no-confusion-map",
        action="store_true",
        help="Nao salvar mapa de confusao ao final.",
    )
    parser.add_argument(
        "--no-epidemic-map",
        action="store_true",
        help="Nao salvar mapa da epidemia (ground truth) ao iniciar.",
    )
    parser.add_argument(
        "--device", default=None, help="cpu ou cuda (default: cuda se disponivel)."
    )
    args = parser.parse_args()

    device = args.device
    if device is None:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.seed == "random":
        seed = None
    else:
        seed = int(args.seed)

    env_config = load_env_config(args.config)
    run_watch(
        env_config,
        setup=_make_setup(args.policy, device),
        agent_name="dqn",
        seed=seed,
        render_fps=args.fps,
        save_confusion_map=not args.no_confusion_map,
        save_epidemic_map=not args.no_epidemic_map,
    )


if __name__ == "__main__":
    main()
