"""Treina o agente tabular Q-Learning no ambiente novo.

Uso (da raiz do repositório):
    poetry run python agents/qlearning/train.py
    poetry run python agents/qlearning/train.py --config experiments/configs/train/qlearning_default.yaml
    poetry run python agents/qlearning/train.py --episodes 50 --output-dir results/qlearning_test

O script NÃO roda automaticamente — execute quando quiser treinar.
Ao final, salva `q_table.pkl` em `output_dir` (default `results/qlearning/`).
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from agents.qlearning.agent import QLearningAgent
from agents.qlearning.state import StateEncoder
from dengue_envs.wrappers import make_env

_DEFAULT_CONFIG = "experiments/configs/train/qlearning_default.yaml"


def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _epsilon_for_episode(ep: int, start: float, final: float, decay_eps: int) -> float:
    if decay_eps <= 0:
        return final
    frac = min(ep / decay_eps, 1.0)
    return start + (final - start) * frac


def train(config_path: str) -> Path:
    config_path = Path(config_path).resolve()
    cfg = _load_yaml(config_path)

    env_config_path = (config_path.parent / cfg["env_config"]).resolve()
    env_config = _load_yaml(env_config_path)
    train_cfg = cfg.get("train", {})

    episodes = int(train_cfg.get("episodes", 200))
    seed = int(train_cfg.get("seed", 42))
    alpha = float(train_cfg.get("alpha", 0.5))
    gamma = float(train_cfg.get("gamma", 0.5))
    eps_start = float(train_cfg.get("epsilon_start", 0.30))
    eps_final = float(train_cfg.get("epsilon_final", 0.05))
    eps_decay = int(train_cfg.get("epsilon_decay_episodes", 150))
    day_bucket_size = int(train_cfg.get("day_bucket_size", 5))
    save_every = int(train_cfg.get("save_every", 25))
    log_every = int(train_cfg.get("log_every", 10))
    state_cfg = cfg.get("state", train_cfg.get("state", {}))
    encoder = StateEncoder.from_config(state_cfg)

    output_dir = Path(cfg.get("output_dir", "results/qlearning"))
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "q_table.pkl"
    log_path = output_dir / "training_log.csv"

    loaded = _try_load(checkpoint, encoder=encoder)
    if loaded is not None and loaded.encoder.version != encoder.version:
        print(
            f"[qlearning] aviso: checkpoint usa estado {loaded.encoder.version!r}, "
            f"config pede {encoder.version!r}; reiniciando Q-table."
        )
        loaded = None
    agent = QLearningAgent(
        alpha=alpha,
        gamma=gamma,
        epsilon=eps_start,
        encoder=encoder,
        day_bucket_size=day_bucket_size,
        q_table=loaded.q_table if loaded else None,
    )

    rng = np.random.default_rng(seed)
    env = make_env(env_config)

    print(f"[qlearning] treinando {episodes} episodios | env={env_config_path.name}")
    print(f"[qlearning] estado={encoder.version} | checkpoint -> {checkpoint.resolve()}")

    log_rows = []
    for ep in range(1, episodes + 1):
        ep_seed = int(rng.integers(0, 2**31))
        agent.epsilon = _epsilon_for_episode(ep - 1, eps_start, eps_final, eps_decay)

        obs, _ = env.reset(seed=ep_seed)
        terminated = truncated = False
        total_reward = 0.0
        steps = 0

        state = agent.encode_env(env)

        while not (terminated or truncated):
            action = agent.choose_action(state, explore=True)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            next_state = agent.encode_env(env)
            agent.update(state, action, float(reward), next_state)

            total_reward += float(reward)
            steps += 1
            state = next_state
            obs = next_obs

        metrics = env.unwrapped.get_episode_metrics()
        env_reward = float(metrics.get("Recompensa Total", total_reward))
        log_rows.append(
            {
                "episode": ep,
                "seed": ep_seed,
                "epsilon": agent.epsilon,
                "steps": steps,
                "reward_accumulated": total_reward,
                "reward_env": env_reward,
                "states_seen": len(agent.q_table),
            }
        )

        if ep % log_every == 0 or ep == 1 or ep == episodes:
            print(
                f"[qlearning] ep {ep:4d}/{episodes} | "
                f"eps={agent.epsilon:.3f} | reward={env_reward:8.1f} | "
                f"states={len(agent.q_table)}"
            )

        if ep % save_every == 0 or ep == episodes:
            agent.save(checkpoint, also_txt=(ep == episodes))

    env.close()

    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
        writer.writeheader()
        writer.writerows(log_rows)

    print(f"[qlearning] treino concluido. Q-table: {checkpoint.resolve()}")
    print(f"[qlearning] log: {log_path.resolve()}")
    return checkpoint


def _try_load(path: Path, encoder: StateEncoder) -> QLearningAgent | None:
    if not path.exists():
        return None
    try:
        return QLearningAgent.load(path, epsilon=0.0, encoder=encoder)
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Treinar Q-Learning tabular.")
    parser.add_argument("--config", default=_DEFAULT_CONFIG, help="YAML de treino.")
    parser.add_argument("--episodes", type=int, default=None, help="Sobrescreve episodes.")
    parser.add_argument("--output-dir", default=None, help="Sobrescreve output_dir.")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    if args.episodes is not None or args.output_dir is not None:
        cfg = _load_yaml(config_path)
        if args.episodes is not None:
            cfg.setdefault("train", {})["episodes"] = args.episodes
        if args.output_dir is not None:
            cfg["output_dir"] = args.output_dir
        tmp = config_path.parent / "_qlearning_cli_override.yaml"
        with open(tmp, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f)
        config_path = tmp

    train(str(config_path))


if __name__ == "__main__":
    main()
