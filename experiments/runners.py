"""Runners de algoritmos acionados por configuração (`experiments/train.py`).

O path canônico de treino do DQN é ``agents/deepq/train.py``. Este módulo
mantém um despacho por nome para o CLI `experiments.train` e o baseline random.
"""
from __future__ import annotations

import tempfile
from typing import Any, Callable, Dict, Protocol

import numpy as np
import yaml

from experiments.config import build_env_factory, get_train_config


class AgentRunner(Protocol):
    name: str

    def run(self, env_factory: Callable[[], Any], config: Dict[str, Any]) -> Dict[str, Any]:
        ...


class RandomRunner:
    """Executa uma política aleatória por N episódios e reporta a recompensa."""

    name = "random"

    def run(self, env_factory: Callable[[], Any], config: Dict[str, Any]) -> Dict[str, Any]:
        train = get_train_config(config)
        episodes = int(train.get("episodes", 5))
        seed = train.get("seed", None)

        env = env_factory()
        rewards = []
        try:
            for ep in range(episodes):
                ep_seed = None if seed is None else int(seed) + ep
                _obs, _info = env.reset(seed=ep_seed)
                terminated = truncated = False
                total = 0.0
                steps = 0
                max_steps = int(train.get("max_steps", 100000))
                while not (terminated or truncated) and steps < max_steps:
                    action = env.action_space.sample()
                    _obs, reward, terminated, truncated, _info = env.step(action)
                    total += float(reward)
                    steps += 1
                rewards.append(total)
        finally:
            close = getattr(env, "close", None)
            if callable(close):
                close()

        rewards_arr = np.asarray(rewards, dtype=float)
        return {
            "algorithm": self.name,
            "episodes": episodes,
            "rewards": rewards,
            "mean_reward": float(rewards_arr.mean()) if len(rewards_arr) else 0.0,
            "std_reward": float(rewards_arr.std()) if len(rewards_arr) else 0.0,
        }


class DQNRunner:
    """Delega ao treino canônico ``agents.deepq.train`` (Tianshou + YAML)."""

    name = "dqn"

    def run(self, env_factory: Callable[[], Any], config: Dict[str, Any]) -> Dict[str, Any]:
        del env_factory  # o treino reconstrói o env a partir do YAML
        from agents.deepq.train import train

        train_cfg = dict(get_train_config(config))
        payload = {
            "env": dict(config.get("env", {})),
            "wrappers": config.get("wrappers", ["map_tensor", "case_by_case"]),
            "train": train_cfg,
            "output_dir": train_cfg.get(
                "output_dir", config.get("output_dir", "results/dqn")
            ),
        }
        with tempfile.NamedTemporaryFile(
            "w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as fh:
            yaml.safe_dump(payload, fh, allow_unicode=True)
            tmp_path = fh.name
        checkpoint = train(tmp_path)
        return {
            "algorithm": self.name,
            "checkpoint": str(checkpoint),
            "output_dir": str(payload["output_dir"]),
        }


class _NotImplementedRunner:
    """Placeholder para algoritmos ainda não fiados ao pipeline."""

    def __init__(self, name: str, hint: str):
        self.name = name
        self._hint = hint

    def run(self, env_factory: Callable[[], Any], config: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError(
            f"O runner '{self.name}' ainda não está integrado ao pipeline de "
            f"configuração. {self._hint}"
        )


_REGISTRY: Dict[str, Callable[[], AgentRunner]] = {
    "random": RandomRunner,
    "dqn": DQNRunner,
    "ppo": lambda: _NotImplementedRunner(
        "ppo",
        "Adaptar agents/ppo ao CaseByCaseWrapper (mesmo contrato do DQN).",
    ),
}


def available_algorithms() -> list:
    return sorted(_REGISTRY.keys())


def get_runner(algorithm: str) -> AgentRunner:
    if algorithm not in _REGISTRY:
        raise ValueError(
            f"Algoritmo desconhecido: {algorithm!r}. "
            f"Disponíveis: {available_algorithms()}"
        )
    return _REGISTRY[algorithm]()


def run_from_config(config: Dict[str, Any]) -> Dict[str, Any]:
    train = get_train_config(config)
    runner = get_runner(train["algorithm"])
    env_factory = build_env_factory(config)
    return runner.run(env_factory, config)
