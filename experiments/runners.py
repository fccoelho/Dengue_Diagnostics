"""Runners de algoritmos acionados por configuração.

Define um contrato comum (`AgentRunner`) e um registro para despachar por nome
de algoritmo. Nesta fase, o `RandomRunner` está totalmente funcional (serve de
baseline e prova o pipeline YAML -> env -> execução). Os runners de DQN e PPO
são stubs explícitos: a integração real será feita na Fase 3, reaproveitando os
scripts em `agents/`.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Protocol

import numpy as np

from experiments.config import build_env_factory, get_train_config


class AgentRunner(Protocol):
    name: str

    def run(self, env_factory: Callable[[], Any], config: Dict[str, Any]) -> Dict[str, Any]:
        ...


class RandomRunner:
    """Executa uma política aleatória por N episódios e reporta a recompensa.

    Não "treina" (não há parâmetros), mas fecha o ciclo config -> env -> métrica,
    servindo de baseline reprodutível.
    """

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
                # Salvaguarda contra episódios longos/infinitos.
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


class _NotImplementedRunner:
    """Placeholder para algoritmos ainda não fiados ao pipeline (Fase 3)."""

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
    "dqn": lambda: _NotImplementedRunner(
        "dqn",
        "Integração prevista para a Fase 3 (unificar agents/deepq em agents/dqn/train.py).",
    ),
    "ppo": lambda: _NotImplementedRunner(
        "ppo",
        "Integração prevista para a Fase 3 (adaptar agents/ppo ao CaseByCaseWrapper).",
    ),
}


def available_algorithms() -> list:
    """Lista os algoritmos registrados."""
    return sorted(_REGISTRY.keys())


def get_runner(algorithm: str) -> AgentRunner:
    """Retorna uma instância de runner para o algoritmo informado."""
    if algorithm not in _REGISTRY:
        raise ValueError(
            f"Algoritmo desconhecido: {algorithm!r}. "
            f"Disponíveis: {available_algorithms()}"
        )
    return _REGISTRY[algorithm]()


def run_from_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Constrói o ambiente pela config e executa o algoritmo escolhido."""
    train = get_train_config(config)
    runner = get_runner(train["algorithm"])
    env_factory = build_env_factory(config)
    return runner.run(env_factory, config)
