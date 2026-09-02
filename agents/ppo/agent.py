"""Runner de benchmark do PPO — carrega o checkpoint e age deterministicamente.

Espelha o `DQNAgentRunner` para que os dois entrem no mesmo benchmark, com as
mesmas seeds e a mesma métrica. A diferença de fundo: o DQN escolhe
``argmax Q``; aqui a política é uma distribuição, e a ação determinística é a
**moda** dos logits — que é o análogo de ε=0 para a comparação ser justa.

A máscara de ação é aplicada dentro do ator (ver `agents/ppo/network.py`):
ações proibidas recebem ``-inf`` e nunca podem ser a moda.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch

from agents.base import EpisodeRunner
from agents.ppo.network import build_actor_critic

DEFAULT_CHECKPOINT = Path("results/ppo_v1_s42/policy_best.pth")


def _infer_pooled_size(state: dict) -> Optional[int]:
    """Deduz o `pooled_size` do checkpoint pela forma da projeção do mapa.

    Mesmo mecanismo do `agents.deepq.agent`: `in_features = 64 * pooled^2`.
    Evita que quem carrega precise saber com que configuração se treinou.
    """
    for chave, peso in state.items():
        if chave.endswith("trunk.map_proj.0.weight"):
            lado = int(round((peso.shape[1] / 64) ** 0.5))
            if lado > 0 and 64 * lado * lado == peso.shape[1]:
                return lado
    return None


class PPOAgent:
    """Envelope fino sobre o ator treinado."""

    def __init__(self, actor: torch.nn.Module):
        self.actor = actor
        self.actor.eval()

    @torch.no_grad()
    def choose_action(self, obs: dict) -> int:
        batch = {k: np.asarray(v)[None] for k, v in obs.items()}
        logits, _ = self.actor(batch)
        return int(logits.argmax(dim=-1).item())

    @classmethod
    def load(cls, path, env, device: str = "cpu") -> "PPOAgent":
        state = torch.load(str(path), map_location=device, weights_only=True)
        actor, _critic = build_actor_critic(
            env, device=device, pooled_size=_infer_pooled_size(state)
        )
        # O checkpoint é o `state_dict` do algoritmo inteiro (ator + crítico +
        # otimizador); aqui só interessam os pesos do ator.
        prefixo = "policy.actor."
        pesos = {
            k[len(prefixo):]: v for k, v in state.items() if k.startswith(prefixo)
        }
        if not pesos:
            raise ValueError(
                f"nenhum peso de ator em {path} — prefixos vistos: "
                f"{sorted({k.split('.')[0] for k in state})}"
            )
        actor.load_state_dict(pesos)
        return cls(actor)


class PPOAgentRunner(EpisodeRunner):
    """Runner de benchmark: carrega ``.pth`` e age pela moda da política."""

    name = "ppo"

    def __init__(
        self,
        *,
        policy_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
    ):
        self.policy_path = Path(policy_path) if policy_path else DEFAULT_CHECKPOINT
        self.device = device
        self._agent: Optional[PPOAgent] = None
        self._bound_env_id: Optional[int] = None

    def _ensure_agent(self, env) -> PPOAgent:
        env_id = id(env)
        if self._agent is None or self._bound_env_id != env_id:
            device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            if not self.policy_path.exists():
                raise FileNotFoundError(
                    f"Policy PPO não encontrada: {self.policy_path.resolve()}. "
                    "Treine com `python -m agents.ppo.train --config ...`."
                )
            self._agent = PPOAgent.load(self.policy_path, env, device=device)
            self._bound_env_id = env_id
        return self._agent

    def choose_action(self, env) -> int:
        from agents.deepq.agent import observation_from_env

        return self._ensure_agent(env).choose_action(observation_from_env(env))
