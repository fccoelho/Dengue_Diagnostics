"""Agente tabular Q-Learning sobre o ambiente novo (wrappers map_tensor + case_by_case).

Decide **uma ação por caso** (`Discrete(7)`: 4 investigativas/neutras + 3 ações
conclusivas, uma por classe — dengue/chik/outro). Estado discretizado via
``StateEncoder`` (``rich_v1`` por padrão; ``compact_v1`` opcional como baseline
mínimo).
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np

from agents.base import EpisodeRunner
from agents.qlearning.state import (
    STATE_VERSION_COMPACT,
    STATE_VERSION_RICH,
    StateEncoder,
    encode_state_from_env,
)

NUM_ACTIONS = 7
DEFAULT_CHECKPOINT = Path("results/qlearning/q_table.pkl")
CHECKPOINT_FORMAT_VERSION = 2


class QLearningAgent:
    """Q-Learning tabular com ε-greedy."""

    def __init__(
        self,
        *,
        alpha: float = 0.5,
        gamma: float = 0.5,
        epsilon: float = 0.15,
        encoder: Optional[StateEncoder] = None,
        day_bucket_size: int = 5,
        q_table: Optional[Dict[str, np.ndarray]] = None,
    ):
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.encoder = encoder or StateEncoder.from_config({})
        self.day_bucket_size = int(day_bucket_size)
        self.q_table: Dict[str, np.ndarray] = q_table or {}

    def _ensure_state(self, state: str) -> np.ndarray:
        if state not in self.q_table:
            self.q_table[state] = np.zeros(NUM_ACTIONS, dtype=np.float64)
        return self.q_table[state]

    def choose_action(self, state: str, *, explore: bool = True) -> int:
        self._ensure_state(state)
        if explore and np.random.uniform() < self.epsilon:
            return int(np.random.randint(NUM_ACTIONS))
        return int(np.argmax(self.q_table[state]))

    def choose_action_from_env(self, env, *, explore: bool = True) -> int:
        state = encode_state_from_env(
            env, encoder=self.encoder, day_bucket_size=self.day_bucket_size
        )
        return self.choose_action(state, explore=explore)

    def encode_env(self, env) -> str:
        return encode_state_from_env(
            env, encoder=self.encoder, day_bucket_size=self.day_bucket_size
        )

    def update(self, state: str, action: int, reward: float, next_state: str) -> None:
        self._ensure_state(state)
        self._ensure_state(next_state)
        best_next = float(np.max(self.q_table[next_state]))
        td_target = float(reward) + self.gamma * best_next
        self.q_table[state][action] += self.alpha * (
            td_target - self.q_table[state][action]
        )

    def save(self, path: Union[str, Path], *, also_txt: bool = False) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "state_version": self.encoder.version,
            "encoder_config": self.encoder.config_dict(),
            "alpha": self.alpha,
            "gamma": self.gamma,
            "q_table": self.q_table,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)

        if also_txt:
            txt_path = path.with_suffix(".txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(f"# state_version={self.encoder.version}\n")
                for key, values in sorted(self.q_table.items()):
                    f.write(f"{key}: {values.tolist()}\n")

        return path

    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        *,
        alpha: float = 0.5,
        gamma: float = 0.5,
        epsilon: float = 0.0,
        encoder: Optional[StateEncoder] = None,
        day_bucket_size: int = 5,
    ) -> "QLearningAgent":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Q-table nao encontrada: {path.resolve()}. "
                "Treine primeiro com `poetry run python agents/qlearning/train.py`."
            )
        with open(path, "rb") as f:
            raw = pickle.load(f)

        q_table, loaded_encoder = cls._parse_checkpoint(raw, encoder)

        return cls(
            alpha=alpha,
            gamma=gamma,
            epsilon=epsilon,
            encoder=loaded_encoder,
            day_bucket_size=day_bucket_size,
            q_table=q_table,
        )

    @staticmethod
    def _parse_checkpoint(
        raw: Any, encoder_override: Optional[StateEncoder]
    ) -> tuple[Dict[str, np.ndarray], StateEncoder]:
        if isinstance(raw, dict) and "q_table" in raw:
            q_table = raw["q_table"]
            if encoder_override is not None:
                enc = encoder_override
            else:
                enc = StateEncoder.from_config(raw.get("encoder_config", {}))
                if "state_version" in raw and raw["state_version"] != enc.version:
                    enc.version = raw["state_version"]
            return q_table, enc

        if isinstance(raw, dict):
            # Checkpoint antigo (só dict state→array): assume estado compacto.
            enc = encoder_override or StateEncoder.from_config({"version": STATE_VERSION_COMPACT})
            return raw, enc

        raise ValueError(f"Formato de checkpoint invalido: {type(raw)}")


class QLearningAgentRunner(EpisodeRunner):
    """Runner de avaliação/benchmark: carrega Q-table e age com ε=0 (greedy)."""

    name = "qlearning"

    def __init__(
        self,
        *,
        q_table_path: Optional[Union[str, Path]] = None,
        encoder: Optional[StateEncoder] = None,
        state_config: Optional[Dict[str, Any]] = None,
        day_bucket_size: int = 5,
        epsilon: float = 0.0,
    ):
        self.q_table_path = Path(q_table_path) if q_table_path else DEFAULT_CHECKPOINT
        self.encoder = encoder or StateEncoder.from_config(state_config)
        self.day_bucket_size = day_bucket_size
        self._epsilon = epsilon
        self._agent: Optional[QLearningAgent] = None

    def _get_agent(self) -> QLearningAgent:
        if self._agent is None:
            if self.q_table_path.exists():
                self._agent = QLearningAgent.load(
                    self.q_table_path,
                    epsilon=self._epsilon,
                    encoder=self.encoder,
                    day_bucket_size=self.day_bucket_size,
                )
            else:
                print(
                    f"[qlearning] aviso: Q-table ausente em {self.q_table_path.resolve()}. "
                    "Usando tabela vazia (comportamento aleatorio)."
                )
                self._agent = QLearningAgent(
                    epsilon=self._epsilon,
                    encoder=self.encoder,
                    day_bucket_size=self.day_bucket_size,
                )
        return self._agent

    def choose_action(self, env) -> int:
        return self._get_agent().choose_action_from_env(env, explore=False)
