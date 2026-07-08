"""Agente tabular Q-Learning sobre o ambiente novo (wrappers map_tensor + case_by_case).

O agente decide **uma ação por caso** (`Discrete(6)`), igual ao random e ao DQN.
O estado é uma chave discreta derivada do mapa tensor e do dia corrente — compacta
o suficiente para uma Q-table, mas mais rica que o legado (`str(obs)+case_id`).

Legado: `qlearning_agent_old.py` (env bruto, ação por dia inteiro).
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np

from agents.base import EpisodeRunner

NUM_ACTIONS = 6
DEFAULT_CHECKPOINT = Path("results/qlearning/q_table.pkl")


def encode_state_from_env(env, *, day_bucket_size: int = 5) -> str:
    """Discretiza o estado a partir do wrapper `CaseByCaseWrapper`.

    Canais do mapa (ver `DengueWrapper`):
      0 = diagnóstico clínico (+1)
      1 = status teste dengue (+1)
      2 = status teste chik (+1)
    """
    case_id, x, y = env.current_case
    if case_id == 0:
        return "terminal"

    map_tensor = env._current_map_obs
    if map_tensor is None:
        return "unknown"

    clinical = int(map_tensor[0, x, y])
    testd = int(map_tensor[1, x, y])
    testc = int(map_tensor[2, x, y])
    day = int(env.unwrapped.t)
    day_bucket = min(day // max(day_bucket_size, 1), 99)

    return f"{day_bucket}|{clinical}|{testd}|{testc}"


class QLearningAgent:
    """Q-Learning tabular com ε-greedy."""

    def __init__(
        self,
        *,
        alpha: float = 0.5,
        gamma: float = 0.5,
        epsilon: float = 0.15,
        day_bucket_size: int = 5,
        q_table: Optional[Dict[str, np.ndarray]] = None,
    ):
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
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
        state = encode_state_from_env(env, day_bucket_size=self.day_bucket_size)
        return self.choose_action(state, explore=explore)

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
        with open(path, "wb") as f:
            pickle.dump(self.q_table, f)

        if also_txt:
            txt_path = path.with_suffix(".txt")
            with open(txt_path, "w", encoding="utf-8") as f:
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
        day_bucket_size: int = 5,
    ) -> "QLearningAgent":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Q-table nao encontrada: {path.resolve()}. "
                "Treine primeiro com `poetry run python agents/qlearning/train.py`."
            )
        with open(path, "rb") as f:
            q_table = pickle.load(f)
        return cls(
            alpha=alpha,
            gamma=gamma,
            epsilon=epsilon,
            day_bucket_size=day_bucket_size,
            q_table=q_table,
        )


class QLearningAgentRunner(EpisodeRunner):
    """Runner de avaliação/benchmark: carrega Q-table e age com ε=0 (greedy)."""

    name = "qlearning"

    def __init__(
        self,
        *,
        q_table_path: Optional[Union[str, Path]] = None,
        day_bucket_size: int = 5,
        epsilon: float = 0.0,
    ):
        self.q_table_path = Path(q_table_path) if q_table_path else DEFAULT_CHECKPOINT
        self.day_bucket_size = day_bucket_size
        self._agent: Optional[QLearningAgent] = None
        self._epsilon = epsilon

    def _get_agent(self) -> QLearningAgent:
        if self._agent is None:
            if self.q_table_path.exists():
                self._agent = QLearningAgent.load(
                    self.q_table_path,
                    epsilon=self._epsilon,
                    day_bucket_size=self.day_bucket_size,
                )
            else:
                print(
                    f"[qlearning] aviso: Q-table ausente em {self.q_table_path.resolve()}. "
                    "Usando tabela vazia (comportamento aleatorio)."
                )
                self._agent = QLearningAgent(
                    epsilon=self._epsilon,
                    day_bucket_size=self.day_bucket_size,
                )
        return self._agent

    def choose_action(self, env) -> int:
        return self._get_agent().choose_action_from_env(env, explore=False)
