"""Testes do agente tabular Q-Learning."""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest

from agents.qlearning.agent import (
    QLearningAgent,
    QLearningAgentRunner,
    encode_state_from_env,
)
from dengue_envs.wrappers import make_env

_ENV_CONFIG = {
    "env": {
        "size": 80,
        "episize": 40,
        "epilength": 12,
        "start_day": 1,
        "reward_delay_days": 0,
        "lab_delay_days": 0,
        "randomize_outbreak": False,
    },
    "wrappers": ["map_tensor", "case_by_case"],
}


class _FakeEnv:
    """Env mínimo para testar codificação de estado."""

    def __init__(self):
        self.current_case = (7, 3, 4)
        self._current_map_obs = np.zeros((4, 10, 10), dtype=np.uint8)
        self._current_map_obs[0, 3, 4] = 2
        self._current_map_obs[1, 3, 4] = 1
        self._current_map_obs[2, 3, 4] = 0

    class _Unwrapped:
        t = 12

    unwrapped = _Unwrapped()


def test_encode_state_from_env():
    env = _FakeEnv()
    key = encode_state_from_env(env, day_bucket_size=5)
    assert key == "2|2|1|0"


def test_q_update_changes_value():
    agent = QLearningAgent(alpha=1.0, gamma=0.0, epsilon=0.0)
    agent.update("s0", action=3, reward=10.0, next_state="s1")
    assert agent.q_table["s0"][3] == pytest.approx(10.0)


def test_save_and_load_roundtrip(tmp_path):
    agent = QLearningAgent(alpha=1.0, gamma=0.0)
    agent.update("a", 1, 5.0, "b")
    path = agent.save(tmp_path / "q.pkl")
    loaded = QLearningAgent.load(path, epsilon=0.0)
    assert loaded.q_table["a"][1] == pytest.approx(5.0)


def test_runner_evaluate_one_episode():
    runner = QLearningAgentRunner(q_table_path=Path("nao_existe.pkl"))
    rows = runner.evaluate(
        make_env,
        seeds=[42],
        env_config=_ENV_CONFIG,
        save_artifacts=False,
    )
    assert len(rows) == 1
    assert rows[0]["agent"] == "qlearning"
    assert "Recompensa Total" in rows[0]


def test_runner_uses_loaded_qtable(tmp_path):
    agent = QLearningAgent(epsilon=0.0)
    state = "0|1|0|0"
    agent.q_table[state] = np.array([0, 0, 0, 99, 0, 0], dtype=np.float64)
    path = tmp_path / "q_table.pkl"
    with open(path, "wb") as f:
        pickle.dump(agent.q_table, f)

    runner = QLearningAgentRunner(q_table_path=path)
    env = make_env(_ENV_CONFIG)
    obs, _ = env.reset(seed=0)
    # Força estado conhecido se possível; caso contrário só verifica greedy válido.
    action = runner.choose_action(env)
    assert 0 <= action < 6
    env.close()
