"""Testes do agente tabular Q-Learning."""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest

from agents.qlearning.agent import (
    CHECKPOINT_FORMAT_VERSION,
    QLearningAgent,
    QLearningAgentRunner,
)
from agents.qlearning.state import STATE_VERSION_RICH, StateEncoder
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


def test_q_update_changes_value():
    agent = QLearningAgent(alpha=1.0, gamma=0.0, epsilon=0.0)
    agent.update("s0", action=3, reward=10.0, next_state="s1")
    assert agent.q_table["s0"][3] == pytest.approx(10.0)


def test_save_and_load_roundtrip(tmp_path):
    agent = QLearningAgent(alpha=1.0, gamma=0.0, encoder=StateEncoder.from_config({}))
    agent.update("dp=0|cl=1", 1, 5.0, "terminal")
    path = agent.save(tmp_path / "q.pkl")
    loaded = QLearningAgent.load(path, epsilon=0.0)
    assert loaded.encoder.version == STATE_VERSION_RICH
    assert loaded.q_table["dp=0|cl=1"][1] == pytest.approx(5.0)


def test_save_includes_metadata(tmp_path):
    agent = QLearningAgent(encoder=StateEncoder.from_config({}))
    path = agent.save(tmp_path / "q.pkl")
    with open(path, "rb") as f:
        payload = pickle.load(f)
    assert payload["format_version"] == CHECKPOINT_FORMAT_VERSION
    assert payload["state_version"] == STATE_VERSION_RICH
    assert "encoder_config" in payload


def test_load_legacy_flat_dict(tmp_path):
    legacy = {"0|1|1|1": np.zeros(6)}
    path = tmp_path / "legacy.pkl"
    with open(path, "wb") as f:
        pickle.dump(legacy, f)
    agent = QLearningAgent.load(path)
    assert "0|1|1|1" in agent.q_table


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
    agent = QLearningAgent(epsilon=0.0, encoder=StateEncoder.from_config({}))
    state = "dp=0|cl=1|adx=0|td=0|tc=0|epi=0|tdn=0|lp=0|cf=0|q=0|dd=0|dc=0|cd=1|ld=0|lc=0|lpd=0|lpc=0|la=0|ct=0|wk=0"
    agent.q_table[state] = np.array([0, 0, 0, 99, 0, 0], dtype=np.float64)
    path = tmp_path / "q_table.pkl"
    agent.save(path)

    runner = QLearningAgentRunner(q_table_path=path)
    env = make_env(_ENV_CONFIG)
    env.reset(seed=0)
    action = runner.choose_action(env)
    assert 0 <= action < 6
    env.close()
