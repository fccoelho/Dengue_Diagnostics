"""Testes do agente DQN no framework novo (Tianshou 2.x)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import torch

from agents.deepq.agent import (
    DEFAULT_CHECKPOINT,
    DQNAgent,
    DQNAgentRunner,
    build_policy,
    observation_from_env,
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


def test_observation_from_env_after_reset():
    env = make_env(_ENV_CONFIG)
    obs, _ = env.reset(seed=0)
    rebuilt = observation_from_env(env)
    assert rebuilt["map"].shape == obs["map"].shape
    assert rebuilt["case_coords"].shape == (2,)
    env.close()


def test_build_and_act(tmp_path):
    env = make_env(_ENV_CONFIG)
    obs, _ = env.reset(seed=1)
    policy = build_policy(env, device="cpu")
    path = tmp_path / "policy.pth"
    torch.save(policy.state_dict(), path)

    agent = DQNAgent.load(path, env, device="cpu")
    action = agent.choose_action(obs)
    assert 0 <= action < 6
    action2 = agent.choose_action_from_env(env)
    assert 0 <= action2 < 6
    env.close()


def test_runner_evaluate_one_episode_without_checkpoint():
    runner = DQNAgentRunner(policy_path=Path("nao_existe_dqn.pth"), device="cpu")
    rows = runner.evaluate(
        make_env,
        seeds=[42],
        env_config=_ENV_CONFIG,
        save_artifacts=False,
    )
    assert len(rows) == 1
    assert rows[0]["agent"] == "dqn"
    assert "Recompensa Total" in rows[0]


def test_runner_evaluate_with_checkpoint(tmp_path):
    env = make_env(_ENV_CONFIG)
    env.reset(seed=0)
    policy = build_policy(env, device="cpu")
    path = tmp_path / "policy.pth"
    torch.save(policy.state_dict(), path)
    env.close()

    runner = DQNAgentRunner(policy_path=path, device="cpu")
    rows = runner.evaluate(
        make_env,
        seeds=[7],
        env_config=_ENV_CONFIG,
        save_artifacts=False,
    )
    assert rows[0]["agent"] == "dqn"
    assert DEFAULT_CHECKPOINT.name == "policy_best.pth"
