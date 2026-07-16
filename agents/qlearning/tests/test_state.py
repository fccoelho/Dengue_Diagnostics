"""Testes do codificador de estado Q-Learning."""
from __future__ import annotations

import numpy as np
import pandas as pd

from agents.qlearning.state import (
    STATE_VERSION_COMPACT,
    STATE_VERSION_RICH,
    StateEncoder,
    encode_state_compact,
)


class _RichFakeEnv:
    def __init__(self):
        self.current_case = (42, 30, 40)
        world_size = 100
        self._current_map_obs = np.zeros((4, world_size, world_size), dtype=np.uint8)
        x, y = 30, 40
        self._current_map_obs[0, x, y] = 1  # clinical dengue
        self._current_map_obs[1, x, y] = 3  # testd positive (+1 -> 3)
        self._current_map_obs[2, x, y] = 1  # testc not tested
        self._current_map_obs[3, x, y] = 1

        class U:
            t = 8
            size = world_size
            epilength = 60
            episize = 150
            dengue_center = (25, 35)
            chik_center = (70, 70)
            obs_cases = pd.DataFrame(
                {
                    "x": [x],
                    "y": [y],
                    "t": [8],
                    "disease": [0],
                    "agent_diagnosis": [0],
                    "testd": [2],
                    "testc": [0],
                    "epiconf": [0],
                },
                index=[42],
            )

        self.unwrapped = U()


def test_encode_state_compact():
    env = _RichFakeEnv()
    key = encode_state_compact(env, day_bucket_size=5)
    assert key == "1|1|3|1"


def test_encode_state_rich_has_many_fields():
    env = _RichFakeEnv()
    enc = StateEncoder.from_config({"version": STATE_VERSION_RICH})
    key = enc.encode(env)
    assert key.startswith("dp=")
    assert "|cl=" in key
    assert "|adx=" in key
    assert "|ld=" in key
    assert "|q=" in key
    assert key != encode_state_compact(env)


def test_terminal_state():
    env = _RichFakeEnv()
    env.current_case = (0, 0, 0)
    enc = StateEncoder.from_config({})
    assert enc.encode(env) == "terminal"


def test_compact_version_via_encoder():
    env = _RichFakeEnv()
    enc = StateEncoder.from_config({"version": STATE_VERSION_COMPACT})
    assert enc.version == STATE_VERSION_COMPACT
    assert enc.encode(env) == encode_state_compact(env)
