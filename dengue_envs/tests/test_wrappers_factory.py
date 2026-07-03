import unittest

import numpy as np
from gymnasium import spaces

from dengue_envs.envs.dengue_diagnostics import DengueDiagnosticsEnv
from dengue_envs.wrappers.case_by_case import CaseByCaseWrapper
from dengue_envs.wrappers.factory import make_env, make_raw_env
from dengue_envs.wrappers.map_tensor import DengueWrapper

SMALL_CFG = {
    "env": {"size": 60, "episize": 20, "epilength": 6, "start_day": 1, "reward_delay_days": 0},
    "wrappers": ["map_tensor", "case_by_case"],
}


class FactoryTestCase(unittest.TestCase):
    def test_make_raw_env(self):
        env = make_raw_env(SMALL_CFG)
        self.assertIsInstance(env, DengueDiagnosticsEnv)
        self.assertEqual(env.size, 60)
        self.assertEqual(env.reward_delay, 0)

    def test_make_env_applies_wrappers(self):
        env = make_env(SMALL_CFG)
        self.assertIsInstance(env, CaseByCaseWrapper)
        self.assertIsInstance(env.env, DengueWrapper)

    def test_make_env_ignores_unknown_keys(self):
        cfg = {"env": {"size": 60, "episize": 20, "epilength": 6, "foo": 123}}
        env = make_raw_env(cfg)
        self.assertEqual(env.size, 60)

    def test_unknown_wrapper_raises(self):
        cfg = {"env": SMALL_CFG["env"], "wrappers": ["does_not_exist"]}
        with self.assertRaises(ValueError):
            make_env(cfg)

    def test_full_rl_env_runs_episode(self):
        env = make_env(SMALL_CFG)
        obs, info = env.reset(seed=42)
        self.assertIn("map", obs)
        self.assertIn("case_coords", obs)
        self.assertTrue(env.observation_space.contains(obs))

        terminated = truncated = False
        steps = 0
        while not (terminated or truncated) and steps < 500:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1
        self.assertTrue(terminated or truncated)


if __name__ == "__main__":
    unittest.main()
