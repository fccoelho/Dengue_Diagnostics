import unittest

from gymnasium import spaces

from dengue_envs.envs.dengue_diagnostics import DengueDiagnosticsEnv


class MyEnvTestCase(unittest.TestCase):
    def setUp(self):
        self.Env = DengueDiagnosticsEnv(size=100, episize=40, epilength=8)

    def test_init(self):
        self.assertIsInstance(self.Env.action_space, spaces.Sequence)
        self.assertIsInstance(self.Env.observation_space, spaces.Dict)

    def test_observation_space(self):
        obs = self.Env.observation_space.sample()
        self.assertIsInstance(obs['tnot'], tuple)
        self.assertIsInstance(obs['testd'], tuple)
        self.assertIsInstance(obs['testc'], tuple)
        self.assertIsInstance(obs['clinical_diagnostic'], tuple)
        self.assertIsInstance(obs['epiconf'], tuple)

    def test_action_space(self):
        action = self.Env.action_space.sample()
        self.assertIsInstance(action, tuple)
        if len(action) > 0:
            self.assertIsInstance(action[0], tuple)

    def test_action_space_case_dimension(self):
        # Case id dimension must cover every generated case, not 2*episize.
        case_dim = self.Env.action_space.feature_space.spaces[0].n
        self.assertEqual(case_dim, self.Env.num_cases)

    def test_reset(self):
        r = self.Env.reset()
        self.assertEqual(2, len(r))

    def test_obs_info_after_reset(self):
        obs, info = self.Env.reset()
        self.assertIn('tnot', obs)
        self.assertIsInstance(obs['clinical_diagnostic'], tuple)
        self.assertEqual(2, len(info))

    def test_step(self):
        self.Env.reset()
        observation, reward, terminated, truncated, info = self.Env.step(self.Env.action_space.sample())
        self.assertIsInstance(reward, float)
        self.assertIn('clinical_diagnostic', observation)

    def _first_active_case_id(self, env):
        for _ in range(env.epilength + 2):
            if env.cases_t:
                return env.get_case_id(env.cases_t[0])
            env.step(tuple())
        return None

    def test_test_results_persist_across_steps(self):
        env = DengueDiagnosticsEnv(size=100, episize=40, epilength=8)
        env.reset(seed=42)

        case_id = self._first_active_case_id(env)
        self.assertIsNotNone(case_id)

        env.step(((case_id, 0),))  # dengue lab test
        status = env.obs_cases.loc[case_id, "testd"]
        self.assertNotEqual(status, 0)

        # Advancing a day with no action must NOT wipe the previous test result.
        env.step(tuple())
        self.assertEqual(env.obs_cases.loc[case_id, "testd"], status)

    def test_agent_diagnosis_persists(self):
        env = DengueDiagnosticsEnv(size=100, episize=40, epilength=8)
        env.reset(seed=7)

        case_id = self._first_active_case_id(env)
        self.assertIsNotNone(case_id)

        env.step(((case_id, 5),))  # discard -> agent_diagnosis = 2 (other)
        self.assertEqual(env.obs_cases.loc[case_id, "agent_diagnosis"], 2)

        env.step(tuple())
        self.assertEqual(env.obs_cases.loc[case_id, "agent_diagnosis"], 2)

    def test_dengue_test_conditioned_on_true_disease(self):
        env = DengueDiagnosticsEnv(size=100, episize=40, epilength=8)
        env.reset(seed=1)

        dengue_ids = env.real_cases[env.real_cases.disease == 0].index
        chik_ids = env.real_cases[env.real_cases.disease == 1].index
        self.assertGreater(len(dengue_ids), 0)
        self.assertGreater(len(chik_ids), 0)

        dengue_positive = [env._dengue_lab_test(int(dengue_ids[0])) for _ in range(300)].count(2)
        chik_positive_on_dengue_test = [env._dengue_lab_test(int(chik_ids[0])) for _ in range(300)].count(2)

        # A dengue test should be positive far more often on a true dengue case.
        self.assertGreater(dengue_positive, chik_positive_on_dengue_test)

    def test_epi_confirm_returns_binary(self):
        env = DengueDiagnosticsEnv(size=100, episize=40, epilength=8)
        env.reset(seed=3)

        case_id = self._first_active_case_id(env)
        self.assertIsNotNone(case_id)
        result = env._epi_confirm(case_id)
        self.assertIn(result, (0, 1))

    def test_full_episode_runs(self):
        env = DengueDiagnosticsEnv(size=100, episize=40, epilength=8)
        env.reset(seed=5)
        terminated = False
        steps = 0
        while not terminated and steps < 200:
            action = tuple((env.get_case_id(c), 3) for c in env.cases_t)
            _, _, terminated, _, _ = env.step(action)
            steps += 1
        self.assertTrue(terminated)


if __name__ == '__main__':
    unittest.main()
