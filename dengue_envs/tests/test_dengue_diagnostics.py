import unittest
from dengue_envs.envs.dengue_diagnostics import DengueDiagnosticsEnv
from gymnasium import spaces

class MyEnvTestCase(unittest.TestCase):
    def setUp(self):
        self.Env = DengueDiagnosticsEnv()
    def test_init(self):
        self.assertIsInstance(self.Env.action_space, spaces.Sequence)
        self.assertIsInstance(self.Env.observation_space, spaces.Dict)
    def test_observation_space(self):
        obs = self.Env.observation_space.sample()
        self. assertIsInstance(obs['tcase'], tuple)
        self. assertIsInstance(obs['testd'], tuple)
        self. assertIsInstance(obs['testc'], tuple)
        self. assertIsInstance(obs['clinical_diagnostic'], tuple)
        self. assertIsInstance(obs['epiconf'], tuple)

    def test_action_space(self):
        action = self.Env.action_space.sample()
        self.assertIsInstance(action, tuple)
        self.assertIsInstance(action[0], tuple)

    def test_reset(self):
        r = self.Env.reset()
        self.assertEqual(2, len(r))

    def test_obs_info_after_reset(self):
        obs, info = self.Env.reset()
        self.assertIn('t', obs)
        self.assertIsInstance(obs['clinical_diagnostic'], tuple)
        cases_at_0 = len([case for case in self.Env.world.case_series[0] if case['t'] == 0])
        self.assertEqual(cases_at_0, len(obs['clinical_diagnostic']))
        self.assertEqual(2,len(info))

    def test_step(self):
        obs, info = self.Env.reset()
        observation, reward, terminated, _, info = self.Env.step([])
        print(observation)



if __name__ == '__main__':
    unittest.main()
