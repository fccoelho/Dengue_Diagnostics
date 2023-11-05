import unittest
from dengue_envs.envs.dengue_diagnostics import DengueDiagnosticsEnv
from gymnasium import spaces

class MyTestCase(unittest.TestCase):
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
        self.assertIsInstance(action[0], int)
        self.assertIsInstance(action[1], int)


if __name__ == '__main__':
    unittest.main()
