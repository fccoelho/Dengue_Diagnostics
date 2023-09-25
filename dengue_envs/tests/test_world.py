import unittest
from dengue_envs.envs.dengue_diagnostics import World


class WorldTestCase(unittest.TestCase):
    def test_instantiate(self):
        world = World()
        self.assertIsInstance(world, World)



if __name__ == "__main__":
    unittest.main()
