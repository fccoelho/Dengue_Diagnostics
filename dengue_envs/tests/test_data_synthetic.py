import unittest

import pandas as pd

from dengue_envs.data import CASE_COLUMNS, EpidemicGenerator, SyntheticGenerator, World


class SyntheticGeneratorTestCase(unittest.TestCase):
    def setUp(self):
        self.gen = SyntheticGenerator(size=80, episize=30, epilength=6)

    def test_build_world_returns_world(self):
        world = self.gen.build_world()
        self.assertIsInstance(world, World)
        self.assertEqual(world.size, 80)

    def test_generate_returns_casedf(self):
        df = self.gen.generate()
        self.assertIsInstance(df, pd.DataFrame)
        for col in CASE_COLUMNS:
            self.assertIn(col, df.columns)

    def test_satisfies_protocol(self):
        # runtime_checkable Protocol: o adaptador deve ser reconhecido.
        self.assertIsInstance(self.gen, EpidemicGenerator)


if __name__ == "__main__":
    unittest.main()
