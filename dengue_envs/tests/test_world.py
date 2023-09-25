import unittest
from dengue_envs.envs.dengue_diagnostics import World
import numpy as np


class WorldTestCase(unittest.TestCase):
    def setUp(self):
        self.world = World()

    def test_instantiate(self):
        self.assertIsInstance(self.world, World)

    def test_case_series(self):
        self.assertEqual(len(self.world.case_series), self.world.epilength)

    def test_number_of_cases(self):
        """
        Some deviation in total number of cases from the model are expected because of rounding errors
        """
        total_cases = self.world.dengue_total + self.world.chik_total
        self.assertLess(
            abs(sum([len(d) for d in self.world.case_series]) - total_cases),
            10,
            f"Number of cases in case_series deviates from total number of cases by more than 10. Total cases: {total_cases}, number of cases in case_series: {sum([len(d) for d in self.world.case_series])}",
        )

    def test_epicurve(self):
        curve = self.world._get_epi_curve()
        self.assertIsInstance(curve, np.ndarray)



if __name__ == "__main__":
    unittest.main()
