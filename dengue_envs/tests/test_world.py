import unittest
from dengue_envs.data.generator import World
import numpy as np
import matplotlib.pyplot as plt


class WorldTestCase(unittest.TestCase):
    def setUp(self):
        self.world = World(popsize=1000, epilength=100)

    def test_instantiate(self):
        self.assertIsInstance(self.world, World)

    def test_case_series(self):
        self.assertEqual(len(self.world.case_series), self.world.epilength)

    def test_number_of_cases(self):
        """
        Some deviation in total number of cases from the model are expected because of rounding errors
        """
        total_cases = self.world.dengue_total + self.world.chik_total
        self.assertEqual(sum([len(d) for d in self.world.case_series]), total_cases,
            f"Number of cases in case_series deviates from total number of cases by more than 10. Total cases: {total_cases}, number of cases in case_series: {sum([len(d) for d in self.world.case_series])}",
        )

    def test_epicurve(self):
        curve = self.world._get_epi_curve()
        self.assertIsInstance(curve, np.ndarray)

    def test_case_dataframe(self):
        df = self.world.casedf
        self.assertListEqual(list(df.index), list(range(len(df))))
        self.assertListEqual(list(df.columns), ["t", "x", "y", "disease"])

    def test_viewer(self):
        self.world.view()
        plt.show()



if __name__ == "__main__":
    unittest.main()
