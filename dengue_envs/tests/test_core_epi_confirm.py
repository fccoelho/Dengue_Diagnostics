import unittest

import numpy as np

from dengue_envs.core.epi_confirm import epi_confirm


class EpiConfirmTestCase(unittest.TestCase):
    def setUp(self):
        self.dmap = np.zeros((30, 30))
        self.cmap = np.zeros((30, 30))
        self.dmap[10, 10] = 5  # densidade alta de dengue
        self.cmap[20, 20] = 5  # densidade alta de chik

    def test_dengue_confirmed_in_dense_area(self):
        self.assertEqual(epi_confirm(0, 10, 10, self.dmap, self.cmap), 1)

    def test_dengue_not_confirmed_in_empty_area(self):
        self.assertEqual(epi_confirm(0, 5, 5, self.dmap, self.cmap), 0)

    def test_chik_confirmed_in_dense_area(self):
        self.assertEqual(epi_confirm(1, 20, 20, self.dmap, self.cmap), 1)

    def test_other_never_confirmed(self):
        self.assertEqual(epi_confirm(2, 10, 10, self.dmap, self.cmap), 0)

    def test_returns_binary(self):
        result = epi_confirm(0, 10, 10, self.dmap, self.cmap)
        self.assertIn(result, (0, 1))


if __name__ == "__main__":
    unittest.main()
