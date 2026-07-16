import unittest
from pathlib import Path
import tempfile

import matplotlib

matplotlib.use("Agg")

from dengue_envs.envs.dengue_diagnostics import DengueDiagnosticsEnv
from agents.artifacts import (
    confusion_map_path,
    epidemic_map_path,
    save_confusion_map,
    save_epidemic_map,
)


class ArtifactsTestCase(unittest.TestCase):
    def test_confusion_map_saved(self):
        env = DengueDiagnosticsEnv(size=100, episize=40, epilength=8)
        env.reset(seed=7)
        env.step(tuple())
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            path = save_confusion_map(env, "random", 7, output_dir=out)
            self.assertEqual(path, confusion_map_path("random", 7, out))
            self.assertTrue(path.exists())
            self.assertGreater(path.stat().st_size, 0)

    def test_epidemic_map_saved(self):
        env = DengueDiagnosticsEnv(size=100, episize=40, epilength=8)
        env.reset(seed=7)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            path = save_epidemic_map(env, 7, output_dir=out, skip_if_exists=False)
            self.assertEqual(path, epidemic_map_path(7, out))
            self.assertTrue(path.exists())
            self.assertGreater(path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
