"""Testes do KrigingDensityGenerator (sem PyKrige / zikario — usa .npz sintético)."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from dengue_envs.data.base import CASE_COLUMNS, EpidemicGenerator
from dengue_envs.data.kriging_generator import (
    MODEL_DISEASES,
    KrigingDensityGenerator,
    KrigingSurfaces,
    KrigingWorld,
    load_kriging_surfaces,
    probability_to_env_grid,
    sample_xy_from_prob,
    save_kriging_surfaces,
)
from dengue_envs.data.view_generator import export_generator_map
from dengue_envs.wrappers.factory import make_env, make_raw_env


def _tiny_surfaces() -> KrigingSurfaces:
    """Dois blobs gaussianos em um grid 20×20 (ny, nx)."""
    ny = nx = 20
    yy, xx = np.mgrid[0:ny, 0:nx]
    dengue = np.exp(-((xx - 5) ** 2 + (yy - 5) ** 2) / 8.0)
    chik = np.exp(-((xx - 15) ** 2 + (yy - 15) ** 2) / 8.0)
    dengue = dengue / dengue.sum()
    chik = chik / chik.sum()
    return KrigingSurfaces(
        xmin=0.0,
        xmax=1000.0,
        ymin=0.0,
        ymax=1000.0,
        cell_size=50.0,
        prob_dengue=dengue,
        prob_chik=chik,
        intensity_dengue=dengue * 10,
        intensity_chik=chik * 10,
    )


def _write_tiny_npz(path: Path) -> Path:
    s = _tiny_surfaces()
    payload = {
        "xmin": s.xmin,
        "xmax": s.xmax,
        "ymin": s.ymin,
        "ymax": s.ymax,
        "cell_size": s.cell_size,
        "variogram_model": "spherical",
        "obs_cell_size": 50.0,
        "prob_dengue": s.prob_dengue,
        "prob_chikungunya": s.prob_chik,
        "intensity_dengue": s.intensity_dengue,
        "intensity_chikungunya": s.intensity_chik,
        "class_prob_dengue": s.prob_dengue,
        "class_prob_chikungunya": s.prob_chik,
    }
    return save_kriging_surfaces(payload, path)


class ProbabilityHelpersTestCase(unittest.TestCase):
    def test_probability_to_env_grid_sums_to_one(self):
        s = _tiny_surfaces()
        out = probability_to_env_grid(s.prob_dengue, 40)
        self.assertEqual(out.shape, (40, 40))
        self.assertAlmostEqual(float(out.sum()), 1.0, places=5)

    def test_sample_xy_respects_shape(self):
        rng = np.random.default_rng(0)
        prob = probability_to_env_grid(_tiny_surfaces().prob_dengue, 30)
        xs, ys = sample_xy_from_prob(prob, 50, rng)
        self.assertEqual(len(xs), 50)
        self.assertTrue(((xs >= 0) & (xs < 30)).all())
        self.assertTrue(((ys >= 0) & (ys < 30)).all())


class KrigingGeneratorTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.npz = _write_tiny_npz(Path(self.tmp.name) / "tiny.npz")
        self.gen = KrigingDensityGenerator(
            size=60,
            episize=40,
            epilength=8,
            surfaces_path=self.npz,
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_load_roundtrip(self):
        loaded = load_kriging_surfaces(self.npz)
        self.assertEqual(loaded.prob_dengue.shape, (20, 20))

    def test_satisfies_protocol(self):
        self.assertIsInstance(self.gen, EpidemicGenerator)

    def test_build_world(self):
        world = self.gen.build_world(random_state=np.random.default_rng(1))
        self.assertIsInstance(world, KrigingWorld)
        self.assertFalse(world.casedf.empty)
        for col in CASE_COLUMNS:
            self.assertIn(col, world.casedf.columns)
        self.assertGreater(world.dengue_total, 0)
        self.assertGreater(world.chik_total, 0)

    def test_generate_returns_dataframe(self):
        df = self.gen.generate(seed=7)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertTrue(set(df["disease"].unique()).issubset({0, 1}))

    def test_spatial_mass_near_foci(self):
        """Dengue deve concentrar-se no canto NW do grid env; chik no SE."""
        world = self.gen.build_world(random_state=np.random.default_rng(0))
        d = world.casedf[world.casedf.disease == 0]
        c = world.casedf[world.casedf.disease == 1]
        # Após resize 20→60, blob em (5,5)/20 ≈ (15,15)/60; (15,15)/20 ≈ (45,45)/60
        self.assertLess(d["x"].mean(), c["x"].mean())
        self.assertLess(d["y"].mean(), c["y"].mean())


class ModelDiseasesTestCase(unittest.TestCase):
    def test_model_excludes_zika(self):
        self.assertEqual(MODEL_DISEASES, ("Dengue", "Chikungunya"))
        self.assertNotIn("Zika", MODEL_DISEASES)


class ViewGeneratorTestCase(unittest.TestCase):
    def test_export_synthetic_png(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "synthetic.png"
            path = export_generator_map(
                "synthetic",
                seed=0,
                size=40,
                episize=20,
                epilength=5,
                output=out,
                show=False,
            )
            self.assertTrue(path.exists())
            self.assertGreater(path.stat().st_size, 1000)


class KrigingFactoryTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.npz = _write_tiny_npz(Path(self.tmp.name) / "tiny.npz")

    def tearDown(self):
        self.tmp.cleanup()

    def test_make_env_kriging_runs(self):
        cfg = {
            "env": {
                "generator": "kriging",
                "surfaces_path": str(self.npz),
                "size": 50,
                "episize": 25,
                "epilength": 6,
                "start_day": 1,
                "reward_delay_days": 0,
                "lab_delay_days": 0,
                "randomize_outbreak": False,
            },
            "wrappers": ["map_tensor", "case_by_case"],
        }
        env = make_env(cfg)
        obs, _ = env.reset(seed=11)
        self.assertIn("map", obs)
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        self.assertIsInstance(float(reward), float)

    def test_unknown_generator_raises(self):
        with self.assertRaises(ValueError):
            make_raw_env({"env": {"generator": "nope", "size": 40}})


if __name__ == "__main__":
    unittest.main()
