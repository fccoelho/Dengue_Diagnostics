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


class TestYearsByDisease(unittest.TestCase):
    """Cada doença usa só os casos dos seus anos (sem PyKrige: o kriging é trocado por um espião)."""

    def _build(self, **kw):
        from unittest import mock

        import dengue_envs.data.kriging_generator as kg

        casos = pd.DataFrame({
            "Doenca": ["Dengue"] * 6 + ["Chikungunya"] * 4,
            "DT_SIN_PRI": pd.to_datetime(["2015-03-01"] * 3 + ["2016-03-01"] * 3 + ["2016-03-01"] * 4),
            "x": [1.0, 2.0, 3.0, 30.0, 31.0, 32.0, 50.0, 51.0, 52.0, 53.0],
            "y": [0.0] * 10,
        })
        recebido = {}

        def espiao_kriging(x, y, **_):
            from dengue_envs.data.kriging import GridSpec

            recebido[len(recebido)] = sorted(x.tolist())
            return None, np.ones((2, 2)), np.zeros((2, 2)), GridSpec(0, 2, 0, 2, 1)

        with mock.patch.object(kg, "load_zikario_cases", return_value=casos) as carrega, \
             mock.patch.object(kg, "intensity_surface_from_points", side_effect=espiao_kriging):
            _, payload = kg.build_kriging_surfaces("ignorado.gpkg", **kw)
        return carrega.call_args.kwargs["years"], recebido, payload

    def test_padrao_usa_os_mesmos_anos_para_as_duas(self):
        anos, recebido, payload = self._build(years=(2015, 2016))
        self.assertEqual(list(anos), [2015, 2016])
        self.assertEqual(recebido[0], [1.0, 2.0, 3.0, 30.0, 31.0, 32.0])  # dengue, os dois anos
        self.assertEqual(int(payload["n_dengue"]), 6)

    def test_dengue_de_um_ano_e_chik_de_outro(self):
        anos, recebido, payload = self._build(
            years=(2016,), years_by_disease={"Dengue": (2015,)})
        # O arquivo é lido com a UNIÃO dos anos; o filtro por doença vem depois.
        self.assertEqual(list(anos), [2015, 2016])
        self.assertEqual(recebido[0], [1.0, 2.0, 3.0])            # só a dengue de 2015
        self.assertEqual(recebido[1], [50.0, 51.0, 52.0, 53.0])  # chik de 2016
        self.assertEqual(payload["years_dengue"].tolist(), [2015])
        self.assertEqual(payload["years_chikungunya"].tolist(), [2016])


class TestTransformSurfaces(unittest.TestCase):
    """Os botões de dificuldade espacial fazem o que dizem, e só isso."""

    def setUp(self):
        self.s = _tiny_surfaces()

    def _acc(self, s):
        from dengue_envs.data.kriging_generator import position_bayes_accuracy

        return position_bayes_accuracy(s, size=40)

    def test_padrao_e_identidade(self):
        from dengue_envs.data.kriging_generator import transform_surfaces

        self.assertIs(transform_surfaces(self.s), self.s)

    def test_saidas_sao_probabilidades(self):
        from dengue_envs.data.kriging_generator import transform_surfaces

        for kw in ({"temperature": 0.0}, {"temperature": 50.0}, {"clamp_quantiles": (0.1, 0.9)},
                   {"mix_uniform": 0.3}):
            t = transform_surfaces(self.s, **kw)
            for p in (t.prob_dengue, t.prob_chik):
                self.assertAlmostEqual(float(p.sum()), 1.0, places=9, msg=str(kw))
                self.assertTrue(np.all(np.isfinite(p)) and np.all(p >= 0), msg=str(kw))

    def test_temperatura_separa_as_doencas_monotonicamente(self):
        from dengue_envs.data.kriging_generator import transform_surfaces

        accs = [self._acc(transform_surfaces(self.s, temperature=t)) for t in (0.0, 0.5, 1.0, 2.0, 8.0)]
        self.assertAlmostEqual(accs[0], 0.5, places=9)  # τ = 0 é a uniforme
        self.assertEqual(accs, sorted(accs))
        self.assertGreater(accs[-1], accs[2])

    def test_mistura_total_apaga_a_geografia(self):
        from dengue_envs.data.kriging_generator import transform_surfaces

        self.assertAlmostEqual(self._acc(transform_surfaces(self.s, mix_uniform=1.0)), 0.5, places=9)
        self.assertLess(self._acc(transform_surfaces(self.s, mix_uniform=0.5)), self._acc(self.s))

    def test_clamp_limita_a_razao_entre_celulas(self):
        from dengue_envs.data.kriging_generator import transform_surfaces

        t = transform_surfaces(self.s, clamp_quantiles=(0.2, 0.8))
        antes = self.s.prob_dengue.max() / self.s.prob_dengue.min()
        depois = t.prob_dengue.max() / t.prob_dengue.min()
        self.assertLess(depois, antes)

    def test_parametros_invalidos(self):
        from dengue_envs.data.kriging_generator import transform_surfaces

        with self.assertRaises(ValueError):
            transform_surfaces(self.s, temperature=-1.0)
        with self.assertRaises(ValueError):
            transform_surfaces(self.s, mix_uniform=1.5)

    def test_yaml_chega_ao_mundo(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_tiny_npz(Path(tmp) / "s.npz")
            base = {"generator": "kriging", "surfaces_path": str(path), "size": 50,
                    "episize": 25, "epilength": 6, "start_day": 1, "reward_delay_days": 0,
                    "lab_delay_days": 0, "randomize_outbreak": False}
            probs = {}
            for nome, extra in (("ref", {}), ("quente", {"surface_temperature": 8.0})):
                env = make_raw_env({"env": {**base, **extra}})
                env.reset(seed=1)
                probs[nome] = env.world.prob_dengue.copy()
            self.assertFalse(np.allclose(probs["ref"], probs["quente"]))
            self.assertGreater(probs["quente"].max(), probs["ref"].max())
