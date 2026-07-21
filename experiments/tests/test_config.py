import os
import unittest

from experiments.config import (
    CONFIGS_DIR,
    build_env,
    build_env_factory,
    build_raw_env,
    deep_merge,
    get_train_config,
    load_config,
    load_named_config,
    normalize_env_config,
)

# Override pequeno para acelerar a construção de ambientes nos testes.
SMALL_ENV = {"env": {"size": 60, "episize": 20, "epilength": 6, "start_day": 1,
                     "reward_delay_days": 0},
             "wrappers": ["map_tensor", "case_by_case"]}


class DeepMergeTestCase(unittest.TestCase):
    def test_recursive_merge(self):
        base = {"env": {"size": 400, "episize": 150}, "wrappers": ["a"]}
        override = {"env": {"size": 600}, "train": {"seed": 1}}
        merged = deep_merge(base, override)
        self.assertEqual(merged["env"]["size"], 600)
        self.assertEqual(merged["env"]["episize"], 150)  # preservado
        self.assertEqual(merged["train"]["seed"], 1)

    def test_lists_are_replaced_not_merged(self):
        merged = deep_merge({"w": ["a", "b"]}, {"w": ["c"]})
        self.assertEqual(merged["w"], ["c"])

    def test_does_not_mutate_inputs(self):
        base = {"env": {"size": 400}}
        deep_merge(base, {"env": {"size": 1}})
        self.assertEqual(base["env"]["size"], 400)


class NormalizeTestCase(unittest.TestCase):
    def test_specificity_range_becomes_tuple(self):
        cfg = normalize_env_config({"clinical_specificity": [0.5, 0.95]})
        self.assertEqual(cfg["clinical_specificity"], (0.5, 0.95))

    def test_specificity_single_becomes_float(self):
        cfg = normalize_env_config({"clinical_specificity": [0.8]})
        self.assertEqual(cfg["clinical_specificity"], 0.8)

    def test_centers_become_tuples(self):
        cfg = normalize_env_config({"dengue_center": [1, 2], "chik_center": [3, 4]})
        self.assertEqual(cfg["dengue_center"], (1, 2))
        self.assertEqual(cfg["chik_center"], (3, 4))


class LoadConfigTestCase(unittest.TestCase):
    def test_packaged_env_default_loads(self):
        cfg = load_named_config("env/synthetic_default.yaml")
        self.assertIn("env", cfg)
        self.assertEqual(cfg["env"]["size"], 400)
        self.assertEqual(cfg["wrappers"], ["map_tensor", "case_by_case"])

    def test_include_is_resolved_and_overridden(self):
        # synthetic_large inclui synthetic_default e sobrescreve size/episize.
        cfg = load_named_config("env/synthetic_large.yaml")
        self.assertEqual(cfg["env"]["size"], 600)
        self.assertEqual(cfg["env"]["episize"], 300)
        # Herdado do include (não redefinido no arquivo grande).
        self.assertEqual(cfg["env"]["reward_delay_days"], 5)
        self.assertNotIn("include", cfg)

    def test_train_config_includes_env(self):
        cfg = load_named_config("train/random.yaml")
        self.assertEqual(cfg["train"]["algorithm"], "random")
        self.assertIn("env", cfg)  # veio do include
        self.assertEqual(cfg["env"]["size"], 400)

    def test_get_train_config_defaults(self):
        train = get_train_config({})
        self.assertEqual(train["algorithm"], "random")
        self.assertIsNone(train["seed"])

    def test_all_packaged_configs_parse(self):
        for root, _dirs, files in os.walk(CONFIGS_DIR):
            for f in files:
                if f.endswith((".yaml", ".yml")):
                    load_config(os.path.join(root, f))  # não deve levantar


class BuildEnvTestCase(unittest.TestCase):
    def test_build_env_returns_wrapped(self):
        from dengue_envs.wrappers.case_by_case import CaseByCaseWrapper

        env = build_env(SMALL_ENV)
        self.assertIsInstance(env, CaseByCaseWrapper)
        self.assertEqual(env.unwrapped.size, 60)

    def test_build_raw_env(self):
        from dengue_envs.envs.dengue_diagnostics import DengueDiagnosticsEnv

        env = build_raw_env(SMALL_ENV)
        self.assertIsInstance(env, DengueDiagnosticsEnv)

    def test_specificity_list_from_yaml_is_usable(self):
        cfg = {"env": {**SMALL_ENV["env"], "clinical_specificity": [0.5, 0.95]},
               "wrappers": SMALL_ENV["wrappers"]}
        env = build_env(cfg)
        # Se a tupla foi aplicada, o env sorteia float dentro do intervalo.
        env.reset(seed=0)
        self.assertIsInstance(env.unwrapped.clinical_specificity, float)
        self.assertTrue(0.5 <= env.unwrapped.clinical_specificity <= 0.95)

    def test_factory_returns_fresh_envs(self):
        factory = build_env_factory(SMALL_ENV)
        e1 = factory()
        e2 = factory()
        self.assertIsNot(e1, e2)


if __name__ == "__main__":
    unittest.main()
