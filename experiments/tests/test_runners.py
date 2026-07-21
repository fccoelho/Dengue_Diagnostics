import unittest

from experiments.runners import (
    DQNRunner,
    RandomRunner,
    available_algorithms,
    get_runner,
    run_from_config,
)

SMALL_CFG = {
    "env": {"size": 60, "episize": 20, "epilength": 6, "start_day": 1,
            "reward_delay_days": 0},
    "wrappers": ["map_tensor", "case_by_case"],
    "train": {"algorithm": "random", "seed": 42, "episodes": 2},
}


class RunnerRegistryTestCase(unittest.TestCase):
    def test_available_algorithms(self):
        algos = available_algorithms()
        self.assertIn("random", algos)
        self.assertIn("dqn", algos)
        self.assertIn("ppo", algos)

    def test_get_random_runner(self):
        self.assertIsInstance(get_runner("random"), RandomRunner)

    def test_unknown_algorithm_raises(self):
        with self.assertRaises(ValueError):
            get_runner("does_not_exist")

    def test_dqn_runner_registered(self):
        runner = get_runner("dqn")
        self.assertIsInstance(runner, DQNRunner)
        self.assertEqual(runner.name, "dqn")

    def test_ppo_runner_is_stub(self):
        runner = get_runner("ppo")
        with self.assertRaises(NotImplementedError):
            runner.run(lambda: None, SMALL_CFG)


class RandomRunnerTestCase(unittest.TestCase):
    def test_run_returns_metrics(self):
        result = run_from_config(SMALL_CFG)
        self.assertEqual(result["algorithm"], "random")
        self.assertEqual(result["episodes"], 2)
        self.assertEqual(len(result["rewards"]), 2)
        self.assertIn("mean_reward", result)
        self.assertIn("std_reward", result)

    def test_rewards_are_floats(self):
        result = run_from_config(SMALL_CFG)
        for r in result["rewards"]:
            self.assertIsInstance(r, float)
        self.assertIsInstance(result["mean_reward"], float)


if __name__ == "__main__":
    unittest.main()
