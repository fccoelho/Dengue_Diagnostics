import unittest

from dengue_envs.metrics.episode_metrics import episode_metrics, step_accuracy


class MetricsTestCase(unittest.TestCase):
    def test_step_accuracy_perfect(self):
        true = [{"disease": 0}, {"disease": 1}]
        estimated = [(0, 0, 0), (0, 0, 1)]
        mean_acc, mape = step_accuracy(true, estimated)
        self.assertAlmostEqual(mean_acc, 1.0)
        self.assertAlmostEqual(mape, 0.0)

    def test_step_accuracy_empty(self):
        mean_acc, mape = step_accuracy([], [])
        self.assertEqual(mean_acc, 0.0)
        self.assertEqual(mape, 0.0)

    def test_episode_metrics_perfect(self):
        y_true = [0, 1, 0, 1]
        y_pred = [0, 1, 0, 1]
        m = episode_metrics(y_true, y_pred, total_tests=2, total_reward=5.0)
        self.assertAlmostEqual(m["Acurácia"], 1.0)
        self.assertAlmostEqual(m["Recompensa Total"], 5.0)
        self.assertEqual(m["Testes Realizados"], 2)

    def test_episode_metrics_keys(self):
        m = episode_metrics([0, 1], [0, 1], total_tests=0, total_reward=0.0)
        for key in [
            "Acurácia",
            "Sensibilidade (Dengue)",
            "Especificidade",
            "F1-Score",
            "Precisão",
            "Redução de Testes (%)",
        ]:
            self.assertIn(key, m)


if __name__ == "__main__":
    unittest.main()
