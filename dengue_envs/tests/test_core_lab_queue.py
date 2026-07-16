import unittest

from dengue_envs.core.lab_queue import LabResultQueue


class LabResultQueueTestCase(unittest.TestCase):
    def test_schedule_matures_after_delay(self):
        q = LabResultQueue(lab_delay_days=3)
        q.schedule(0, case_id=7, result=2, t=1)  # amadurece em t=4
        # Nada disponível antes do prazo.
        for t in range(1, 4):
            self.assertEqual(q.pop_matured(t), [])
        # No dia 4 o resultado chega.
        self.assertEqual(q.pop_matured(4), [(0, 7, 2)])
        # E some da fila depois de consumido.
        self.assertEqual(q.pop_matured(4), [])

    def test_zero_delay_matures_same_day(self):
        q = LabResultQueue(lab_delay_days=0)
        q.schedule(1, case_id=3, result=1, t=5)
        self.assertEqual(q.pop_matured(5), [(1, 3, 1)])

    def test_multiple_results_same_day(self):
        q = LabResultQueue(lab_delay_days=2)
        q.schedule(0, 1, 2, t=0)
        q.schedule(1, 2, 1, t=0)
        matured = q.pop_matured(2)
        self.assertIn((0, 1, 2), matured)
        self.assertIn((1, 2, 1), matured)
        self.assertEqual(len(matured), 2)

    def test_flush_returns_all_pending(self):
        q = LabResultQueue(lab_delay_days=5)
        q.schedule(0, 1, 2, t=0)
        q.schedule(1, 2, 3, t=1)
        flushed = q.flush()
        self.assertEqual(len(flushed), 2)
        self.assertEqual(q.pending, {})

    def test_reset_clears_queue(self):
        q = LabResultQueue(lab_delay_days=5)
        q.schedule(0, 1, 2, t=0)
        q.reset()
        self.assertEqual(q.pending, {})


if __name__ == "__main__":
    unittest.main()
