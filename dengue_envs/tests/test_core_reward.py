import unittest

import pandas as pd

from dengue_envs.core.reward import RewardEngine


def _cases():
    real = pd.DataFrame({"disease": [0, 1]}, index=[0, 1])
    obs = pd.DataFrame({"agent_diagnosis": [0, 1]}, index=[0, 1])
    return real, obs


class RewardEngineTestCase(unittest.TestCase):
    def test_immediate_cost_is_paid_now(self):
        eng = RewardEngine(reward_delay_days=5)
        real, obs = _cases()
        # Ação "nada" (id 3) custa 0.1 imediatamente, sem desfecho.
        r = eng.compute([(0, 3)], t=0, real_cases=real, obs_cases=obs)
        self.assertAlmostEqual(r, -0.1)

    def test_confirm_correct_is_delayed(self):
        eng = RewardEngine(reward_delay_days=5, reward_correct_decision=10.0)
        real, obs = _cases()
        # Confirm (id 4) correto: custo 0 agora, +10 agendado para t+5.
        r0 = eng.compute([(0, 4)], t=0, real_cases=real, obs_cases=obs)
        self.assertAlmostEqual(r0, 0.0)
        self.assertIn(5, eng.pending_rewards)
        self.assertAlmostEqual(eng.pending_rewards[5], 10.0)
        # Nada vence entre t=1..4.
        for t in range(1, 5):
            self.assertAlmostEqual(eng.compute([], t=t, real_cases=real, obs_cases=obs), 0.0)
        # No t=5 o bônus vence.
        r5 = eng.compute([], t=5, real_cases=real, obs_cases=obs)
        self.assertAlmostEqual(r5, 10.0)

    def test_no_delay_pays_immediately(self):
        eng = RewardEngine(reward_delay_days=0, reward_correct_decision=10.0)
        real, obs = _cases()
        r = eng.compute([(0, 4)], t=0, real_cases=real, obs_cases=obs)
        self.assertAlmostEqual(r, 10.0)

    def test_incorrect_confirm_is_penalized(self):
        eng = RewardEngine(reward_delay_days=0, penalty_incorrect_decision=-20.0)
        real, obs = _cases()
        # Caso 0 é dengue (0); confirmar caso cujo agent_diagnosis diverge da verdade.
        obs.loc[0, "agent_diagnosis"] = 1  # erra
        r = eng.compute([(0, 4)], t=0, real_cases=real, obs_cases=obs)
        self.assertAlmostEqual(r, -20.0)

    def test_terminated_settles_pending_queue(self):
        eng = RewardEngine(reward_delay_days=5, reward_correct_decision=10.0)
        real, obs = _cases()
        eng.compute([(0, 4)], t=0, real_cases=real, obs_cases=obs)  # agenda +10 em t=5
        # Termina em t=2: fila pendente deve ser liquidada.
        r = eng.compute([], t=2, real_cases=real, obs_cases=obs, terminated=True)
        self.assertAlmostEqual(r, 10.0)
        self.assertEqual(eng.pending_rewards, {})

    def test_action_on_unknown_case_only_costs(self):
        eng = RewardEngine(reward_delay_days=0)
        real, obs = _cases()
        # case_id 99 não está em obs_cases -> só paga custo, sem desfecho.
        r = eng.compute([(99, 4)], t=0, real_cases=real, obs_cases=obs)
        self.assertAlmostEqual(r, 0.0)  # custo de confirm é 0

    def test_reset_clears_queue(self):
        eng = RewardEngine(reward_delay_days=5)
        real, obs = _cases()
        eng.compute([(0, 4)], t=0, real_cases=real, obs_cases=obs)
        eng.reset()
        self.assertEqual(eng.pending_rewards, {})


if __name__ == "__main__":
    unittest.main()
