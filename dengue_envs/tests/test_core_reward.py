import unittest

import pandas as pd

from dengue_envs.core.reward import RewardEngine


def _cases():
    real = pd.DataFrame({"disease": [0, 1]}, index=[0, 1])
    obs = pd.DataFrame({"agent_diagnosis": [0, 1]}, index=[0, 1])
    return real, obs


def _cases_with_tests(diseases, diagnoses, testd, testc):
    """Constrói real/obs com colunas de teste para os cenários terminais."""
    idx = list(range(len(diseases)))
    real = pd.DataFrame({"disease": diseases}, index=idx)
    obs = pd.DataFrame(
        {"agent_diagnosis": diagnoses, "testd": testd, "testc": testc}, index=idx
    )
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

    def test_discard_correct_when_true_other(self):
        # Descartar corretamente um caso que é "outro" (2) rende bônus.
        eng = RewardEngine(reward_delay_days=0, reward_correct_decision=10.0)
        real, obs = _cases_with_tests([2], [2], [0], [0])
        r = eng.compute([(0, 5)], t=0, real_cases=real, obs_cases=obs)
        self.assertAlmostEqual(r, 10.0)

    def test_discard_missed_real_case_is_heavily_penalized(self):
        # Descartar um caso que era doença real = falso negativo (penalidade pesada).
        eng = RewardEngine(reward_delay_days=0, penalty_missed_case=-30.0)
        # doença real = dengue (0), mas agent_diagnosis foi mutado para 2 pelo step.
        real, obs = _cases_with_tests([0], [2], [0], [0])
        r = eng.compute([(0, 5)], t=0, real_cases=real, obs_cases=obs)
        self.assertAlmostEqual(r, -30.0)

    def test_terminated_settles_pending_queue(self):
        # Isola a liquidação da fila (sem placar final).
        eng = RewardEngine(
            reward_delay_days=5,
            reward_correct_decision=10.0,
            final_correct_bonus=0.0,
            penalty_untested_misdiagnosed=0.0,
        )
        real, obs = _cases()
        eng.compute([(0, 4)], t=0, real_cases=real, obs_cases=obs)  # agenda +10 em t=5
        # Termina em t=2: fila pendente deve ser liquidada.
        r = eng.compute([], t=2, real_cases=real, obs_cases=obs, terminated=True)
        self.assertAlmostEqual(r, 10.0)
        self.assertEqual(eng.pending_rewards, {})

    def test_terminal_bonus_counts_correct_cases(self):
        # 2 casos corretos -> +1 cada no fim do episódio.
        eng = RewardEngine(
            reward_delay_days=5,
            final_correct_bonus=1.0,
            penalty_untested_misdiagnosed=-10.0,
        )
        real, obs = _cases_with_tests([0, 1], [0, 1], [0, 0], [0, 0])
        r = eng.compute([], t=3, real_cases=real, obs_cases=obs, terminated=True)
        self.assertAlmostEqual(r, 2.0)

    def test_terminal_penalty_for_untested_misdiagnosed(self):
        # Caso errado E nunca testado -> penalidade final.
        eng = RewardEngine(
            reward_delay_days=5,
            final_correct_bonus=1.0,
            penalty_untested_misdiagnosed=-10.0,
        )
        # caso 0 correto (+1); caso 1 errado e não testado (-10).
        real, obs = _cases_with_tests([0, 1], [0, 2], [0, 0], [0, 0])
        r = eng.compute([], t=3, real_cases=real, obs_cases=obs, terminated=True)
        self.assertAlmostEqual(r, 1.0 - 10.0)

    def test_terminal_no_untested_penalty_when_case_was_tested(self):
        # Caso errado mas testado -> não recebe a penalidade de "não testado".
        eng = RewardEngine(
            reward_delay_days=5,
            final_correct_bonus=1.0,
            penalty_untested_misdiagnosed=-10.0,
        )
        # caso 0 correto (+1); caso 1 errado, porém testado para dengue (testd=1) -> 0.
        real, obs = _cases_with_tests([0, 1], [0, 2], [0, 1], [0, 0])
        r = eng.compute([], t=3, real_cases=real, obs_cases=obs, terminated=True)
        self.assertAlmostEqual(r, 1.0)

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
