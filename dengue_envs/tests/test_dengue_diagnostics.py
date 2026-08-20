import unittest

from gymnasium import spaces

from dengue_envs.envs.dengue_diagnostics import DengueDiagnosticsEnv


class MyEnvTestCase(unittest.TestCase):
    def setUp(self):
        self.Env = DengueDiagnosticsEnv(size=100, episize=40, epilength=8)

    def test_init(self):
        self.assertIsInstance(self.Env.action_space, spaces.Sequence)
        self.assertIsInstance(self.Env.observation_space, spaces.Dict)

    def test_observation_space(self):
        obs = self.Env.observation_space.sample()
        self.assertIsInstance(obs['tnot'], tuple)
        self.assertIsInstance(obs['testd'], tuple)
        self.assertIsInstance(obs['testc'], tuple)
        self.assertIsInstance(obs['clinical_diagnostic'], tuple)
        self.assertIsInstance(obs['epiconf'], tuple)

    def test_action_space(self):
        action = self.Env.action_space.sample()
        self.assertIsInstance(action, tuple)
        if len(action) > 0:
            self.assertIsInstance(action[0], tuple)

    def test_action_space_case_dimension(self):
        # Case id dimension must cover every generated case, not 2*episize.
        case_dim = self.Env.action_space.feature_space.spaces[0].n
        self.assertEqual(case_dim, self.Env.num_cases)

    def test_reset(self):
        r = self.Env.reset()
        self.assertEqual(2, len(r))

    def test_obs_info_after_reset(self):
        obs, info = self.Env.reset()
        self.assertIn('tnot', obs)
        self.assertIsInstance(obs['clinical_diagnostic'], tuple)
        # info NÃO deve carregar arrays grandes (grids 400×400): libs de RL
        # guardam o info de cada transição no replay buffer, o que estouraria a
        # RAM. Os mapas seguem em env.unwrapped.dmap/.cmap. Ver _get_info.
        self.assertIsInstance(info, dict)
        self.assertNotIn('dengue_grid', info)
        self.assertNotIn('chik_grid', info)

    def test_step(self):
        self.Env.reset()
        observation, reward, terminated, truncated, info = self.Env.step(self.Env.action_space.sample())
        self.assertIsInstance(reward, float)
        self.assertIn('clinical_diagnostic', observation)

    def _first_active_case_id(self, env):
        for _ in range(env.epilength + 2):
            if env.cases_t:
                return env.get_case_id(env.cases_t[0])
            env.step(tuple())
        return None

    def test_test_results_persist_across_steps(self):
        # lab_delay_days=0 isola a persistência (resultado aplicado no mesmo dia).
        env = DengueDiagnosticsEnv(size=100, episize=40, epilength=8, lab_delay_days=0)
        env.reset(seed=42)

        case_id = self._first_active_case_id(env)
        self.assertIsNotNone(case_id)

        env.step(((case_id, 0),))  # dengue lab test
        status = env.obs_cases.loc[case_id, "testd"]
        self.assertNotEqual(status, 0)

        # Advancing a day with no action must NOT wipe the previous test result.
        env.step(tuple())
        self.assertEqual(env.obs_cases.loc[case_id, "testd"], status)

    def test_lab_result_respects_delay(self):
        # Com atraso, o laudo só fica disponível `lab_delay_days` depois do pedido.
        delay = 3
        env = DengueDiagnosticsEnv(
            size=100, episize=40, epilength=30, lab_delay_days=delay
        )
        env.reset(seed=42)

        case_id = self._first_active_case_id(env)
        self.assertIsNotNone(case_id)

        day_ordered = env.t
        env.step(((case_id, 0),))  # pede o teste de dengue hoje
        # O resultado ainda não chegou (fila pendente).
        self.assertEqual(env.obs_cases.loc[case_id, "testd"], 0)

        # Avança os dias até o laudo amadurecer.
        while env.t <= day_ordered + delay:
            env.step(tuple())

        self.assertNotEqual(env.obs_cases.loc[case_id, "testd"], 0)

    def test_agent_diagnosis_persists(self):
        env = DengueDiagnosticsEnv(size=100, episize=40, epilength=8)
        env.reset(seed=7)

        case_id = self._first_active_case_id(env)
        self.assertIsNotNone(case_id)

        env.step(((case_id, 6),))  # conclude OTHER -> agent_diagnosis = 2 (other)
        self.assertEqual(env.obs_cases.loc[case_id, "agent_diagnosis"], 2)

        env.step(tuple())
        self.assertEqual(env.obs_cases.loc[case_id, "agent_diagnosis"], 2)

    def test_dengue_test_conditioned_on_true_disease(self):
        env = DengueDiagnosticsEnv(size=100, episize=40, epilength=8)
        env.reset(seed=1)

        dengue_ids = env.real_cases[env.real_cases.disease == 0].index
        chik_ids = env.real_cases[env.real_cases.disease == 1].index
        self.assertGreater(len(dengue_ids), 0)
        self.assertGreater(len(chik_ids), 0)

        dengue_positive = [env._dengue_lab_test(int(dengue_ids[0])) for _ in range(300)].count(2)
        chik_positive_on_dengue_test = [env._dengue_lab_test(int(chik_ids[0])) for _ in range(300)].count(2)

        # A dengue test should be positive far more often on a true dengue case.
        self.assertGreater(dengue_positive, chik_positive_on_dengue_test)

    def test_epi_confirm_returns_binary(self):
        env = DengueDiagnosticsEnv(size=100, episize=40, epilength=8)
        env.reset(seed=3)

        case_id = self._first_active_case_id(env)
        self.assertIsNotNone(case_id)
        result = env._epi_confirm(case_id)
        self.assertIn(result, (0, 1))

    def test_full_episode_runs(self):
        env = DengueDiagnosticsEnv(size=100, episize=40, epilength=8)
        env.reset(seed=5)
        terminated = False
        steps = 0
        while not terminated and steps < 200:
            action = tuple((env.get_case_id(c), 3) for c in env.cases_t)
            _, _, terminated, _, _ = env.step(action)
            steps += 1
        self.assertTrue(terminated)

    def test_horizon_is_derived_from_epilength_and_delays(self):
        # Horizonte = (epilength - 1) + max(reward_delay, lab_delay).
        env = DengueDiagnosticsEnv(
            size=100, episize=40, epilength=20, reward_delay_days=5, lab_delay_days=3
        )
        self.assertEqual(env.horizon, 19 + 5)

        env2 = DengueDiagnosticsEnv(
            size=100, episize=40, epilength=20, reward_delay_days=2, lab_delay_days=7
        )
        self.assertEqual(env2.horizon, 19 + 7)

        # settle_days explícito tem prioridade.
        env3 = DengueDiagnosticsEnv(
            size=100, episize=40, epilength=20, reward_delay_days=5, settle_days=0
        )
        self.assertEqual(env3.horizon, 19)

    def test_episode_terminates_exactly_at_horizon(self):
        env = DengueDiagnosticsEnv(size=100, episize=40, epilength=8, settle_days=2)
        env.reset(seed=5)
        self.assertEqual(env.horizon, 7 + 2)
        terminated = False
        last_t = None
        steps = 0
        while not terminated and steps < 200:
            last_t = env.t
            _, _, terminated, _, _ = env.step(tuple())
            steps += 1
        # O passo que termina é o que processa o dia == horizon.
        self.assertEqual(last_t, env.horizon)

    def test_reset_regenerates_world_from_seed(self):
        """Cada seed deve gerar um surto diferente; a mesma seed reproduz."""
        env = DengueDiagnosticsEnv(
            size=200, episize=80, epilength=20, randomize_outbreak=True
        )
        env.reset(seed=1)
        day1_seed1 = len(env.real_cases[env.real_cases.t == env.start_day])
        centers_seed1 = (env.dengue_center, env.chik_center, env.dengue_r0, env.chik_r0)
        totals_seed1 = (env.world.dengue_total, env.world.chik_total)

        env.reset(seed=2)
        day1_seed2 = len(env.real_cases[env.real_cases.t == env.start_day])
        centers_seed2 = (env.dengue_center, env.chik_center, env.dengue_r0, env.chik_r0)

        self.assertNotEqual(centers_seed1, centers_seed2)
        self.assertLess(env.chik_r0, env.dengue_r0)
        self.assertLess(totals_seed1[1], totals_seed1[0])

        env.reset(seed=1)
        day1_repeat = len(env.real_cases[env.real_cases.t == env.start_day])
        centers_repeat = (env.dengue_center, env.chik_center, env.dengue_r0, env.chik_r0)
        self.assertEqual(centers_seed1, centers_repeat)
        self.assertEqual(day1_seed1, day1_repeat)
        # Seeds diferentes => surtos distintos (centros/R0; contagens podem coincidir).
        self.assertNotEqual(centers_seed1[:2], centers_seed2[:2])


if __name__ == '__main__':
    unittest.main()
