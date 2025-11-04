import unittest
import numpy as np
import pandas as pd
from gymnasium import spaces
from unittest.mock import MagicMock, patch
from agents.deepq.files.dengue_wrapper import DengueWrapper, CaseByCaseWrapper
from dengue_envs.envs.dengue_diagnostics import DengueDiagnosticsEnv


class TestDengueWrapper(unittest.TestCase):

    def setUp(self):
        """
        Cria um mock do ambiente base (DengueDiagnosticsEnv).
        """
        self.base_env_mock = MagicMock(spec=DengueDiagnosticsEnv)

        self.base_env_mock.size = 20
        self.base_env_mock.unwrapped.size = 20

        self.wrapper = DengueWrapper(self.base_env_mock)

    def test_init_observation_space(self):
        """
        Testa se o wrapper define corretamente o novo observation_space.
        """
        expected_shape = (4, 20, 20)
        self.assertIsInstance(self.wrapper.observation_space, spaces.Box)
        self.assertEqual(self.wrapper.observation_space.shape, expected_shape)
        self.assertEqual(self.wrapper.observation_space.dtype, np.float32)

    def test_observation_logic(self):
        """
        Testa a lógica de conversão de dicionário para tensor
        no método .observation().
        """
        self.base_env_mock.unwrapped.t = 1

        def mock_get_case_xy(case_id):
            if case_id == 10: return (2, 2)
            if case_id == 11: return (3, 3)
            return (0, 0)

        self.base_env_mock.unwrapped.get_case_xy = mock_get_case_xy

        mock_df = pd.DataFrame([
            {'t': 0, 'x': 8, 'y': 8},
            {'t': 1, 'x': 4, 'y': 4},
            {'t': 1, 'x': 5, 'y': 5},
        ])
        self.base_env_mock.unwrapped.obs_cases = mock_df

        obs_dict = {
            "clinical_diagnostic": [(1, 1, 0)],
            "testd": [(10, 1)],
            "testc": [(11, 2)]
        }

        tensor_obs = self.wrapper.observation(obs_dict)

        self.assertEqual(tensor_obs.shape, (4, 20, 20))

        self.assertEqual(tensor_obs[0, 1, 1], 1.0)
        self.assertEqual(tensor_obs[1, 2, 2], 2.0)
        self.assertEqual(tensor_obs[2, 3, 3], 3.0)

        self.assertEqual(tensor_obs[3, 4, 4], 1.0)
        self.assertEqual(tensor_obs[3, 5, 5], 1.0)

        self.assertEqual(tensor_obs.sum(), 1.0 + 2.0 + 3.0 + 1.0 + 1.0)
        self.assertEqual(tensor_obs[0, 0, 0], 0.0)
        self.assertEqual(tensor_obs[3, 8, 8], 0.0)

    def test_reset(self):
        """Testa se o reset retorna o tensor processado."""
        dummy_dict_obs = {"clinical_diagnostic": [(1, 1, 0)]}
        dummy_info = {"key": "value"}

        self.base_env_mock.reset.return_value = (dummy_dict_obs, dummy_info)

        self.base_env_mock.t = 0
        self.base_env_mock.unwrapped.obs_cases = pd.DataFrame()

        obs, info = self.wrapper.reset()

        self.base_env_mock.reset.assert_called_once()

        self.assertIsInstance(obs, np.ndarray)
        self.assertEqual(obs.shape, (4, 20, 20))
        self.assertEqual(obs[0, 1, 1], 1.0)
        self.assertEqual(info, dummy_info)

    def test_step(self):
        """Testa se o step retorna o tensor processado."""
        dummy_dict_obs = {"clinical_diagnostic": [(2, 2, 1)]}
        dummy_info = {"key": "value"}
        dummy_action = ((0, 1),)  # Tupla de ação

        self.base_env_mock.step.return_value = (
            dummy_dict_obs, 1.0, False, False, dummy_info
        )

        self.base_env_mock.t = 1
        self.base_env_mock.unwrapped.obs_cases = pd.DataFrame()

        obs, rew, term, trunc, info = self.wrapper.step(dummy_action)

        self.base_env_mock.step.assert_called_once_with(dummy_action)

        # Verifica saídas
        self.assertIsInstance(obs, np.ndarray)
        self.assertEqual(obs[0, 2, 2], 2.0)
        self.assertEqual(rew, 1.0)
        self.assertFalse(term)
        self.assertEqual(info, dummy_info)


class TestCasetByCaseWrapper(unittest.TestCase):

    def setUp(self):
        """
        Cria um mock do DengueWrapper (o env do CaseByCaseWrapper).
        """
        self.dw_env_mock = MagicMock(spec=DengueWrapper)

        self.base_env_mock = MagicMock(spec=DengueDiagnosticsEnv)
        self.dw_env_mock.unwrapped = self.base_env_mock

        self.base_env_mock.size = 20
        self.dw_env_mock.observation_space = spaces.Box(
            low=0, high=4, shape=(4, 20, 20), dtype=np.float32
        )

        self.wrapper = CaseByCaseWrapper(self.dw_env_mock)

        self.dummy_tensor_obs = np.zeros((4, 20, 20), dtype=np.float32)

    def _mock_active_cases(self, t, cases_list):
        """Helper para mockar os casos ativos."""
        self.base_env_mock.t = t
        df_data = []
        for case in cases_list:
            df_data.append({'Index': case[0], 'x': case[1], 'y': case[2], 't': t})

        columns = ['Index', 'x', 'y', 't']

        if not df_data:
            self.base_env_mock.obs_cases = pd.DataFrame(columns=columns).set_index('Index')
        else:
            self.base_env_mock.obs_cases = pd.DataFrame(df_data, columns=columns).set_index('Index')

    def test_init_spaces(self):
        """Testa se os espaços de ação e observação estão corretos."""
        self.assertIsInstance(self.wrapper.action_space, spaces.Discrete)
        self.assertEqual(self.wrapper.action_space.n, 6)

        self.assertIsInstance(self.wrapper.observation_space, spaces.Dict)
        self.assertIn("map", self.wrapper.observation_space.spaces)
        self.assertIn("case_coords", self.wrapper.observation_space.spaces)
        self.assertEqual(self.wrapper.observation_space.spaces["map"].shape, (4, 20, 20))

    def test_reset_with_cases(self):
        """Testa o reset quando há casos ativos no t=0."""
        self.dw_env_mock.reset.return_value = (self.dummy_tensor_obs, {"info_key": 1})
        self._mock_active_cases(t=0, cases_list=[(100, 5, 5), (101, 8, 8)])

        obs, info = self.wrapper.reset()

        self.dw_env_mock.reset.assert_called_once()
        self.assertEqual(info, {"info_key": 1})

        self.assertEqual(self.wrapper.current_case, (100, 5, 5))
        self.assertTrue(np.array_equal(obs["case_coords"], np.array([5., 5.])))
        self.assertTrue(np.array_equal(obs["map"], self.dummy_tensor_obs))
        self.assertEqual(len(list(self.wrapper.case_iterator)), 1)

    def test_reset_no_cases(self):
        """Testa o reset quando NÃO há casos ativos no t=0."""
        self.dw_env_mock.reset.return_value = (self.dummy_tensor_obs, {})
        self._mock_active_cases(t=0, cases_list=[])

        obs, info = self.wrapper.reset()

        self.assertEqual(self.wrapper.current_case, (0, 0, 0))
        self.assertTrue(np.array_equal(obs["case_coords"], np.array([0., 0.])))
        self.assertEqual(len(list(self.wrapper.case_iterator)), 0)

    def test_step_loop_and_env_step(self):
        """
        Testa o ciclo de vida completo:
        1. Reset com 2 casos.
        2. step() para o caso 1 (retorna obs do caso 2, rew=0).
        3. step() para o caso 2 (chama o env.step() real, retorna obs do t=1).
        """
        self.dw_env_mock.reset.return_value = (self.dummy_tensor_obs, {})
        self._mock_active_cases(t=0, cases_list=[(100, 5, 5), (101, 8, 8)])
        self.wrapper.reset()

        self.assertEqual(self.wrapper.current_case, (100, 5, 5))
        obs1, rew1, term1, trunc1, info1 = self.wrapper.step(action=3)

        self.dw_env_mock.step.assert_not_called()
        self.assertEqual(rew1, 0.0)
        self.assertFalse(term1)
        self.assertEqual(self.wrapper.pending_actions, [(100, 3)])
        self.assertEqual(self.wrapper.current_case, (101, 8, 8))
        self.assertTrue(np.array_equal(obs1["case_coords"], np.array([8., 8.])))

        next_t_obs_tensor = np.ones((4, 20, 20), dtype=np.float32)
        self.dw_env_mock.step.return_value = (next_t_obs_tensor, 10.0, False, False, {"real_info": 1})

        with patch.object(self.wrapper, '_get_active_cases', return_value=[(200, 1, 1)]):
            obs2, rew2, term2, trunc2, info2 = self.wrapper.step(action=4)

        expected_action_tuple = ((100, 3), (101, 4))  # TUPLA de tuplas
        self.dw_env_mock.step.assert_called_once_with(expected_action_tuple)

        self.assertEqual(self.wrapper.pending_actions, [])

        self.assertEqual(rew2, 10.0)
        self.assertFalse(term2)
        self.assertEqual(info2, {"real_info": 1})

        self.assertEqual(self.wrapper.current_case, (200, 1, 1))

        self.assertTrue(np.array_equal(obs2["map"], next_t_obs_tensor))
        self.assertTrue(np.array_equal(obs2["case_coords"], np.array([1., 1.])))

    def test_step_and_terminate(self):
        """Testa se o wrapper lida com o término do episódio."""
        self.dw_env_mock.reset.return_value = (self.dummy_tensor_obs, {})
        self._mock_active_cases(t=0, cases_list=[(100, 5, 5)])  # 1 caso
        self.wrapper.reset()

        term_obs_tensor = np.full((4, 20, 20), 9.0, dtype=np.float32)
        self.dw_env_mock.step.return_value = (
            term_obs_tensor, 5.0, True, False, {"final_info": 1}
        )

        obs, rew, term, trunc, info = self.wrapper.step(action=1)

        self.dw_env_mock.step.assert_called_once_with(((100, 1),))
        self.assertEqual(rew, 5.0)
        self.assertTrue(term)
        self.assertFalse(trunc)
        self.assertEqual(info, {"final_info": 1})

        self.assertEqual(self.wrapper.current_case, (0, 0, 0))
        self.assertTrue(np.array_equal(obs["map"], term_obs_tensor))
        self.assertTrue(np.array_equal(obs["case_coords"], np.array([0., 0.])))

    if __name__ == "__main__":
        unittest.main()