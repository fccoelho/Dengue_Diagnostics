import unittest
from dengue_envs.envs.cnn_wrapper import *


class TestObservationProcessing(unittest.TestCase):
    def test_converte_dicionario_para_tensor_corretamente(self):
        """
        Testa a função `process_observation_to_tensor` de forma isolada.
        """
        world_size, current_t = 20, 5
        all_cases_df = pd.DataFrame([
            {'t': 5, 'x': 2, 'y': 3}, {'t': 4, 'x': 8, 'y': 8}
        ], index=[1, 2])
        case_coords_map = {10: (11, 12), 20: (13, 14)}
        obs_dict = {
            'clinical_diagnostic': [(1, 1, 0)], 'testd': [(10, 2)], 'testc': [(20, 1)]
        }

        tensor = process_observation_to_tensor(
            obs_dict, world_size, current_t, all_cases_df, case_coords_map
        )

        self.assertEqual(tensor.shape, (4, world_size, world_size))
        self.assertEqual(tensor[0, 1, 1], 1.0)  # Clinical (diag=0 -> val=1)
        self.assertEqual(tensor[1, 12, 11], 3.0)  # TestD (status=2 -> val=3)
        self.assertEqual(tensor[2, 14, 13], 2.0)  # TestC (status=1 -> val=2)
        self.assertEqual(tensor[3, 3, 2], 1.0)  # Máscara de caso ativo (t=5)
        self.assertEqual(tensor[3, 8, 8], 0.0)  # Máscara de caso inativo (t=4)
        self.assertAlmostEqual(np.sum(tensor), 1.0 + 3.0 + 2.0 + 1.0)

    def test_dicionario_de_observacao_vazio(self):
        """
        Verifica o comportamento quando o obs_dict de entrada está vazio.
        """
        world_size, current_t = 20, 5
        all_cases_df = pd.DataFrame([{'t': 5, 'x': 2, 'y': 3}], index=[1])
        case_coords_map = {}
        obs_dict = {}

        tensor = process_observation_to_tensor(
            obs_dict, world_size, current_t, all_cases_df, case_coords_map
        )

        self.assertEqual(np.sum(tensor[0:3]), 0.0)
        # Canal da máscara de casos ativos (3) deve ser preenchido normalmente
        self.assertEqual(tensor[3, 3, 2], 1.0)
        self.assertEqual(np.sum(tensor[3]), 1.0)

    def test_nenhum_caso_ativo_no_timestep(self):
        """
        Verifica se o canal 3 fica zerado se não houver casos no timestep atual.
        """
        world_size, current_t = 20, 99  # t=99 não existe no DataFrame
        all_cases_df = pd.DataFrame([{'t': 5, 'x': 2, 'y': 3}], index=[1])
        obs_dict = {'clinical_diagnostic': [(1, 1, 0)]}
        case_coords_map = {}

        tensor = process_observation_to_tensor(
            obs_dict, world_size, current_t, all_cases_df, case_coords_map
        )

        self.assertEqual(np.sum(tensor[3]), 0.0)
        self.assertEqual(tensor[0, 1, 1], 1.0)

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)