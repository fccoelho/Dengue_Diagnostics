import unittest
from unittest.mock import patch
from dengue_envs.envs.cnn_collector import *


class TestActionGeneration(unittest.TestCase):

    def test_gera_acoes_com_base_no_maior_q_valor(self):
        """
        Testa o caminho de 'exploitation' (eps=0), garantindo que
        a ação com o maior Q-valor seja escolhida.
        """
        world_size, current_t = 20, 5

        q_map = np.zeros((1, 6, world_size, world_size))
        q_map[0, :, 3, 2] = [10, 20, 30, 40, 50, 0]

        env_data = [{
            'current_t': current_t,
            'obs_cases': pd.DataFrame([{'t': 5, 'x': 2, 'y': 3}], index=[101])
        }]

        actions = generate_actions_from_q_maps(q_map, env_data, epsilon=0.0)

        expected_actions = [((101, 4),)]
        self.assertEqual(actions, expected_actions)

    @patch('random.randrange')
    def test_gera_acoes_aleatorias_com_exploracao(self, mock_randrange):
        """
        Testa o caminho de 'exploration' (eps=1), garantindo que
        uma ação aleatória seja escolhida.
        """
        mock_randrange.return_value = 5

        q_map = np.zeros((1, 6, 20, 20))
        env_data = [{
            'current_t': 5,
            'obs_cases': pd.DataFrame([{'t': 5, 'x': 2, 'y': 3}], index=[101])
        }]

        actions = generate_actions_from_q_maps(q_map, env_data, epsilon=1.0)

        expected_actions = [((101, 5),)]
        self.assertEqual(actions, expected_actions)
        mock_randrange.assert_called_once_with(6)  # Garante que a função aleatória foi chamada


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)