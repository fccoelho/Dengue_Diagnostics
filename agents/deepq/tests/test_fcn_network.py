import unittest
from importlib.metadata import files

import torch
from agents.deepq.network import DengueNet


class TestDengueNet(unittest.TestCase):

    def setUp(self):
        """
        Configura os parâmetros padrão para todos os testes.
        Isso é executado antes de cada método de teste.
        """
        self.map_shape = (4, 400, 400)
        self.action_shape = 6
        self.batch_size = 8

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        try:
            self.model = DengueNet(
                map_shape=self.map_shape,
                action_shape=self.action_shape,
                device=self.device
            ).to(self.device)
        except Exception as e:
            self.fail(f"Falha ao inicializar DengueNet: {e}")

    def test_initialization(self):
        """
        Testa se o modelo e seus submódulos foram inicializados
        e movidos para o dispositivo correto.
        """
        self.assertIsInstance(self.model, DengueNet)

        param = next(self.model.parameters())
        self.assertEqual(param.device.type, self.device)

        head_param = next(self.model.head.parameters())
        self.assertEqual(head_param.device.type, self.device)

    def test_forward_pass_single_item(self):
        """
        Testa o 'forward' com uma única observação (sem dimensão de batch).
        O modelo deve adicionar internamente a dimensão do batch.
        """
        dummy_map = torch.rand(self.map_shape, device=self.device)
        dummy_coords = torch.rand(2, device=self.device)

        single_obs = {
            "map": dummy_map,
            "case_coords": dummy_coords
        }

        with torch.no_grad():
            q_values, state = self.model(single_obs)

        self.assertIsNone(state)  # O 'state' não é usado, deve ser None
        self.assertEqual(q_values.shape, (1, self.action_shape))
        self.assertEqual(q_values.device.type, self.device)

    def test_forward_pass_batch(self):
        """
        Testa o 'forward' com um batch de observações.
        """
        batch_map_shape = (self.batch_size, *self.map_shape)  # (B, 4, 400, 400)
        batch_coords_shape = (self.batch_size, 2)  # (B, 2)

        dummy_map_batch = torch.rand(batch_map_shape, device=self.device)
        dummy_coords_batch = torch.rand(batch_coords_shape, device=self.device)

        batch_obs = {
            "map": dummy_map_batch,
            "case_coords": dummy_coords_batch
        }

        with torch.no_grad():
            q_values, state = self.model(batch_obs)

        self.assertIsNone(state)
        self.assertEqual(q_values.shape, (self.batch_size, self.action_shape))
        self.assertEqual(q_values.device.type, self.device)

    def test_backward_pass_gradient_flow(self):
        """
        Testa se os gradientes fluem corretamente (backward pass).
        Isso verifica se não há partes "quebradas" no gráfico computacional.
        """

        batch_map_shape = (self.batch_size, *self.map_shape)
        batch_coords_shape = (self.batch_size, 2)
        dummy_map_batch = torch.rand(batch_map_shape, device=self.device)
        dummy_coords_batch = torch.rand(batch_coords_shape, device=self.device)

        batch_obs = {
            "map": dummy_map_batch,
            "case_coords": dummy_coords_batch
        }

        self.model.zero_grad()

        q_values, _ = self.model(batch_obs)

        loss = q_values.sum()
        try:
            loss.backward()
        except Exception as e:
            self.fail(f"Backward pass falhou com a exceção: {e}")

        param = next(self.model.head.parameters())
        self.assertIsNotNone(param.grad)
        self.assertGreater(torch.abs(param.grad).sum(), 0)


if __name__ == "__main__":
    unittest.main()