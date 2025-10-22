import unittest
import torch
from tianshou.data import Batch
from agents.deepq.files.fcn_network import DengueFCN


class TestDengueFCN(unittest.TestCase):

    def setUp(self):
        """
        Configura parâmetros comuns para todos os testes.
        Este método é executado antes de cada teste individual.
        """
        self.batch_size = 4
        self.input_channels = 4
        self.height = 100
        self.width = 100
        self.input_shape = (self.input_channels, self.height, self.width)

    def test_inicializacao_da_rede(self):
        """
        Testa se a rede pode ser instanciada sem erros.
        """
        try:
            net = DengueFCN(input_shape=self.input_shape)
            self.assertIsInstance(net, DengueFCN)
        except Exception as e:
            self.fail(f"A instanciação da DengueFCN falhou com o erro: {e}")

    def test_formato_da_saida_do_forward_pass(self):
        """
        Verifica se a saída da rede tem o shape (Batch, 6, H, W).
        Este é o teste mais importante para a arquitetura.
        """
        net = DengueFCN(input_shape=self.input_shape)
        dummy_input = torch.randn(self.batch_size, *self.input_shape)

        output, _ = net(dummy_input)

        expected_shape = (self.batch_size, 6, self.height, self.width)
        self.assertEqual(output.shape, expected_shape)

    def test_compatibilidade_com_batch_do_tianshou(self):
        """
        Garante que a rede consegue processar a observação quando ela vem
        dentro de um objeto `Batch` do Tianshou.
        """
        net = DengueFCN(input_shape=self.input_shape)
        dummy_input_tensor = torch.randn(self.batch_size, *self.input_shape)

        tianshou_batch = Batch(obs=dummy_input_tensor)

        output, _ = net(tianshou_batch)

        expected_shape = (self.batch_size, 6, self.height, self.width)
        self.assertEqual(output.shape, expected_shape)


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)