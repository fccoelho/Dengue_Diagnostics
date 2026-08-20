"""Baseline "testar tudo" (teto de informação do ambiente).

Este agente pede um teste de laboratório para **todos** os casos (ação 0). Não é
uma política realista — testagem universal é justamente o que o trabalho quer
evitar — mas é a referência que faltava, por dois motivos:

1. **Teto de acurácia.** Como cada caso recebe UMA única ação e o laudo tem
   ``inconclusive_prob`` e erro próprios, nenhuma política pode superar muito
   este agente. O teto teórico é

       acc_max ≈ (1 - p_inconclusivo) · sens_lab + p_inconclusivo · acc_clínica

   ou seja, ~0,88 com os parâmetros atuais (sens=espec=0,9; p_inc=0,1). Serve
   para saber quanta acurácia ainda está "na mesa" para o RL.

2. **Teto de recompensa.** Com os pesos atuais, deixar um caso errado e não
   testado custa -10, enquanto testar custa apenas 1,0. Isso torna a testagem
   universal quase ótima em recompensa — um diagnóstico importante sobre o
   desenho da função de recompensa.

Comparar um agente treinado contra este baseline mostra o quanto ele realmente
economiza testes **sem** perder desempenho.
"""
from __future__ import annotations

from agents.base import EpisodeRunner

# Ação "testar dengue" no CaseByCaseWrapper.
TEST_DENGUE = 0


class TestAllAgentRunner(EpisodeRunner):
    """Baseline que solicita teste de laboratório para todos os casos."""

    name = "testall"

    def choose_action(self, env) -> int:
        return TEST_DENGUE
