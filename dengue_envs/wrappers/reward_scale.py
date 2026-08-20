"""Reescala a recompensa vista pelo agente durante o TREINO.

Motivação (medido, ver ACHADOS.md §3.4): a recompensa por passo tem média ~+1,3
mas desvio ~16 e chega a variar de −90 a +210, porque os desfechos atrasados de
dezenas de casos maturam no mesmo dia. Com ~500 passos por episódio, o retorno
fica na casa dos ±2000.

Fazer uma rede prever alvos dessa magnitude com erro quadrático produz gradientes
enormes e é a receita clássica de divergência do DQN — foi o que observamos em
todas as rodadas (oscilações de +1019 para −7245 entre épocas consecutivas).

Multiplicar a recompensa por uma constante positiva **não altera a política
ótima** (a ordem entre políticas é preservada), apenas coloca os alvos de TD numa
escala tratável. É a mesma motivação do *reward clipping* do DQN original do
Atari, porém sem distorcer as proporções entre as ações.

Usar **apenas no treino**: a avaliação e o benchmark devem reportar a recompensa
na escala original, senão os números deixam de ser comparáveis com os resultados
anteriores.
"""
from __future__ import annotations

import gymnasium as gym


class RewardScaleWrapper(gym.RewardWrapper):
    """Multiplica a recompensa por `scale` (não muda a política ótima)."""

    def __init__(self, env, scale: float = 0.05):
        super().__init__(env)
        if scale <= 0:
            raise ValueError(f"reward_scale deve ser > 0; recebido {scale}")
        self.scale = float(scale)

    def reward(self, reward):
        return float(reward) * self.scale
