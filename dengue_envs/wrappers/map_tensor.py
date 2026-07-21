"""Wrapper de observação: converte o Dict do ambiente em tensor 4-canais.

Versão canônica (Fase 1) do `DengueWrapper`, idêntica em comportamento à que
vinha sendo usada no treino em `agents/deepq/files/dengue_wrapper.py`. Aquela
cópia legada permanece intacta para não quebrar scripts existentes.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from dengue_envs.envs.dengue_diagnostics import DengueDiagnosticsEnv


class DengueWrapper(gym.ObservationWrapper):
    """Converte a observação Dict em um tensor (4, size, size) em uint8.

    Canais:
      0 -> diagnóstico clínico (0,1,2 mapeados para 1,2,3)
      1 -> status do teste de dengue (0-3 -> 1-4)
      2 -> status do teste de chik (0-3 -> 1-4)
      3 -> máscara dos casos ativos no dia atual
    """

    def __init__(self, env: DengueDiagnosticsEnv):
        super().__init__(env)
        world_size = self.unwrapped.size

        self.observation_space = spaces.Box(
            low=0, high=5,
            shape=(4, world_size, world_size),
            dtype=np.uint8,
        )
        self._world_size = world_size

    def reset(self, **kwargs):
        obs_dict, info_dict = self.env.reset(**kwargs)
        return self.observation(obs_dict), info_dict

    def step(self, action):
        obs_dict, reward, terminated, truncated, info_dict = self.env.step(action)
        return self.observation(obs_dict), reward, terminated, truncated, info_dict

    def observation(self, obs_dict):
        tensor = np.zeros((4, self._world_size, self._world_size), dtype=np.uint8)

        # Canal 0: diagnóstico clínico
        for case in obs_dict.get("clinical_diagnostic", []):
            x, y, diag = case
            if 0 <= x < self._world_size and 0 <= y < self._world_size:
                tensor[0, int(x), int(y)] = diag + 1

        # Canal 1: status teste dengue
        for case_id, status in obs_dict.get("testd", []):
            x, y = self.unwrapped.get_case_xy(case_id)
            if 0 <= x < self._world_size and 0 <= y < self._world_size:
                tensor[1, int(x), int(y)] = status + 1

        # Canal 2: status teste chik
        for case_id, status in obs_dict.get("testc", []):
            x, y = self.unwrapped.get_case_xy(case_id)
            if 0 <= x < self._world_size and 0 <= y < self._world_size:
                tensor[2, int(x), int(y)] = status + 1

        # Canal 3: máscara de casos ativos
        current_t = self.unwrapped.t
        for _, case_data in self.unwrapped.obs_cases.iterrows():
            if case_data.t == current_t:
                x, y = int(case_data.x), int(case_data.y)
                if 0 <= x < self._world_size and 0 <= y < self._world_size:
                    tensor[3, x, y] = 1

        return tensor
