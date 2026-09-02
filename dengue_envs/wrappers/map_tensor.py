"""Wrapper de observação: converte o Dict do ambiente em tensor 6-canais.

Versão canônica (Fase 1) do `DengueWrapper`. A cópia legada em
`agents/deepq/files/dengue_wrapper.py` reexporta esta.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from dengue_envs.envs.dengue_diagnostics import DengueDiagnosticsEnv


class DengueWrapper(gym.ObservationWrapper):
    """Converte a observação Dict em um tensor (6, map_size, map_size) uint8.

    Canais:
      0 -> diagnóstico clínico (0,1,2 mapeados para 1,2,3)
      1 -> status do teste de dengue (0-3 -> 1-4)
      2 -> status do teste de chik (0-3 -> 1-4)
      3 -> máscara dos casos ativos no dia atual
      4 -> densidade de casos CONFIRMADOS de dengue (laudo positivo)
      5 -> densidade de casos CONFIRMADOS de chik (laudo positivo)

    Os canais 4 e 5 são o mapa epidemiológico que o **agente** construiu, não a
    verdade do gerador. Eles dão a visão global ("quanto já sei do surto"), que
    é o que permite julgar se vale comprar uma `epi_confirm`; a densidade
    *local* do caso atual vai por `case_features`, porque o pooling do encoder
    dissolve a célula individual.

    **`map_size`: resolução da observação, independente do tamanho do mundo.**
    O encoder termina em ``AdaptiveAvgPool2d((6, 6))``, então um mapa 400×400
    vira 6×6 de qualquer maneira — a resolução extra é descartada pela rede,
    mas paga o preço inteiro no caminho dos dados. Medido (batch 64, CUDA):

    ===========  ========  ==============  ============  ===========
    resolução    KB/obs    GPU fwd+bwd     gather+H2D    buffer 6k
    ===========  ========  ==============  ============  ===========
    400×400      937,5     1,7 s/100upd    7,6 s/100upd  5,36 GB
    100×100       58,6     1,0 s/100upd    0,7 s/100upd  0,34 GB
    ===========  ========  ==============  ============  ===========

    E o gargalo é esse caminho, não o ambiente: a coleta de 1000 passos custa
    ~0,8 s contra ~30 s das 100 atualizações de gradiente que a acompanham.
    """

    def __init__(self, env: DengueDiagnosticsEnv, map_size: int | None = None):
        super().__init__(env)
        world_size = int(self.unwrapped.size)
        map_size = world_size if map_size is None else int(map_size)
        if map_size <= 0 or map_size > world_size:
            raise ValueError(
                f"map_size deve estar em [1, {world_size}]; recebido {map_size}"
            )
        if world_size % map_size:
            raise ValueError(
                f"map_size ({map_size}) deve dividir o tamanho do mundo "
                f"({world_size}) para que as células agreguem blocos iguais"
            )

        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(6, map_size, map_size),
            dtype=np.uint8,
        )
        self._world_size = world_size
        self._map_size = map_size
        # Fator de agregação: quantas células do mundo cabem numa da observação.
        self._block = world_size // map_size

    def _cell(self, v) -> int:
        """Converte uma coordenada do mundo para a célula da observação."""
        return int(v) // self._block

    def _in_world(self, x, y) -> bool:
        return 0 <= x < self._world_size and 0 <= y < self._world_size

    def _downsample_counts(self, m: np.ndarray) -> np.ndarray:
        """Agrega um mapa de contagens do mundo para a resolução da observação."""
        if self._block == 1:
            return m
        b, n = self._block, self._map_size
        return m.reshape(n, b, n, b).sum(axis=(1, 3))

    def reset(self, **kwargs):
        obs_dict, info_dict = self.env.reset(**kwargs)
        return self.observation(obs_dict), info_dict

    def step(self, action):
        obs_dict, reward, terminated, truncated, info_dict = self.env.step(action)
        return self.observation(obs_dict), reward, terminated, truncated, info_dict

    def observation(self, obs_dict):
        n = self._map_size
        tensor = np.zeros((6, n, n), dtype=np.uint8)

        # Canal 0: diagnóstico clínico
        for case in obs_dict.get("clinical_diagnostic", []):
            x, y, diag = case
            if self._in_world(x, y):
                tensor[0, self._cell(x), self._cell(y)] = diag + 1

        # Canal 1: status teste dengue
        for case_id, status in obs_dict.get("testd", []):
            x, y = self.unwrapped.get_case_xy(case_id)
            if self._in_world(x, y):
                tensor[1, self._cell(x), self._cell(y)] = status + 1

        # Canal 2: status teste chik
        for case_id, status in obs_dict.get("testc", []):
            x, y = self.unwrapped.get_case_xy(case_id)
            if self._in_world(x, y):
                tensor[2, self._cell(x), self._cell(y)] = status + 1

        # Canal 3: máscara de casos ativos.
        # Filtro vetorizado + scatter NumPy em vez de `iterrows()` (que é O(N)
        # por observação e domina o tempo de step quando há muitos casos).
        current_t = self.unwrapped.t
        active = self.unwrapped.obs_cases
        if not active.empty:
            active = active[active.t == current_t]
            if not active.empty:
                xs = active.x.to_numpy(dtype=np.intp)
                ys = active.y.to_numpy(dtype=np.intp)
                valid = (
                    (xs >= 0) & (xs < self._world_size)
                    & (ys >= 0) & (ys < self._world_size)
                )
                tensor[3, xs[valid] // self._block, ys[valid] // self._block] = 1

        # Canais 4 e 5: mapa de confirmados por laudo, acumulado pelo agente.
        # `clip` porque o tensor é uint8 e a contagem, agregada em blocos, pode
        # em tese passar de 255.
        u = self.unwrapped
        tensor[4] = np.clip(self._downsample_counts(u.confirmed_dmap), 0, 255)
        tensor[5] = np.clip(self._downsample_counts(u.confirmed_cmap), 0, 255)

        return tensor
