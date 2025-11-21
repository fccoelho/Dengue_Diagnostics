# dengue_wrappers.py
import gymnasium as gym
import numpy as np
from dengue_envs.envs.dengue_diagnostics import DengueDiagnosticsEnv
from gymnasium import spaces


class DengueWrapper(gym.ObservationWrapper):
    """
    Converte a observação Dict do DengueDiagnosticsEnv para um tensor 4-channel.
    (Baseado na Célula 4 do seu notebook)
    """

    def __init__(self, env: DengueDiagnosticsEnv):
        super().__init__(env)
        world_size = self.unwrapped.size
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(4, world_size, world_size),
            dtype=np.uint8
        )
        self._world_size = world_size

    def reset(self, **kwargs):
        obs_dict, info_dict = self.env.reset(**kwargs)
        tensor_obs = self.observation(obs_dict)
        return tensor_obs, info_dict

    def step(self, action):
        obs_dict, reward, terminated, truncated, info_dict = self.env.step(action)
        tensor_obs = self.observation(obs_dict)
        return tensor_obs, reward, terminated, truncated, info_dict

    def observation(self, obs_dict):
        """
        Converte o dicionário de observação em um tensor.
        """
        tensor = np.zeros((4, self._world_size, self._world_size), dtype=np.uint8)

        # Canal 0: Diagnóstico clínico (0,1,2 -> 1,2,3)
        for case in obs_dict.get('clinical_diagnostic', []):
            x, y, diag = case
            if 0 <= x < self._world_size and 0 <= y < self._world_size:
                tensor[0, int(x), int(y)] = diag + 1

        # Canal 1: Status Teste Dengue (testd) (0-3 -> 1-4)
        for case_id, status in obs_dict.get('testd', []):
            x, y = self.unwrapped.get_case_xy(case_id)
            if 0 <= x < self._world_size and 0 <= y < self._world_size:
                tensor[1, int(x), int(y)] = status + 1

        # Canal 2: Status Teste Chik (testc) (0-3 -> 1-4)
        for case_id, status in obs_dict.get('testc', []):
            x, y = self.unwrapped.get_case_xy(case_id)
            if 0 <= x < self._world_size and 0 <= y < self._world_size:
                tensor[2, int(x), int(y)] = status + 1

        # Canal 3: Máscara de casos ativos
        current_t = self.unwrapped.t
        for _, case_data in self.unwrapped.obs_cases.iterrows():
            if case_data.t == current_t:
                x, y = int(case_data.x), int(case_data.y)
                if 0 <= x < self._world_size and 0 <= y < self._world_size:
                    tensor[3, x, y] = 1.0

        return tensor


class CaseByCaseWrapper(gym.Wrapper):
    """
    Transforma o problema de "N decisões por passo" para "1 decisão por passo, N vezes".
    Isto torna o action_space um simples Discrete(6).
    """

    def __init__(self, env: DengueWrapper):
        super().__init__(env)

        self.action_space = spaces.Discrete(6)

        self.observation_space = spaces.Dict({
            "map": env.observation_space,
            "case_coords": spaces.Box(
                low=0, high=self.unwrapped.size, shape=(2,), dtype=np.float32
            )
        })

        self.active_cases = []
        self.case_iterator = iter(self.active_cases)
        self.pending_actions = []
        self.current_case = (0, 0, 0)  # (case_id, x, y)
        self._current_map_obs = None

    def _get_active_cases(self):
        """Pega os casos ativos no timestep atual do ambiente base."""
        df = self.unwrapped.obs_cases
        t = self.unwrapped.t
        active_df = df[df.t == t]
        cases = []
        for case in active_df.itertuples():
            cases.append((case.Index, int(case.x), int(case.y)))
        return cases

    def _make_obs(self):
        """Cria a observação Dict para o agente."""
        return {
            "map": self._current_map_obs,
            "case_coords": np.array([self.current_case[1], self.current_case[2]], dtype=np.float32)
        }

    def _next_case(self):
        """
        Avança para o próximo caso. Se não houver casos neste dia,
        avança os dias no ambiente automaticamente até encontrar casos
        ou o episódio terminar.
        """
        while True:
            try:
                # 1. Tenta pegar o próximo caso da lista atual (do dia atual)
                self.current_case = next(self.case_iterator)

                # Se conseguiu, retorna a observação para o agente agir
                return self._make_obs(), 0.0, False, False, {}

            except StopIteration:
                # 2. Acabaram os casos deste timestep (ou a lista estava vazia).
                # Hora de avançar o ambiente real.

                # Envia as ações acumuladas deste dia
                action_tuple = tuple(self.pending_actions)
                self.pending_actions = []

                # Chama o step do ambiente base (avança o tempo t -> t+1)
                obs_tensor, reward, terminated, truncated, info = self.env.step(action_tuple)

                # Atualiza o mapa global
                self._current_map_obs = obs_tensor

                # Se o episódio acabou, retorna o fim
                if terminated or truncated:
                    self.current_case = (0, 0, 0)  # Agora sim, um dummy final seguro
                    return self._make_obs(), reward, terminated, truncated, info

                # Se não acabou, carrega os casos do NOVO dia
                self.active_cases = self._get_active_cases()
                self.case_iterator = iter(self.active_cases)

                # O loop 'while True' vai voltar ao topo.
                # Se houver casos no novo dia, o 'try' vai funcionar e retornar.
                # Se a lista estiver vazia (dia sem casos), vai cair no 'except' de novo
                # e avançar mais um dia automaticamente, sem incomodar o agente.

    def reset(self, **kwargs):
        obs_tensor, info = self.env.reset(**kwargs)
        self._current_map_obs = obs_tensor
        self.pending_actions = []

        self.active_cases = self._get_active_cases()
        self.case_iterator = iter(self.active_cases)

        try:
            self.current_case = next(self.case_iterator)
        except StopIteration:
            self.current_case = (0, 0, 0)  # dummy

        return self._make_obs(), info

    def step(self, action: int):
        """
        O agente passa uma única ação (0-5) para o caso atual.
        """
        self.pending_actions.append((int(self.current_case[0]), int(action)))

        return self._next_case()