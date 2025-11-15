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
            low=0, high=4,  # Status 0-3 são codificados como 1-4
            shape=(4, world_size, world_size),  # Formato Channels-first
            dtype=np.float32
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
        tensor = np.zeros((4, self._world_size, self._world_size), dtype=np.float32)

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
        Avança para o próximo caso ou, se não houver mais casos,
        executa o step do ambiente real.
        """
        try:
            # Tenta pegar o próximo caso da lista
            self.current_case = next(self.case_iterator)
            # Retorna a observação do novo caso, com recompensa 0 (decisão intermediária)
            return self._make_obs(), 0.0, False, False, {}

        except StopIteration:
            if not self.pending_actions:
                self.pending_actions = []

            action_tuple = tuple(self.pending_actions)

            obs_tensor, reward, terminated, truncated, info = self.env.step(action_tuple)

            self._current_map_obs = obs_tensor

            self.pending_actions = []

            if terminated or truncated:
                self.current_case = (0, 0, 0)
                return self._make_obs(), reward, terminated, truncated, info

            self.active_cases = self._get_active_cases()
            self.case_iterator = iter(self.active_cases)

            try:
                self.current_case = next(self.case_iterator)
            except StopIteration:
                # O novo timestep também não tem casos. Raro, mas possível.
                # Retorna o estado atual e espera o próximo 'step'.
                self.current_case = (0, 0, 0)  # dummy

            return self._make_obs(), reward, terminated, truncated, info

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