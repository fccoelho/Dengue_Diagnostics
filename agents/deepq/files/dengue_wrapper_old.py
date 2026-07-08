import gymnasium as gym
import numpy as np
from dengue_envs.envs.dengue_diagnostics import DengueDiagnosticsEnv
from gymnasium import spaces


class DengueWrapper(gym.ObservationWrapper):
    """
    Converte a observação Dict do DengueDiagnosticsEnv para um tensor 4-channel.
    Mantém o tamanho original (sem downsampling) utilizando o menor tipo de dado (uint8).
    """

    def __init__(self, env: DengueDiagnosticsEnv):
        super().__init__(env)
        world_size = self.unwrapped.size

        # O limite máximo dos seus dados é 4 (status + 1), então high=5 é suficiente
        # dtype travado em np.uint8 (1 byte por célula)
        self.observation_space = spaces.Box(
            low=0, high=5,
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
        Converte o dicionário de observação em um tensor ultraleve na memória RAM.
        """
        # Tensor inicializado explicitamente como np.uint8
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
                    tensor[3, x, y] = 1  # Usando 1 inteiro no lugar de 1.0 float

        return tensor


class CaseByCaseWrapper(gym.Wrapper):
    """
    Transforma o problema de "N decisões por passo" para "1 decisão por passo, N vezes".
    """

    def __init__(self, env: DengueWrapper):
        super().__init__(env)

        self.action_space = spaces.Discrete(6)

        self.observation_space = spaces.Dict({
            "map": env.observation_space,
            "case_coords": spaces.Box(
                # Normalizado de 0.0 a 1.0 em float32 (Apenas 8 bytes, não pesa nada)
                low=0.0, high=1.0, shape=(2,), dtype=np.float32
            )
        })

        self.active_cases = []
        self.case_iterator = iter(self.active_cases)
        self.pending_actions = []
        self.current_case = (0, 0, 0)
        self._current_map_obs = None

    def _get_active_cases(self):
        df = self.unwrapped.obs_cases
        t = self.unwrapped.t
        active_df = df[df.t == t]
        cases = []
        for case in active_df.itertuples():
            cases.append((case.Index, int(case.x), int(case.y)))
        return cases

    def _refresh_active_cases(self):
        self.active_cases = self._get_active_cases()
        self.case_iterator = iter(self.active_cases)

    def _advance_empty_days(self):
        """Skip days with no reported cases (common at start_day=1 or sparse curves)."""
        terminated, truncated = False, False
        info = {}

        while not self.active_cases:
            obs_tensor, reward, terminated, truncated, info = self.env.step(tuple())
            self._current_map_obs = obs_tensor
            if terminated or truncated:
                self.current_case = (0, 0, 0)
                return terminated, truncated, info
            self._refresh_active_cases()

        return terminated, truncated, info

    def _make_obs(self):
        # Coordenadas normalizadas dividindo pelo tamanho original
        normalized_x = self.current_case[1] / self.unwrapped.size
        normalized_y = self.current_case[2] / self.unwrapped.size

        return {
            "map": self._current_map_obs,
            "case_coords": np.array([normalized_x, normalized_y], dtype=np.float32)
        }

    def _next_case(self):
        accumulated_reward = 0.0
        last_info = {}

        while True:
            try:
                self.current_case = next(self.case_iterator)
                return self._make_obs(), accumulated_reward, False, False, last_info

            except StopIteration:
                action_tuple = tuple(self.pending_actions)
                self.pending_actions = []

                obs_tensor, reward, terminated, truncated, info = self.env.step(action_tuple)
                accumulated_reward += reward
                self._current_map_obs = obs_tensor
                last_info = info

                if terminated or truncated:
                    self.current_case = (0, 0, 0)
                    return self._make_obs(), accumulated_reward, terminated, truncated, info

                self._refresh_active_cases()
                term, trunc, skip_info = self._advance_empty_days()
                if skip_info:
                    last_info = skip_info
                if term or trunc:
                    self.current_case = (0, 0, 0)
                    return self._make_obs(), accumulated_reward, term, trunc, last_info

    def reset(self, **kwargs):
        obs_tensor, info = self.env.reset(**kwargs)
        self._current_map_obs = obs_tensor
        self.pending_actions = []

        self._refresh_active_cases()
        self._advance_empty_days()

        try:
            self.current_case = next(self.case_iterator)
        except StopIteration:
            self.current_case = (0, 0, 0)

        return self._make_obs(), info

    def step(self, action: int):
        if self.current_case[0] == 0 and not self.active_cases:
            return self._make_obs(), 0.0, True, False, {}
        self.pending_actions.append((int(self.current_case[0]), int(action)))
        return self._next_case()