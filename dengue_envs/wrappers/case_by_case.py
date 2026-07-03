"""Wrapper que transforma "N decisões por dia" em "1 decisão por passo".

Versão canônica (Fase 1) do `CaseByCaseWrapper`, idêntica em comportamento à
que vinha sendo usada no treino em `agents/deepq/files/dengue_wrapper.py`.
Aquela cópia legada permanece intacta.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from dengue_envs.wrappers.map_tensor import DengueWrapper


class CaseByCaseWrapper(gym.Wrapper):
    """Expõe uma ação discreta por caso, avançando o dia quando os casos acabam."""

    def __init__(self, env: DengueWrapper):
        super().__init__(env)

        self.action_space = spaces.Discrete(6)

        self.observation_space = spaces.Dict({
            "map": env.observation_space,
            "case_coords": spaces.Box(
                low=0.0, high=1.0, shape=(2,), dtype=np.float32
            ),
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
        """Pula dias sem casos reportados (comum em start_day=1 ou curvas esparsas)."""
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
        normalized_x = self.current_case[1] / self.unwrapped.size
        normalized_y = self.current_case[2] / self.unwrapped.size
        return {
            "map": self._current_map_obs,
            "case_coords": np.array([normalized_x, normalized_y], dtype=np.float32),
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
