import numpy as np
import pandas as pd
import random
from typing import List, Dict, Any, Tuple
from tianshou.data import Collector


def generate_actions_from_q_maps(
        q_maps_batch: np.ndarray,
        envs_data: List[Dict[str, Any]],
        epsilon: float
) -> List[Tuple]:
    """
    Função pura que gera ações para um batch de ambientes a partir de seus mapas de Q-valores.

    Args:
        q_maps_batch (np.ndarray): Batch de mapas de Q-valores da rede. Shape: (num_envs, 6, H, W).
        envs_data (List[Dict]): Lista de dicionários, um para cada ambiente, contendo
                               os dados necessários (ex: {'current_t': t, 'obs_cases': df}).
        epsilon (float): A probabilidade de exploração (epsilon-greedy).

    Returns:
        List[Tuple]: Uma lista de tuplas de ações, uma para cada ambiente.
    """
    processed_actions = []

    for i, q_map in enumerate(q_maps_batch):
        env_data = envs_data[i]
        current_t = env_data['current_t']
        obs_cases_df = env_data['obs_cases']

        # Pega os casos que estão ativos NESTE timestep específico
        df_current_t_cases = obs_cases_df[obs_cases_df.t == current_t]

        actions_for_this_env = []

        for case_id, case_data in df_current_t_cases.iterrows():
            y, x = int(case_data.y), int(case_data.x)

            action_idx = 0
            if random.random() > epsilon:
                # Exploitation
                q_values_for_case = q_map[:, y, x]
                action_idx = np.argmax(q_values_for_case)
            else:
                # Exploration
                action_idx = random.randrange(6)

            actions_for_this_env.append((case_id, int(action_idx)))

        processed_actions.append(tuple(actions_for_this_env))

    return processed_actions


class FCNCollector(Collector):
    """
    Coletor customizado que delega a lógica de geração de ações para uma
    função externa testável.
    """

    def __init__(self, policy, env, buffer, eps=0.1, **kwargs):
        super().__init__(policy, env, buffer, **kwargs)
        self.eps = eps

    def _get_actions_for_envs(self, result: 'Batch') -> List[Tuple]:
        """
        Coleta os dados do ambiente e chama a função de processamento.
        """
        q_maps_batch = result.out.cpu().numpy()

        envs_data = [
            {
                'current_t': worker.env.unwrapped.t,
                'obs_cases': worker.env.unwrapped.obs_cases
            }
            for worker in self.env.workers
        ]

        return generate_actions_from_q_maps(
            q_maps_batch=q_maps_batch,
            envs_data=envs_data,
            epsilon=self.eps
        )

    def step_env(self, result: 'Batch'):
        """
        Sobrescreve o método original para usar nossa lógica de tradução de ação.
        """
        actions = self._get_actions_for_envs(result)
        obs_next, rew, terminated, truncated, info = self.env.step(actions)

        # Atualiza os dados do Tianshou
        self.data.done = np.logical_or(terminated, truncated)
        result.obs_next = obs_next
        result.rew = rew
        result.done = self.data.done
        result.info = info

        return result