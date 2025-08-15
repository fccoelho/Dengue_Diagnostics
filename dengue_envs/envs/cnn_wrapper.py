import gymnasium as gym
import numpy as np
import pandas as pd


def process_observation_to_tensor(
    obs_dict: dict,
    world_size: int,
    current_t: int,
    all_cases_df: pd.DataFrame,
    case_coords_map: dict
) -> np.ndarray:
    """
    Função pura que converte um dicionário de observações em um tensor 4D.

    Args:
        obs_dict (dict): Dicionário com os dados de observação do passo atual.
        world_size (int): A dimensão (altura e largura) do mapa.
        current_t (int): O timestep atual.
        all_cases_df (pd.DataFrame): DataFrame contendo todos os casos da simulação.
        case_coords_map (dict): Um mapa de {case_id: (x, y)} para busca rápida de coordenadas.

    Returns:
        np.ndarray: O tensor 4D resultante.
    """
    tensor = np.zeros((4, world_size, world_size), dtype=np.float32)

    # Channel 0: Clinical diagnostics
    for case in obs_dict.get('clinical_diagnostic', []):
        x, y, diag = case
        if 0 <= x < world_size and 0 <= y < world_size:
            tensor[0, y, x] = diag + 1  # Usando (y, x) como padrão para imagens

    # Channel 1: TestD status
    for case_id, status in obs_dict.get('testd', []):
        x, y = case_coords_map.get(case_id, (-1, -1)) # Busca no mapa ao invés de chamar método
        if 0 <= x < world_size and 0 <= y < world_size:
            tensor[1, y, x] = status + 1

    # Channel 2: TestC status
    for case_id, status in obs_dict.get('testc', []):
        x, y = case_coords_map.get(case_id, (-1, -1)) # Busca no mapa
        if 0 <= x < world_size and 0 <= y < world_size:
            tensor[2, y, x] = status + 1

    # Channel 3: Active case mask
    # Itera sobre o DataFrame fornecido
    active_cases = all_cases_df[all_cases_df['t'] == current_t]
    for _, case_data in active_cases.iterrows():
        x, y = int(case_data.x), int(case_data.y)
        if 0 <= x < world_size and 0 <= y < world_size:
            tensor[3, y, x] = 1.0

    return tensor


class DengueWrapper(gym.ObservationWrapper):
    """
    Converte a observação do dicionário para um tensor 4D, delegando a lógica
    de processamento para uma função externa testável.
    """
    def __init__(self, env):
        super().__init__(env)
        world_size = self.unwrapped.size
        self.observation_space = gym.spaces.Box(
            low=0, high=4,
            shape=(4, world_size, world_size),
            dtype=np.float32
        )
        self._world_size = world_size

    def observation(self, obs_dict: dict) -> np.ndarray:
        """
        Coleta os dados do ambiente e chama a função de processamento.
        """
        current_t = self.unwrapped.t
        all_cases_df = self.unwrapped.obs_cases


        case_coords_map = {
            case_id: (row.x, row.y)
            for case_id, row in all_cases_df.iterrows()
        }

        return process_observation_to_tensor(
            obs_dict=obs_dict,
            world_size=self._world_size,
            current_t=current_t,
            all_cases_df=all_cases_df,
            case_coords_map=case_coords_map
        )

