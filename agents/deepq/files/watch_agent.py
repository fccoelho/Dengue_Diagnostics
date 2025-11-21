import torch
import pygame
import time
import numpy as np
from tianshou.data import Batch
from tianshou.policy import DQNPolicy
from typing import Tuple
from dengue_envs.envs.dengue_diagnostics import DengueDiagnosticsEnv
from dengue_wrapper import DengueWrapper, CaseByCaseWrapper
from fcn_network import DengueNet


# --- Configurações ---
POLICY_PATH = "dqn_dengue_policy6.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RENDER_FPS = 10
WORLD_SIZE = 400
MIN_BORDER_DISTANCE = 50
MAX_RADIUS = 100
MIN_RADIUS = 50

def generate_random_center(size: int, margin: int) -> Tuple[int, int]:
    """Gera um par de coordenadas aleatórias dentro dos limites do mapa."""
    x = np.random.randint(margin, size - margin)
    y = np.random.randint(margin, size - margin)
    return int(x), int(y)

def make_env():
    """
    Cria a pilha de ambientes, exatamente como no treinamento,
    mas com render_mode="human".
    """

    dengue_center = generate_random_center(WORLD_SIZE, MIN_BORDER_DISTANCE)
    chik_center = generate_random_center(WORLD_SIZE, MIN_BORDER_DISTANCE)
    dengue_radius = np.random.randint(MIN_RADIUS, MAX_RADIUS)
    chik_radius = np.random.randint(MIN_RADIUS, MAX_RADIUS)

    env = DengueDiagnosticsEnv(
        epilength=60,
        size=WORLD_SIZE,
        clinical_specificity=(0.5, 0.95),
        dengue_center=dengue_center,
        chik_center=chik_center,
        dengue_radius=dengue_radius,
        chik_radius=chik_radius,
        render_mode="human"
    )

    env = DengueWrapper(env)
    env = CaseByCaseWrapper(env)
    return env


def load_policy(env):
    """
    Carrega a política DQN treinada.
    """
    map_shape = env.observation_space.spaces["map"].shape
    action_shape = env.action_space.n
    net = DengueNet(map_shape, action_shape, device=DEVICE).to(DEVICE)

    N_STEP = 3
    TARGET_UPDATE_FREQ = 500

    policy = DQNPolicy(
        model=net,
        optim=None,
        discount_factor=0.99,
        action_space=env.action_space,
        estimation_step=N_STEP,
        target_update_freq=TARGET_UPDATE_FREQ
    )

    try:
        policy.load_state_dict(torch.load(POLICY_PATH, map_location=DEVICE, weights_only=True))
    except FileNotFoundError:
        print(f"Erro: Arquivo de política não encontrado em '{POLICY_PATH}'")
        print("Certifique-se de que o treinamento foi concluído e o arquivo foi salvo.")
        exit(1)

    policy.eval()
    policy.set_eps(0.0)

    print(f"Política '{POLICY_PATH}' carregada com sucesso.")
    return policy


if __name__ == "__main__":
    env = make_env()
    policy = load_policy(env)

    obs, info = env.reset()
    terminated, truncated = False, False
    total_reward = 0

    # 1. Inicialize o contador de ações
    actions_taken_count = 0

    clock = pygame.time.Clock()

    print("Iniciando visualização do agente...")

    while not terminated and not truncated:
        try:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    print("Janela fechada pelo usuário.")
                    terminated = True
                    break

            if terminated:
                break

            batch = Batch(obs=[obs], info=[info])

            result = policy(batch)
            action = result.act[0].item()

            # --- EXTRAÇÃO DE INFORMAÇÕES ---
            # Incrementa contador
            actions_taken_count += 1

            # Pega o ID do caso atual direto do wrapper
            # O wrapper armazena (case_id, x, y) em current_case
            current_case_id = env.current_case[0]

            # Pega o total de casos do ambiente base (unwrapped)
            total_cases_env = len(env.unwrapped.obs_cases)

            # Print formatado para acompanhar
            # print(f"Passo: {actions_taken_count} | "
            #       f"Ação: {action} | "
            #       f"Case ID: {current_case_id} | "
            #       f"Total Casos no Mundo: {total_cases_env}")
            # -------------------------------

            obs, reward, terminated, truncated, info = env.step(action)

            total_reward += reward

            print(total_reward)

            env.render()

            clock.tick(RENDER_FPS)

        except pygame.error as e:
            print(f"Erro de Pygame (janela pode ter sido fechada): {e}")
            break
        except KeyboardInterrupt:
            print("\nVisualização interrompida pelo usuário (Ctrl+C).")
            terminated = True

    print("--- Fim da Visualização ---")
    print(f"Recompensa total do episódio: {total_reward:.2f}")
    print(f"Total de ações tomadas: {actions_taken_count}")  # Print final

    env.close()
    pygame.quit()