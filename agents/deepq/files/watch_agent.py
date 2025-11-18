import torch
import pygame
import time
from tianshou.data import Batch
from tianshou.policy import DQNPolicy

# Importações do seu projeto
from dengue_envs.envs.dengue_diagnostics import DengueDiagnosticsEnv
from dengue_wrapper import DengueWrapper, CaseByCaseWrapper
from fcn_network import DengueNet

# --- Configurações ---
POLICY_PATH = "dqn_dengue_policy2.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RENDER_FPS = 10  # Quantos "passos" do agente por segundo

def make_env():
    """
    Cria a pilha de ambientes, exatamente como no treinamento,
    mas com render_mode="human".
    """
    # 1. Cria o ambiente base com renderização
    env = DengueDiagnosticsEnv(
        epilength=60,
        size=400,
        clinical_specificity=0.5,
        render_mode="human"  # Habilita a janela do pygame
    )
    # 2. Aplica os mesmos wrappers
    env = DengueWrapper(env)
    env = CaseByCaseWrapper(env)
    return env


def load_policy(env):
    """
    Carrega a política DQN treinada.
    """
    # 1. Recria a arquitetura da rede
    map_shape = env.observation_space.spaces["map"].shape
    action_shape = env.action_space.n
    net = DengueNet(map_shape, action_shape, device=DEVICE).to(DEVICE)

    # 2. Recria a política COM a mesma estrutura do treinamento
    # (Puxe esses valores do seu train_dqn.py)
    N_STEP = 3
    TARGET_UPDATE_FREQ = 500

    policy = DQNPolicy(
        model=net,
        optim=None,
        discount_factor=0.99,
        action_space=env.action_space,
        estimation_step=N_STEP,  # <-- ADICIONE ESTA LINHA
        target_update_freq=TARGET_UPDATE_FREQ  # <-- ADICIONE ESTA LINHA
    )

    # 3. Carrega os pesos salvos
    try:
        # Corrigindo o FutureWarning
        policy.load_state_dict(torch.load(POLICY_PATH, map_location=DEVICE, weights_only=True))
    except FileNotFoundError:
        print(f"Erro: Arquivo de política não encontrado em '{POLICY_PATH}'")
        print("Certifique-se de que o treinamento foi concluído e o arquivo foi salvo.")
        exit(1)

    # 4. Define a política para o modo de avaliação (determinístico)
    policy.eval()
    policy.set_eps(0.0)  # Sem ações aleatórias (epsilon-greedy = 0)

    print(f"Política '{POLICY_PATH}' carregada com sucesso.")
    return policy


if __name__ == "__main__":
    # 1. Inicializa o ambiente e a política
    env = make_env()
    policy = load_policy(env)

    # 2. Prepara para o loop de visualização
    obs, info = env.reset()
    terminated, truncated = False, False
    total_reward = 0

    # O clock do Pygame é necessário para controlar o FPS
    # O env.render() no DengueDiagnosticsEnv usa pygame
    clock = pygame.time.Clock()

    print("Iniciando visualização do agente...")

    # 3. Loop do episódio
    while not terminated and not truncated:
        try:
            # Lida com eventos da janela (ex: fechar no "X")
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    print("Janela fechada pelo usuário.")
                    terminated = True  # Encerra o loop
                    break

            if terminated:
                break

            # 1. Cria um "Batch" do Tianshou para a política
            #    (A rede foi programada para lidar com um único 'obs' dict)
            batch = Batch(obs=[obs], info=[info])

            # 2. Pede à política para decidir a ação
            result = policy(batch)
            action = result.act[0].item()  # Pega a ação (como int)

            # print("______________________________")
            # print()
            # print(action)
            # print()
            # print("______________________________")
            # 3. Executa a ação no ambiente
            obs, reward, terminated, truncated, info = env.step(action)

            total_reward += reward

            # 4. Renderiza o estado atual
            env.render()

            # 5. Controla a velocidade de visualização
            clock.tick(RENDER_FPS)

        except pygame.error as e:
            # Isso pode acontecer se a janela for fechada abruptamente
            print(f"Erro de Pygame (janela pode ter sido fechada): {e}")
            break
        except KeyboardInterrupt:
            print("\nVisualização interrompida pelo usuário (Ctrl+C).")
            terminated = True

    print("--- Fim da Visualização ---")
    print(f"Recompensa total do episódio: {total_reward:.2f}")

    # 5. Limpeza
    env.close()
    pygame.quit()