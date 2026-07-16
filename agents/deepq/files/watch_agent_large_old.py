import torch
import numpy as np
import os
import time

from tianshou.policy import DQNPolicy
from tianshou.data import Batch

# Importações do seu repositório
from dengue_envs.envs.dengue_diagnostics import DengueDiagnosticsEnv
from dengue_wrapper import DengueWrapper, CaseByCaseWrapper
from fcn_network import DengueNet

# --- CONFIGURAÇÕES DO AMBIENTE (Devem ser idênticas ao treino) ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WORLD_SIZE = 600
EPISIZE = 250
REWARD_DELAY_DAYS = 5
MIN_BORDER_DISTANCE = 80
MAX_RADIUS = 200
MIN_RADIUS = 80

# Caminho para os pesos treinados (AJUSTE AQUI PARA A SUA SEED VENCEDORA)
MODEL_PATH = os.path.join("logs", "experiment_large_grid", "dqn_large_grid_delay_seed_400", "policy_large_best.pth")


def make_test_env():
    """Cria o ambiente com o render_mode='human' para assistirmos ao agente."""
    dengue_center = (np.random.randint(MIN_BORDER_DISTANCE, WORLD_SIZE - MIN_BORDER_DISTANCE),
                     np.random.randint(MIN_BORDER_DISTANCE, WORLD_SIZE - MIN_BORDER_DISTANCE))
    chik_center = (np.random.randint(MIN_BORDER_DISTANCE, WORLD_SIZE - MIN_BORDER_DISTANCE),
                   np.random.randint(MIN_BORDER_DISTANCE, WORLD_SIZE - MIN_BORDER_DISTANCE))

    env = DengueDiagnosticsEnv(
        size=WORLD_SIZE,
        episize=EPISIZE,
        epilength=60,
        reward_delay_days=REWARD_DELAY_DAYS,
        clinical_specificity=(0.5, 0.95),
        dengue_center=dengue_center,
        chik_center=chik_center,
        dengue_radius=np.random.randint(MIN_RADIUS, MAX_RADIUS),
        chik_radius=np.random.randint(MIN_RADIUS, MAX_RADIUS),
        render_mode="human"  # LIGANDO A INTERFACE GRÁFICA
    )
    env = DengueWrapper(env)
    env = CaseByCaseWrapper(env)
    return env


def watch_agent():
    print(f"[*] Carregando ambiente e modelo: {MODEL_PATH}")
    env = make_test_env()

    # Coleta shapes
    map_shape = env.observation_space.spaces["map"].shape
    action_shape = env.action_space.n

    # Instancia a rede
    net = DengueNet(map_shape, action_shape, device=DEVICE).to(DEVICE)
    optim = torch.optim.Adam(net.parameters(), lr=1e-4)

    # CORREÇÃO AQUI: Passamos os hiperparâmetros idênticos ao do treino
    # para forçar o Tianshou a recriar a arquitetura exata (incluindo o model_old)
    policy = DQNPolicy(
        model=net,
        optim=optim,
        discount_factor=0.99,
        estimation_step=4,
        target_update_freq=1500,  # <-- Isso resolve o erro do model_old
        action_space=env.action_space
    )

    # Carrega os pesos e usa strict=False para evitar qualquer outro aviso menor
    if os.path.exists(MODEL_PATH):
        policy.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE), strict=False)
        print("[+] Pesos carregados com sucesso!")
    else:
        raise FileNotFoundError(f"Arquivo não encontrado: {MODEL_PATH}")

    # Modo de Avaliação (Desliga parâmetros de treino e zera Epsilon)
    policy.eval()
    policy.set_eps(0.0)

    obs, info = env.reset()
    done = False

    print("\nIniciando simulação...")
    while not done:
        # Prepara a observação para o Tianshou (ele espera um Batch com a chave 'obs')
        obs_batch = Batch(obs=[obs], info=[info])

        # Pede a ação para a política
        with torch.no_grad():
            result = policy(obs_batch)
            action = result.act[0].item()

        # Executa no ambiente
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Descomente a linha abaixo caso os frames estejam passando rápido demais na interface
        # time.sleep(0.05)

    print("\nSimulação Finalizada! Gerando métricas...")

    base_env = env.unwrapped

    metrics = base_env.get_episode_metrics()
    print("\n" + "=" * 40)
    print("   RESULTADOS FINAIS DO EPISÓDIO")
    print("=" * 40)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")

    os.makedirs("resultados_teste", exist_ok=True)
    map_path = os.path.join("resultados_teste", "mapa_confusao_large.png")
    base_env.plot_confusion_map(title="Agente DQN - Grid Expandido", save_path=map_path)
    print(f"\n[*] Mapa de confusão salvo em: {map_path}")

    env.close()


if __name__ == "__main__":
    watch_agent()