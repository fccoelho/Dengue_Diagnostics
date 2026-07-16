import torch
import numpy as np
import matplotlib.pyplot as plt
from tianshou.data import Batch
from tianshou.policy import DQNPolicy

# Importe as suas classes
from dengue_envs.envs.dengue_diagnostics import DengueDiagnosticsEnv
from dengue_wrapper import DengueWrapper, CaseByCaseWrapper
from fcn_network import DengueNet

# --- CONFIGURAÇÕES ---
# Coloque o nome exato do arquivo da sua melhor política aqui
POLICY_PATH = "dqn_dengue_policy6.pth"
SAVE_IMAGE = "mapa_tcc_resultado_50.png"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Configuração do Mundo para o Teste (Pode fixar para ser reprodutível)
WORLD_SIZE = 400
DENGUE_CENTER = (100, 100)
CHIK_CENTER = (300, 300)


def make_eval_env():
    """Cria o ambiente para avaliação (sem renderização humana para ser rápido)"""
    env = DengueDiagnosticsEnv(
        epilength=60,
        size=WORLD_SIZE,
        clinical_specificity=0.5,  # Fixar em 0.8 para o teste padrão
        dengue_center=DENGUE_CENTER,
        chik_center=CHIK_CENTER,
        # Render mode None pois usaremos matplotlib
        render_mode=None
    )
    env = DengueWrapper(env)
    env = CaseByCaseWrapper(env)
    return env


def run_evaluation():
    print(f"Carregando política de: {POLICY_PATH}")

    env = make_eval_env()

    # Configura a Rede e Política
    map_shape = env.observation_space.spaces["map"].shape
    action_shape = env.action_space.n
    net = DengueNet(map_shape, action_shape, device=DEVICE).to(DEVICE)

    policy = DQNPolicy(
        model=net,
        optim=None,
        discount_factor=0.99,
        action_space=env.action_space,
        estimation_step=3,
        target_update_freq=500
    )

    # Carrega os pesos
    try:
        policy.load_state_dict(torch.load(POLICY_PATH, map_location=DEVICE))
    except FileNotFoundError:
        print("Erro: Arquivo de política não encontrado.")
        return

    policy.eval()
    policy.set_eps(0.0)  # Sem aleatoriedade

    print("Executando episódio de diagnóstico...")
    obs, info = env.reset()
    terminated = False

    # Loop do episódio
    while not terminated:
        batch = Batch(obs=[obs], info=[info])
        result = policy(batch)
        action = result.act[0].item()

        obs, reward, terminated, truncated, info = env.step(action)

        if truncated: break

    print("Episódio finalizado. Gerando gráfico...")

    # Acessa o ambiente original (unwrapped) para chamar a função de plotagem
    env.unwrapped.plot_confusion_map(
        title="Resultado do Diagnóstico do Agente (Cenário de Teste)",
        save_path=SAVE_IMAGE
    )

    print("Concluído!")
    # Mostra a imagem na tela (se estiver num ambiente gráfico)
    plt.show()


if __name__ == "__main__":
    run_evaluation()