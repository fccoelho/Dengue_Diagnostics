import torch
import os
import numpy as np
from typing import Tuple
from torch.utils.tensorboard import SummaryWriter

from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import SubprocVectorEnv
from tianshou.policy import DQNPolicy
from tianshou.trainer import OffpolicyTrainer
from tianshou.utils import TensorboardLogger

# Seus módulos
from dengue_envs.envs.dengue_diagnostics import DengueDiagnosticsEnv
from dengue_wrapper import DengueWrapper, CaseByCaseWrapper
from fcn_network import DengueNet

# --- CONFIGURAÇÕES GERAIS ---
print(f"CUDA Available: {torch.cuda.is_available()}")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Hiperparâmetros Fixos
LR = 1e-4
GAMMA = 0.99
N_STEP = 3
TARGET_UPDATE_FREQ = 1000
BUFFER_SIZE = 1000
BATCH_SIZE = 64

# Configuração do Experimento
EPOCH = 50  # 50 Épocas por agente
STEP_PER_EPOCH = 10000  # 10k passos por época
STEP_PER_COLLECT = 1000
UPDATE_PER_STEP = 0.1

# Exploração
EPS_TRAIN_START = 1.0
EPS_TRAIN_FINAL = 0.05
EPS_TRAIN_DECAY = 50000  # Decaimento ao longo de 20 épocas
EPS_TEST = 0.01

# Ambientes
NUM_ENVS = 4
NUM_TEST_ENVS = 4

# Mundo
WORLD_SIZE = 400
MIN_BORDER_DISTANCE = 50
MAX_RADIUS = 100
MIN_RADIUS = 50

# --- LISTA DE EXPERIMENTOS ---
# Vamos treinar 3 agentes diferentes
SEEDS = [100]


def generate_random_center(size: int, margin: int) -> Tuple[int, int]:
    x = np.random.randint(margin, size - margin)
    y = np.random.randint(margin, size - margin)
    return int(x), int(y)


def make_env():
    """Factory do ambiente."""
    # Nota: A seed global do numpy/torch cuida da aleatoriedade aqui
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
        chik_radius=chik_radius
    )
    env = DengueWrapper(env)
    env = CaseByCaseWrapper(env)
    return env


def train_one_agent(seed):
    """Função que treina UM agente completo com uma seed específica."""

    experiment_name = f"dqn_seed_{seed}"
    print(f"\n{'=' * 40}")
    print(f"   INICIANDO TREINAMENTO: {experiment_name}")
    print(f"{'=' * 40}\n")

    # 1. Definir Seeds para reprodutibilidade deste agente
    np.random.seed(seed)
    torch.manual_seed(seed)

    # 2. Criar Ambientes
    train_envs = SubprocVectorEnv([make_env for _ in range(NUM_ENVS)])
    test_envs = SubprocVectorEnv([make_env for _ in range(NUM_TEST_ENVS)])

    # Seed nos ambientes
    train_envs.seed(seed)
    test_envs.seed(seed)

    # 3. Rede e Política
    # Criamos uma instância dummy para pegar shapes
    dummy_env = make_env()
    map_shape = dummy_env.observation_space.spaces["map"].shape
    action_shape = dummy_env.action_space.n

    net = DengueNet(map_shape, action_shape, device=DEVICE).to(DEVICE)
    optim = torch.optim.Adam(net.parameters(), lr=LR)

    policy = DQNPolicy(
        model=net,
        optim=optim,
        discount_factor=GAMMA,
        estimation_step=N_STEP,
        target_update_freq=TARGET_UPDATE_FREQ,
        action_space=dummy_env.action_space
    )

    # 4. Buffer
    buffer = VectorReplayBuffer(
        total_size=BUFFER_SIZE,
        buffer_num=NUM_ENVS,
        ignore_obs_next=True
    )

    # 5. Coletores
    train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)
    test_collector = Collector(policy, test_envs)

    # Inicialização forçada do buffer
    print("-> Inicializando buffer...")
    train_collector.collect(n_step=100, reset_before_collect=True)

    # 6. Logger (Pastas separadas por seed!)
    log_path = os.path.join("logs", "experiment_50_epochs", experiment_name)
    writer = SummaryWriter(log_path)
    logger = TensorboardLogger(writer)

    def train_fn(epoch, env_step):
        if env_step <= EPS_TRAIN_DECAY:
            eps = EPS_TRAIN_START - env_step / EPS_TRAIN_DECAY * \
                  (EPS_TRAIN_START - EPS_TRAIN_FINAL)
        else:
            eps = EPS_TRAIN_FINAL
        policy.set_eps(eps)

    def test_fn(epoch, env_step):
        policy.set_eps(EPS_TEST)

    def save_best_fn(policy):
        # Salva na pasta do experimento específico
        path = os.path.join(log_path, "policy_best.pth")
        torch.save(policy.state_dict(), path)

    # 7. Treinamento
    trainer = OffpolicyTrainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=EPOCH,
        step_per_epoch=STEP_PER_EPOCH,
        step_per_collect=STEP_PER_COLLECT,
        update_per_step=UPDATE_PER_STEP,
        episode_per_test=NUM_TEST_ENVS,
        batch_size=BATCH_SIZE,
        train_fn=train_fn,
        test_fn=test_fn,
        stop_fn=None,
        save_best_fn=save_best_fn,
        logger=logger
    )

    result = trainer.run()

    # Salvar modelo final
    final_path = os.path.join(log_path, "policy_final.pth")
    torch.save(policy.state_dict(), final_path)
    print(f"-> Treino finalizado para Seed {seed}. Salvo em {final_path}")

    # Fechar ambientes para liberar memória para o próximo agente
    train_envs.close()
    test_envs.close()


if __name__ == "__main__":
    for seed in SEEDS:
        try:
            train_one_agent(seed)
        except Exception as e:
            print(f"ERRO CRÍTICO na Seed {seed}: {e}")
            continue