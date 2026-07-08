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
from tianshou.env import DummyVectorEnv

# Importações do seu repositório original
from dengue_envs.envs.dengue_diagnostics import DengueDiagnosticsEnv
from dengue_wrapper import DengueWrapper, CaseByCaseWrapper
from fcn_network import DengueNet

# --- CONFIGURAÇÕES DO NOVO EXPERIMENTO EXPANDIDO ---
print(f"CUDA Available: {torch.cuda.is_available()}")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Hiperparâmetros Ajustados para Estabilidade em Espaços Maiores
LR = 5e-5               # Taxa de aprendizado ligeiramente menor para evitar divergência
GAMMA = 0.99            # Mantido para propagar bem o credit assignment do delay
N_STEP = 4              # Aumentado para ajudar a capturar a dependência temporal do delay
TARGET_UPDATE_FREQ = 1500
BUFFER_SIZE = 1000     # NOVO: Buffer expandido para comportar mais estados
BATCH_SIZE = 32        # NOVO: Batch size maior para lidar com a alta variância do grid expandido

# Configuração de Épocas
EPOCH = 60              # Aumentado ligeiramente para dar mais tempo de convergência
STEP_PER_EPOCH = 12000
STEP_PER_COLLECT = 1200
UPDATE_PER_STEP = 0.1

# Exploração Reajustada
EPS_TRAIN_START = 1.0
EPS_TRAIN_FINAL = 0.05
EPS_TRAIN_DECAY = 15000
EPS_TEST = 0.01

# Ambientes Paralelos
NUM_ENVS = 2
NUM_TEST_ENVS = 2

# Novas dimensões do Mundo Epidemiológico
WORLD_SIZE = 300          # NOVO: Grid expandido (era 400)
EPISIZE = 100             # NOVO: Mais casos ocorrendo simultaneamente (era 150)
REWARD_DELAY_DAYS = 0     # NOVO: Ativação do delay epidemiológico de 0 dias (sem delay)
MIN_BORDER_DISTANCE = 80
MAX_RADIUS = 100          # NOVO: Raios máximos adaptados ao novo tamanho de cidade
MIN_RADIUS = 80

# Seeds para o novo experimento
SEEDS = [42]


def generate_random_center(size: int, margin: int) -> Tuple[int, int]:
    x = np.random.randint(margin, size - margin)
    y = np.random.randint(margin, size - margin)
    return int(x), int(y)


def make_env():
    """Factory do ambiente customizado para o Grid Maior."""
    dengue_center = generate_random_center(WORLD_SIZE, MIN_BORDER_DISTANCE)
    chik_center = generate_random_center(WORLD_SIZE, MIN_BORDER_DISTANCE)
    dengue_radius = np.random.randint(MIN_RADIUS, MAX_RADIUS)
    chik_radius = np.random.randint(MIN_RADIUS, MAX_RADIUS)

    # Inicializando o ambiente com as novas flags de escala e delay
    env = DengueDiagnosticsEnv(
        size=WORLD_SIZE,
        episize=EPISIZE,
        epilength=60,
        reward_delay_days=REWARD_DELAY_DAYS,  # Injetando o delay aqui
        clinical_specificity=(0.5, 0.95),
        dengue_center=dengue_center,
        chik_center=chik_center,
        dengue_radius=dengue_radius,
        chik_radius=chik_radius
    )
    env = DengueWrapper(env)
    env = CaseByCaseWrapper(env)
    return env


def train_large_grid_agent(seed):
    """Executa o pipeline completo de treino para uma seed específica."""
    experiment_name = f"dqn_large_grid_delay_seed_{seed}"
    print(f"\n{'=' * 50}")
    print(f" INICIANDO EXPERIMENTO EXPANDIDO: {experiment_name}")
    print(f"{'=' * 50}\n")

    np.random.seed(seed)
    torch.manual_seed(seed)

    # Criando os vetores de sub-processos para os ambientes
    train_envs = DummyVectorEnv([make_env for _ in range(NUM_ENVS)])
    test_envs = DummyVectorEnv([make_env for _ in range(NUM_TEST_ENVS)])

    train_envs.seed(seed)
    test_envs.seed(seed)

    # Coleta dinâmica de dimensões através do wrapper instanciado
    dummy_env = make_env()
    map_shape = dummy_env.observation_space.spaces["map"].shape
    action_shape = dummy_env.action_space.n

    # Instanciando a rede neural (ela se ajusta ao novo tamanho de mapa automaticamente)
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

    # Configuração da Memória de Replay
    buffer = VectorReplayBuffer(
        total_size=BUFFER_SIZE,
        buffer_num=NUM_ENVS,
        ignore_obs_next=True
    )

    train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)
    test_collector = Collector(policy, test_envs)

    print("-> Pré-populando o buffer de experiências...")
    train_collector.collect(n_step=1000, reset_before_collect=True)

    # Logs direcionados para uma pasta específica do experimento em escala
    log_path = os.path.join("logs", "experiment_large_grid", experiment_name)
    writer = SummaryWriter(log_path)
    logger = TensorboardLogger(writer)

    def train_fn(epoch, env_step):
        # Lógica de decaimento linear ajustada para a nova escala temporal
        if env_step <= EPS_TRAIN_DECAY:
            eps = EPS_TRAIN_START - env_step / EPS_TRAIN_DECAY * \
                  (EPS_TRAIN_START - EPS_TRAIN_FINAL)
        else:
            eps = EPS_TRAIN_FINAL
        policy.set_eps(eps)

    def test_fn(epoch, env_step):
        policy.set_eps(EPS_TEST)

    def save_best_fn(policy):
        path = os.path.join(log_path, "policy_large_best.pth")
        torch.save(policy.state_dict(), path)
        print(f"[*] Novo melhor modelo salvo em: {path}")

    # Inicialização do Treinador Off-Policy do Tianshou
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

    # Salvando os pesos finais obtidos
    final_path = os.path.join(log_path, "policy_large_final.pth")
    torch.save(policy.state_dict(), final_path)
    print(f"-> Treino concluído com sucesso. Resultados salvos em {log_path}")

    train_envs.close()
    test_envs.close()


if __name__ == "__main__":
    for seed in SEEDS:
        try:
            train_large_grid_agent(seed)
        except Exception as e:
            print(f"🚨 ERRO CRÍTICO na execução da Seed {seed}: {e}")
            continue

# TODO: implementar com matrizes esparsas