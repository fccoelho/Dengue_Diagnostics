import numpy as np
import pandas as pd
import torch
import os
from tianshou.data import Batch
from tianshou.policy import DQNPolicy
from dengue_envs.envs.dengue_diagnostics import DengueDiagnosticsEnv
from agents.deepq.files.dengue_wrapper import DengueWrapper, CaseByCaseWrapper
from agents.deepq.files.fcn_network import DengueNet
from agents.random.agent_random_old import AleatoryAgent
from agents.qlearning.qlearning_agent import QLearning_Agent


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_EPISODES = 10
QTABLE_PATH = "final_q_table.pkl"
DQN_POLICY_PATHS = [
    "C:/Users/segun/Documents/GitHub/Dengue_Diagnostics/agents/deepq/files/dqn_dengue_policy.pth",
    "C:/Users/segun/Documents/GitHub/Dengue_Diagnostics/agents/deepq/files/dqn_dengue_policy2.pth",
    "C:/Users/segun/Documents/GitHub/Dengue_Diagnostics/agents/deepq/files/dqn_dengue_policy3.pth",
    "C:/Users/segun/Documents/GitHub/Dengue_Diagnostics/agents/deepq/files/dqn_dengue_policy4.pth",
    "C:/Users/segun/Documents/GitHub/Dengue_Diagnostics/agents/deepq/files/dqn_dengue_policy5.pth",
    "C:/Users/segun/Documents/GitHub/Dengue_Diagnostics/agents/deepq/files/dqn_dengue_policy6.pth",
    "C:/Users/segun/Documents/GitHub/Dengue_Diagnostics/agents/deepq/files/dqn_dengue_policy_SEED_850.pth",
    "C:/Users/segun/Documents/GitHub/Dengue_Diagnostics/agents/deepq/files/dqn_dengue_policy_SEED_42.pth",
    "C:/Users/segun/Documents/GitHub/Dengue_Diagnostics/agents/deepq/files/logs/experiment_50_epochs/dqn_seed_100/policy_best.pth"
]


def load_dqn_policy(env, path):
    map_shape = env.observation_space.spaces["map"].shape
    action_shape = env.action_space.n
    net = DengueNet(map_shape, action_shape, device=DEVICE).to(DEVICE)

    policy = DQNPolicy(
        model=net, optim=None, discount_factor=0.99,
        action_space=env.action_space, estimation_step=3, target_update_freq=500
    )

    try:
        policy.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
        policy.eval()
        policy.set_eps(0.0)
        print(f"--> DQN carregado: {os.path.basename(path)}")
        return policy
    except FileNotFoundError:
        print(f"ERRO: Modelo não encontrado em {path}")
        return None


def load_qlearning_agent(env):
    try:
        return QLearning_Agent(env, qtable=QTABLE_PATH)
    except FileNotFoundError:
        print("Aviso: Tabela Q-Learning não encontrada. Usando vazia.")
        return QLearning_Agent(env)


def run_dqn_episode(raw_env, policy):
    env = DengueWrapper(raw_env)
    env = CaseByCaseWrapper(env)
    obs, info = env.reset()
    terminated, truncated = False, False

    while not terminated and not truncated:
        batch = Batch(obs=[obs], info=[info])
        result = policy(batch)
        action = result.act[0].item()
        obs, reward, terminated, truncated, info = env.step(action)

    return raw_env.get_episode_metrics()


def run_qlearning_episode(env, agent):
    res = agent.reset()
    if isinstance(res, tuple) and len(res) == 2:
        obs, info = res
    else:
        obs, info = res, {}

    done = False
    while not done:
        actions = []
        cases_t = env.cases_t
        for case in cases_t:
            id = env.get_case_id(case)
            state = str(info) + str(id)
            if state not in agent.q_table:
                agent.q_table[state] = np.zeros(6)
            action_idx = np.argmax(agent.q_table[state])
            actions.append((id, action_idx))

        obs, reward, done, info = agent.step(tuple(actions))
        if hasattr(agent, 'curr_obs'):
            agent.curr_obs = (obs, info)

    return env.get_episode_metrics()


def run_random_episode(env, agent):
    res = agent.reset()
    if isinstance(res, tuple) and len(res) == 2:
        obs, info = res
    else:
        obs = res

    done = False
    while not done:
        actions = []
        cases_t = env.cases_t
        for case in cases_t:
            id = env.get_case_id(case)
            action = agent.choose_action()
            actions.append((id, action))

        obs, reward, done, info = agent.step(tuple(actions))

    return env.get_episode_metrics()


def generate_episode_params(seed, world_size=400):
    rng = np.random.default_rng(seed)
    return {
        "clinical_specificity": rng.uniform(0.5, 0.95),
        "dengue_center": (int(rng.integers(50, world_size - 50)), int(rng.integers(50, world_size - 50))),
        "chik_center": (int(rng.integers(50, world_size - 50)), int(rng.integers(50, world_size - 50))),
        "dengue_radius": int(rng.integers(50, 100)),
        "chik_radius": int(rng.integers(50, 100)),
        "epilength": 60,
        "size": world_size,
        "render_mode": None
    }


def run_benchmark():
    results = []
    seeds = [100 + i * 50 for i in range(NUM_EPISODES)]

    print(f"Iniciando Benchmark: {NUM_EPISODES} episódios.")

    dummy_env = DengueDiagnosticsEnv(size=400)
    dummy_env = DengueWrapper(dummy_env)
    dummy_env = CaseByCaseWrapper(dummy_env)

    loaded_policies = {}
    for path in DQN_POLICY_PATHS:
        name = f"DQN_{os.path.basename(path).replace('.pth', '')}"
        policy = load_dqn_policy(dummy_env, path)
        if policy:
            loaded_policies[name] = policy

    for i, seed in enumerate(seeds):
        params = generate_episode_params(seed)
        print(f"\n--- Ep {i + 1}/{NUM_EPISODES} (Seed: {seed} | Espec: {params['clinical_specificity']:.2f}) ---")

        env = DengueDiagnosticsEnv(**params)
        env.reset(seed=seed)
        metrics = run_random_episode(env, AleatoryAgent(env))
        metrics.update({'Agente': 'Aleatório', 'Seed': seed, 'Espec_Medica': params['clinical_specificity']})
        results.append(metrics)
        env.close()

        env = DengueDiagnosticsEnv(**params)
        env.reset(seed=seed)
        metrics = run_qlearning_episode(env, load_qlearning_agent(env))
        metrics.update({'Agente': 'Q-Learning', 'Seed': seed, 'Espec_Medica': params['clinical_specificity']})
        results.append(metrics)
        env.close()

        for name, policy in loaded_policies.items():
            env = DengueDiagnosticsEnv(**params)
            env.reset(seed=seed)
            metrics = run_dqn_episode(env, policy)
            metrics.update({'Agente': name, 'Seed': seed, 'Espec_Medica': params['clinical_specificity']})
            results.append(metrics)
            print(f"   > {name}: Acc={metrics['Acurácia']:.2f}")
            env.close()

    df = pd.DataFrame(results)

    target_cols = [
        'Agente', 'Seed', 'Espec_Medica',
        'Acurácia',
        'Sensibilidade (Dengue)',
        'Especificidade',
        'F1-Score',
        'Precisão',
        'Custo Total de Testes',
        'Custo por Acerto',
        'Redução de Testes (%)',
        'Recompensa Total'
    ]

    final_cols = [c for c in target_cols if c in df.columns]

    # 1. Salvar CSV Completo (Raw Data)
    df[final_cols].to_csv("benchmark_completo.csv", index=False)
    print("\nArquivo 'benchmark_completo.csv' salvo!")

    # 2. Calcular e Salvar Médias
    summary_mean = df.groupby('Agente')[final_cols[2:]].mean(numeric_only=True)
    summary_mean.to_csv("medias_finais.csv")
    print("Arquivo 'medias_finais.csv' salvo!")

    # 3. Calcular e Salvar Desvios Padrão
    summary_std = df.groupby('Agente')[final_cols[2:]].std(numeric_only=True)
    summary_std.to_csv("desvios_padroes.csv")
    print("Arquivo 'desvios_padroes.csv' salvo!")

    print("\n" + "=" * 50)
    print("RESUMO DE ACURÁCIA MÉDIA:")
    print(summary_mean['Acurácia'].sort_values(ascending=False))
    print("=" * 50)


if __name__ == "__main__":
    run_benchmark()