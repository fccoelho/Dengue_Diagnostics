from dengue_envs.envs.dengue_diagnostics import DengueDiagnosticsEnv
import numpy as np
import pygame
import matplotlib.pyplot as plt


class AleatoryAgent:

    def __init__(self, env):
        self.env = env
        self.total_reward = 0
        self.curr_obs = None  # Inicializa como None

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        self.total_reward += reward
        self.curr_obs = obs
        return obs, reward, done, info

    def choose_action(self):
        """ Return a random number between 0 and 5 """
        return np.random.choice(6)

    def reset(self):
        self.total_reward = 0
        self.curr_obs, info = self.env.reset()
        return self.curr_obs, info

    def play(self, episode_id=1):
        self.reset()

        step = 0
        while True:
            actions = []
            cases_t = self.env.cases_t

            # Se não houver casos neste timestep, passamos uma lista vazia ou ação vazia
            for case in cases_t:
                id = self.env.get_case_id(case)
                action = self.choose_action()
                action = (id, action)
                actions.append(action)

            # Converte para tupla conforme esperado pelo ambiente
            obs, reward, done, info = self.step(tuple(actions))

            # Renderização
            if self.env.render_mode == "human":
                self.env.render()
                pygame.event.pump()
                pygame.time.delay(10)

            # Debug rewards
            # rewards = self.env.get_individual_rewards_at_t(self.env.t) # Cuidado: t já incrementou no step

            if done:
                print(f"Episódio {episode_id} finalizado no step {step}.")
                self.env.plot_confusion_map(
                    title=f"Mapa de Confusão - Episódio {episode_id} (Aleatório)",
                    save_path=f"confusao_random_ep_{episode_id}.png"
                )
                break

            step += 1


if __name__ == "__main__":
    history = []
    # Aumentei para 2 episódios para testar o reset
    num_episodes = 2

    # Render mode "human" para ver o Pygame, ou None para ser mais rápido
    env = DengueDiagnosticsEnv(epilength=30, size=500, render_mode="human")

    agent = AleatoryAgent(env)

    for i in range(num_episodes):
        print(f"Iniciando episódio {i + 1}...")
        agent.play(episode_id=i + 1)
        print(f"Total reward: {agent.total_reward}")
        history.append(agent.total_reward)

    env.close()

    plt.figure()
    plt.plot(history)
    plt.title("Total reward per episode")
    plt.xlabel("Episode")
    plt.ylabel("Total reward")
    plt.savefig("random_agent_results.png")
    plt.show()