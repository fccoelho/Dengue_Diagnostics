from dengue_envs.envs.dengue_diagnostics import DengueDiagnosticsEnv
import numpy as np
import pickle
import pygame
import matplotlib.pyplot as plt

class QLearning_Agent():
    def __init__(self, env, qtable = None):
        self.env = env
        self.render_mode = env.render_mode
        self.total_reward = 0
        self.curr_obs = env.reset()
        #our qtable will be a dict with the state as the key and the value as a np array
        #with the the q value (the order is the same as the action space)
        if qtable is None:
            self.q_table = dict()
        else:
            with open(qtable, "rb") as f:
                self.q_table = pickle.load(f)
        self.alpha = 0.5
        self.gamma = 0.5
        self.epsilon = 0.15

    def choose_action(self, state):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(6)
        if np.random.uniform(0, 1) < self.epsilon:
            action = np.random.choice(6)
        else:
            action = np.argmax(self.q_table[state])
        return action

    def step(self, action):
        obs, reward, done, _, info = self.env.step(action)
        self.total_reward += reward
        return obs, reward, done, info

    def save_q_table(self, final=False):
        # save a txt
        with open("q_table.txt", "w") as f:
            for key, value in self.q_table.items():
                f.write(f"{key[-3:]}: {value}\n")

        with open("q_table.pkl", "wb") as f:
            pickle.dump(self.q_table, f)

        if final:
            with open("final_q_table.pkl", "wb") as f:
                pickle.dump(self.q_table, f)

    def reset(self):
        self.total_reward = 0
        self.curr_obs = self.env.reset()
        obs, info = self.env.reset()
        return obs, info

    def update_q_table(self, state, action, reward, next_state):
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(6)
        self.q_table[state][action] += self.alpha * (
                reward + self.gamma * np.max(self.q_table[next_state]) - self.q_table[state][action]
        )

    def run(self):
        self.curr_obs = self.reset()
        for _ in range(self.env.epilength):
            actions = []
            cases_t = self.env.cases_t
            for case in cases_t:
                id = self.env.get_case_id(case)
                state = str(self.curr_obs[-1]) + str(id)
                if state not in self.q_table:
                    self.q_table[state] = np.zeros(6)
                action = self.choose_action(state)
                action = (id, action)
                actions.append(action)
            obs, reward, done, info = self.step(tuple(actions))

            for i, case in enumerate(cases_t):
                id = self.env.get_case_id(case)
                state = str(info) + str(id)
                if state not in self.q_table:
                    self.q_table[state] = np.zeros(6)
                self.update_q_table(str(self.curr_obs[-1]) + str(id), actions[i][1], reward, state)

            self.curr_obs = (obs, info)

            if done:
                self.save_q_table(final = True)
                break

            if _ % 2 == 0:
                self.save_q_table()
        self.save_q_table(final=True)

        if self.render_mode == "human":
            self.env.render()  # This is crucial for updating the window
            pygame.event.pump()  # Keep Pygame events flowing (avoids freezing)

            pygame.time.delay(10)  # Adjust this value as needed

        self.env.close()

    def play(self):
        # only run based on q table doesnt update
        self.curr_obs = self.reset()

        for _ in range(self.env.epilength):
            actions = []
            cases_t = self.env.cases_t
            self.z += len(cases_t)
            for case in cases_t:
                id = self.env.get_case_id(case)
                state = str(self.curr_obs[-1]) #+ str(id)
                if state not in self.q_table:
                    self.q_table[state] = np.zeros(6)
                action = np.argmax(self.q_table[state])
                action = (id, action)
                actions.append(action)
            obs, reward, done, info = self.step(tuple(actions))
            rewards = self.env.get_individual_rewards_at_t(_)
            print('R', rewards)
            if done:
                break


if __name__ == "__main__":
        history = []
        accuracy = []
        env = DengueDiagnosticsEnv(epilength=12, size=500, render_mode="console")
        agent = QLearning_Agent(env)
        for i in range(200):
            agent.run()
            print(f"Total reward: {agent.total_reward} for run {i+1}")
            history.append(agent.total_reward)
            accuracy.append(env.accuracy[-1])

        # Create a new figure
        fig, [ax1, ax2] = plt.subplots(2,1)
        # Plot the rewards
        ax1.plot(history)
        # Add a title
        ax1.set_title("Total reward per episode")
        # Add labels to the axes
        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Total reward")
        ax2.plot(accuracy)
        ax2.set_title("Accuracy per episode")
        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Accuracy")

        # Display the plot
        plt.show()
        # save plot
        plt.savefig("q_learning.png")

    # # train 100 time
    # for i in range(100):
    #     env = DengueDiagnosticsEnv(epilength=365, size=500, render_mode="human")
    #     agent = QLearning_Agent(env) if i == 0 else QLearning_Agent(env, qtable="final_q_table.pkl")
    #     agent.run()
    #     print(f"Episode {i+1} done")

