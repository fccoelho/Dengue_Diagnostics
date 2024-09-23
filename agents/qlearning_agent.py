from dengue_envs.envs.dengue_diagnostics import DengueDiagnosticsEnv
import numpy as np


class QLearning_Agent():
    def __init__(self, env):
        self.env = env
        self.total_reward = 0
        self.curr_obs = env.reset()
        #our qtable will be a dict with the state as the key and the value as a np array
        #with the the q value (the order is the same as the action space)
        self.q_table = dict()
        self.alpha = 0.1
        self.gamma = 0.6
        self.epsilon = 0.1

    def choose_action(self, state):
        if np.random.uniform(0, 1) < self.epsilon:
            action = np.random.choice(6)
        else:
            action = np.argmax(self.q_table[state])
            print("Action: ", action)
        return action

    def step(self, action):
        print(action, 'On step')
        obs, reward, done, _, info = self.env.step(action)
        self.total_reward += reward
        return obs, reward, done, info

    def save_q_table(self):
        # save a txt
        with open("q_table.txt", "w") as f:
            for key, value in self.q_table.items():
                f.write(f"{key[-1]}: {value}\n")

    def reset(self):
        self.total_reward = 0
        self.curr_obs = self.env.reset()
        action = self.env.action_space.sample()  # Random action selection
        obs, reward, done, _, info = env.step(action)
        return obs, reward, done, _, info

    def update_q_table(self, state, action, reward, next_state):
        r = self.q_table[state][action] = self.q_table[state][action] + self.alpha * (
                reward + self.gamma * np.max(self.q_table[next_state]) - self.q_table[state][action])

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
            print(obs['clinical_diagnostic'])
            rewards = self.env.get_individual_rewards_at_t(_)
            for i, case in enumerate(cases_t):
                id = self.env.get_case_id(case)
                state = str(info) + str(id)
                if state not in self.q_table:
                    self.q_table[state] = np.zeros(6)
                self.update_q_table(str(self.curr_obs[-1]) + str(id), actions[i][1], reward, state)

            if done:
                break

            if _ % 10 == 0:
                self.save_q_table()

        print(f"Total reward: {self.total_reward}")
        self.env.close()


if __name__ == "__main__":
    # Create the environment
    env = DengueDiagnosticsEnv(epilength=365, size=500, render_mode="human")
    # Create the agent
    agent = QLearning_Agent(env)
    # Run the simulation
    agent.run()
