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
            action = self.env.action_space.sample()
        else:
            action = np.argmax(self.q_table[state])

        print(f"Action: {action}")
        return action

    def get_all_actions(self):
        return env.cases

    def step(self, action):
        obs, reward, done, _, info = self.env.step(action)
        self.total_reward += reward
        self.curr_obs = obs
        return obs, reward, done, info

    def reset(self):
        self.total_reward = 0
        self.curr_obs = self.env.reset()
        return self.curr_obs

    def update_q_table(self, state, action, reward, next_state):
        self.q_table[state][action] = self.q_table[state][action] + self.alpha * (
                reward + self.gamma * np.max(self.q_table[next_state]) - self.q_table[state][action])

    def run(self):
        for _ in range(365):
            state = str(self.curr_obs)
            if _ == 0:
                self.q_table[state] = np.zeros(len(self.get_all_actions()))
                action = env.action_space.sample()
            else:
                action = self.choose_action(state)
            obs, reward, done, info = self.step(action)
            next_state = str(obs)
            if next_state not in self.q_table:
                self.q_table[next_state] = np.zeros(len(self.get_all_actions()))
            self.update_q_table(state, action, reward, next_state)
            if done:
                break
        print(f"Total reward: {self.total_reward}")
        self.env.close()


if __name__ == "__main__":
    # Create the environment
    env = DengueDiagnosticsEnv(epilength=365, size=500, render_mode="human")
    # Create the agent
    agent = QLearning_Agent(env)
    # Run the simulation
    agent.run()
