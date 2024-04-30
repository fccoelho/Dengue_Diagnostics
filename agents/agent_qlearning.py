import numpy as np
from dengue_envs.envs.dengue_diagnostics import DengueDiagnosticsEnv


class QLearningAgent:
    def __init__(self, env):
        self.env = env
        self.total_reward = 0
        self.curr_obs = env.reset()
        self.gamma = 0.99
        self.alpha = 0.1
        self.values = {}
        self.states = {}
        self.actions = {}
        self.rewards = {}
        self.next_states = {}
        self.done = {}
        self.total_reward = 0
        self.curr_obs = env.reset()
        self.episode = 0

    def best_value_and_action(self, state):
        best_v = -np.inf
        best_a = None
        state = tuple(state.items())
        for action in range(4):
            if (state, action) not in self.values:
                self.values[(state, action)] = 0
            if self.values[(state, action)] > best_v:
                best_v = self.values[(state, action)]
                best_a = action
        return best_v, best_a

    def value_update(self, state, action, reward, next_state):
        best_v, _ = self.best_value_and_action(next_state)
        new_v = reward + self.gamma * best_v
        if isinstance(state, tuple):
            state = state[0]
        old_v = self.values[(tuple(state.items()), action)]
        self.values[(tuple(state.items()), action)] = old_v * (1 - self.alpha) + new_v * self.alpha

    def step(self, action):
        obs, reward, done, _, info = self.env.step(action)
        self.total_reward += reward
        self.rewards[self.episode] = reward
        self.next_states[self.episode] = obs
        self.done[self.episode] = done
        self.value_update(self.states[self.episode], self.actions[self.episode], reward, obs)
        self.states[self.episode] = obs
        self.actions[self.episode] = action
        self.episode += 1
        return obs, reward, done, info

    def reset(self):
        self.total_reward = 0
        self.curr_obs = self.env.reset()
        self.states[self.episode] = self.curr_obs
        self.actions[self.episode] = self.env.action_space.sample()
        return self.curr_obs


if __name__ == "__main__":
    # Create the environment
    env = DengueDiagnosticsEnv(epilength=365, size=500, render_mode="human")
    # Create the agent
    agent = QLearningAgent(env)
    # Run the simulation
    for _ in range(100):
        obs, reward, done, info = agent.step(env.action_space.sample())
        if done:
            break
    # Print the total reward
    print(f"Total reward: {agent.total_reward}")
    # Close the environment
    env.close()
