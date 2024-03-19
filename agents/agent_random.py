from dengue_envs.envs.dengue_diagnostics import DengueDiagnosticsEnv


class AleatoryAgent:

    def __init__(self, env):
        self.env = env
        self.total_reward = 0
        self.curr_obs = env.reset()

    def step(self, action):
        action = self.env.action_space.sample()
        obs, reward, done, _, info = self.env.step(action)
        self.total_reward += reward
        self.curr_obs = obs
        return obs, reward, done, info

    def reset(self):
        self.total_reward = 0
        self.curr_obs = self.env.reset()
        return self.curr_obs


if __name__ == "__main__":
    # Create the environment
    env = DengueDiagnosticsEnv(epilength=365, size=500, render_mode="human")
    # Create the agent
    agent = AleatoryAgent(env)
    # Run the simulation
    for _ in range(100):
        obs, reward, done, info = agent.step(env.action_space.sample())
        if done:
            break
    # Print the total reward
    print(f"Total reward: {agent.total_reward}")
    # Close the environment
    env.close()
