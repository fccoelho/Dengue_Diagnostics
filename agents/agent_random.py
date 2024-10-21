from dengue_envs.envs.dengue_diagnostics import DengueDiagnosticsEnv
import numpy as np

class AleatoryAgent:

    def __init__(self, env):
        self.env = env
        self.total_reward = 0
        self.curr_obs = env.reset()

    def step(self, action):
        obs, reward, done, _, info = self.env.step(action)
        self.total_reward += reward
        self.curr_obs = obs
        return obs, reward, done, info

    def choose_action(self):
        """ Return a random number between 0 and 5 """
        return np.random.choice(6)


    def reset(self):
        self.total_reward = 0
        self.curr_obs = self.env.reset()
        action = self.env.action_space.sample()  # Random action selection
        obs, reward, done, _, info = env.step(action)
        return obs, reward, done, _, info


    def play(self):
        self.curr_obs = self.reset()

        for _ in range(self.env.epilength):
            actions = []
            cases_t = self.env.cases_t
            for case in cases_t:
                id = self.env.get_case_id(case)
                action = self.choose_action()
                action = (id, action)
                actions.append(action)
            print('A', actions)
            obs, reward, done, info = self.step(tuple(actions))
            rewards = self.env.get_individual_rewards_at_t(_)
            print('R', rewards)
            if done:
                break


if __name__ == "__main__":
    # Create the environment
    env = DengueDiagnosticsEnv(epilength=365, size=500, render_mode="human")
    # Create the agent
    agent = AleatoryAgent(env)
    # Run the simulation
    agent.play()
    # Print the total reward
    print(f"Total reward: {agent.total_reward}")
    # Close the environment
    env.close()
