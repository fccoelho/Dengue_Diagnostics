from agents import BaseAgent
import dengue_envs
import gymnasium as gym
from collections import defaultdict
import numpy as np
import tqdm
import matplotlib.pyplot as plt

class ExpSarsaAgent(BaseAgent):
    def __init__(self, env):
        super().__init__()
        self.env = env
        self.n_actions = len(env.actions)
        self.q_value = defaultdict(lambda: np.zeros(self.n_actions))

    def update_policy(self, state, action, new_state, reward)->np.array:
        """
        Update q(s,a) value using the expected SARSA algorithm.
        :param state: Tuple (s,a) representing the state and action.
        :return:
        """
        if isinstance(state, dict):
            state = str(state)
        if isinstance(new_state, dict):
            new_state = str(new_state)
        target = 0
        q_next = self.q_value[new_state]
        best_actions = np.argwhere(q_next == np.max(q_next)).flatten()
        for action_ in range(self.n_actions):
            if action_ in best_actions:
                target += (1 - self.epsilon)/len(best_actions) + self.epsilon/self.n_actions * q_next[action_]
            else:
                target += self.epsilon/self.n_actions * q_next[action_]
        target *= self.gamma
        self.q_value[state] += self.step_size * (reward + target - self.q_value[state])
        #TODO: update self.policy
        return self.q_value

    def select_action(self, obs, info):
        """
        Select an action based on the epsilon-greedy policy.
        :param obs: The current observation
        :param info: Additional information with spatial distribution
        :return:
        """
        actions = []
        cases_t = self.env.cases_t
        for case in cases_t:
            id = self.env.get_case_id(case)
            state = str(info)
            if np.random.binomial(1, self.epsilon):
                actions.append((id, np.random.choice(self.n_actions)))
            else:
                values_ = self.q_value[state]
                actions.append((id, np.random.choice(np.argwhere(values_ == np.max(values_)).flatten())))
        return tuple(actions)

def plot_results(rewards, accuracy):
    """
    Plot the results of the agent's performance.
    :param rewards: List of rewards
    :param accuracy: List of accuracies
    :return:
    """

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].plot(rewards)
    ax[0].set_title("Rewards")
    ax[1].plot(accuracy)
    ax[1].set_title("Accuracy")
    plt.show()

if __name__ == "__main__":
    from dengue_envs.envs.dengue_diagnostics import DengueDiagnosticsEnv
    # env = gym.make("DengueDiag-v0")  # using the standard gym interface
    env = DengueDiagnosticsEnv(epilength=12, size=500, render_mode="console")
    agent = ExpSarsaAgent(env)
    obs, info = env.reset()
    episodes = 500
    history = []
    accuracy = []
    for _ in tqdm.tqdm(range(episodes), desc="Episode"):
        done = False
        while not done:
            action = agent.select_action(obs,info)
            new_obs, reward, done, _, info = env.step(action)
            agent.update_policy(obs, action, new_obs, reward)
            obs = new_obs
        history.append(env.rewards[-1])
        accuracy.append(env.accuracy[-1])

    plot_results(history, accuracy)

