from agents import BaseAgent
import dengue_envs
import gymnasium as gym
from collections import defaultdict
import numpy as np
import tqdm
import matplotlib.pyplot as plt
from hashlib import md5
from line_profiler_pycharm import profile


class ExpSarsaAgent(BaseAgent):
    def __init__(self, env):
        super().__init__()
        self.epsilon = 0.05 # epsilon is the probability of selecting a random action
        self.gamma = 1 # gamma is the discount factor for future rewards
        self.step_size = 0.5 # step_size is the learning rate
        self.env = env
        self.n_actions = len(env.actions)
        self.q_value = defaultdict(lambda: np.zeros(self.n_actions))

    @profile
    def update_policy(self, state, action, new_state, reward)->np.array:
        """
        Update q(s,a) value using the expected SARSA algorithm.
        :param state: Tuple (s,a) representing the state and action.
        :return:
        """
        if isinstance(state, dict):
            state =state['clinical_diagnostic']
        if isinstance(new_state, dict):
            new_state = new_state['clinical_diagnostic']
        target = 0
        q_next = self.q_value[new_state]
        best_actions = np.argwhere(q_next == np.max(q_next)).flatten()
        for action_ in range(self.n_actions):
            if action_ in best_actions:
                target += (1 - self.epsilon)/len(best_actions) + self.epsilon/self.n_actions * q_next[action_]
            else:
                target += self.epsilon/self.n_actions * q_next[action_]
            target *= self.gamma
            self.q_value[state][action_] += self.step_size * (reward + target - self.q_value[state][action_])
        #TODO: update self.policy
        return self.q_value

    @profile
    def select_action(self, obs, info):
        """
        Select an action based on the epsilon-greedy policy.
        :param obs: The current observation
        :param info: Additional information with spatial distribution
        :return:
        """
        actions = []
        cases_t = self.env.cases_t
        state = obs['clinical_diagnostic']
        for case in cases_t:
            id = self.env.get_case_id(case)
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

    fig, ax = plt.subplots(2, 1, figsize=(12, 6))
    ax[0].plot(rewards)
    ax[0].set_title("Expected Sarsa Agent")
    ax[0].grid()
    ax[0].set_ylabel("Rewards")
    ax[1].plot(accuracy)
    ax[1].grid()
    ax[1].set_ylabel("Accuracy")
    ax[1].set_xlabel("Episodes")
    # ax[1].set_title("Accuracy")
    plt.show()

if __name__ == "__main__":
    from dengue_envs.envs.dengue_diagnostics import DengueDiagnosticsEnv
    # env = gym.make("DengueDiag-v0")  # using the standard gym interface
    env = DengueDiagnosticsEnv(epilength=12, size=500, render_mode="console")
    # env = DengueDiagnosticsEnv(epilength=12, size=500, render_mode="human")
    agent = ExpSarsaAgent(env)

    episodes = 300
    history = []
    accuracy = []
    for e in tqdm.tqdm(range(episodes), desc="Episode"):
    # for e in range(episodes):
        obs, info = env.reset()
        done = False
        while not done:
            action = agent.select_action(obs,info)
            # print ([a[1] for a in action])
            new_obs, reward, done, _, info = env.step(action)
            agent.update_policy(obs, action, new_obs, reward)
            obs = new_obs
        history.append(env.rewards[-1])
        accuracy.append(env.accuracy[-1])
        # print(f"Episode {e+1} done with reward {env.rewards[-1]} and accuracy {env.accuracy[-1]}")

    plot_results(history, accuracy)
    # plt.figure()
    # qtable = np.array(list(agent.q_value.values()))
    # plt.imshow(qtable)
    # plt.show()

