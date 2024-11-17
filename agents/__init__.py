from typing import Dict, Any

class BaseAgent:
    def __init__(self):
        self.policy = {}  # policy is a function that maps states to actions
        self.q_value = None  # action_value is a function that maps states and actions to values. Here it can be an array with shape (n_states, n_actions)
        self.epsilon = 0.1 # epsilon is the probability of selecting a random action
        self.gamma = 0.99 # gamma is the discount factor for future rewards
        self.step_size = 0.5 # step_size is the learning rate

    def play(self, n_episodes=1, render=False):
        for _ in range(n_episodes):
            state = self.env.reset()
            done = False
            while not done:
                action = self.policy(state)
                state, reward, done, _ = self.env.step(action)
                if render:
                    self.env.render()
            self.env.close()

    def update_policy(self, state: Dict[str,Any], action:str):
        raise NotImplementedError

    def select_action(self, state: Dict[str,Any]) -> str:
        raise NotImplementedError