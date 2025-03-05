from dengue_envs.envs.dengue_diagnostics import DengueDiagnosticsEnv
from stable_baselines3.common.env_checker import check_env
import gymnasium as gym
from stable_baselines3 import DQN

env = DengueDiagnosticsEnv()
print("Observation Space:", env.observation_space)
print("Action Space:", env.action_space)

check_env(env)



