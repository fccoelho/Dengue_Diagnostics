from dengue_envs.envs.dengue_diagnostics import DengueDiagnosticsEnv
from stable_baselines3.common.env_checker import check_env

env = DengueDiagnosticsEnv()

check_env(env)
