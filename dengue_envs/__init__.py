from gym.envs.registration import register

register(
    id="dengue_envs/DengueDiag-v0",
    entry_point="dengue_envs.envs:DengueDiagnosticsEnv",
)
