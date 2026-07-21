"""DQN (Tianshou) integrado ao framework de agentes.

Imports pesados (Tianshou) ficam em ``agent`` / ``train``; a rede pode ser
importada sem carregar o pipeline completo.
"""
from agents.deepq.network import DengueNet

__all__ = [
    "DengueNet",
    "DEFAULT_CHECKPOINT",
    "DQNAgent",
    "DQNAgentRunner",
    "build_policy",
    "load_policy",
    "observation_from_env",
]


def __getattr__(name: str):
    if name in {
        "DEFAULT_CHECKPOINT",
        "DQNAgent",
        "DQNAgentRunner",
        "build_policy",
        "load_policy",
        "observation_from_env",
    }:
        from agents.deepq import agent as _agent

        return getattr(_agent, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
