"""Wrappers RL canônicos + fábrica ``make_env``."""
from dengue_envs.wrappers.case_by_case import CaseByCaseWrapper
from dengue_envs.wrappers.factory import make_env, make_raw_env
from dengue_envs.wrappers.map_tensor import DengueWrapper
from dengue_envs.wrappers.reward_scale import RewardScaleWrapper

__all__ = [
    "DengueWrapper",
    "CaseByCaseWrapper",
    "RewardScaleWrapper",
    "make_env",
    "make_raw_env",
]
