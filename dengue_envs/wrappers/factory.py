"""Fábrica de ambientes: cria o env bruto e aplica os wrappers de RL.

Ponto único de construção do ambiente para todos os algoritmos. Recebe um
dicionário de configuração (compatível com o YAML previsto na Fase 2) e retorna
o ambiente pronto para treino/avaliação.

Exemplo:
    from dengue_envs.wrappers.factory import make_env
    env = make_env({
        "env": {"size": 400, "episize": 150, "epilength": 60,
                 "reward_delay_days": 5, "start_day": 1},
        "wrappers": ["map_tensor", "case_by_case"],
    })
"""
from __future__ import annotations

from typing import Optional

from dengue_envs.envs.dengue_diagnostics import DengueDiagnosticsEnv
from dengue_envs.wrappers.case_by_case import CaseByCaseWrapper
from dengue_envs.wrappers.map_tensor import DengueWrapper

# Parâmetros aceitos pelo DengueDiagnosticsEnv.__init__
_ENV_KEYS = {
    "size",
    "episize",
    "epilength",
    "reward_delay_days",
    "dengue_center",
    "chik_center",
    "dengue_radius",
    "chik_radius",
    "clinical_specificity",
    "start_day",
    "render_mode",
}

_WRAPPER_BUILDERS = {
    "map_tensor": DengueWrapper,
    "case_by_case": CaseByCaseWrapper,
}

# Pipeline padrão de RL usado por DQN/PPO.
DEFAULT_WRAPPERS = ["map_tensor", "case_by_case"]


def make_raw_env(config: Optional[dict] = None, **kwargs) -> DengueDiagnosticsEnv:
    """Cria o ambiente base (sem wrappers).

    Parâmetros vêm de `config["env"]` e/ou kwargs diretos (kwargs têm prioridade).
    Chaves desconhecidas são ignoradas com segurança.
    """
    env_cfg = {}
    if config:
        env_cfg.update(config.get("env", {}))
    env_cfg.update(kwargs)

    filtered = {k: v for k, v in env_cfg.items() if k in _ENV_KEYS}
    return DengueDiagnosticsEnv(**filtered)


def make_env(config: Optional[dict] = None, **kwargs):
    """Cria o ambiente base e aplica a pilha de wrappers.

    - `config["env"]`: parâmetros do ambiente.
    - `config["wrappers"]`: lista de wrappers a aplicar, na ordem
      (ex.: ["map_tensor", "case_by_case"]). Se ausente, usa DEFAULT_WRAPPERS.
    - kwargs extras são repassados ao ambiente base.
    """
    env = make_raw_env(config, **kwargs)

    wrappers = DEFAULT_WRAPPERS
    if config and "wrappers" in config:
        wrappers = config["wrappers"]

    for name in wrappers:
        if name not in _WRAPPER_BUILDERS:
            raise ValueError(
                f"Wrapper desconhecido: {name!r}. "
                f"Disponíveis: {sorted(_WRAPPER_BUILDERS)}"
            )
        env = _WRAPPER_BUILDERS[name](env)

    return env
