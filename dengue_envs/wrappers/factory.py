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

Distribuição espacial (``env.generator``):
  - ``synthetic`` (padrão): ``World`` SIR + truncnorm
  - ``kriging``: amostra de ``P(célula|doença)`` do Ordinary Kriging
    (requer ``surfaces_path`` apontando para o ``.npz``)
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

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
    "clinical_quality",
    "lab_sensitivity",
    "lab_specificity",
    "lab_inconclusive_prob",
    "max_case_revisits",
    "other_prevalence",
    "other_recognition_prob",
    "test_cost",
    "epi_confirm_cost",
    "epi_radius",
    "epi_threshold",
    "epi_density_scale",
    "do_nothing_cost",
    "start_day",
    "lab_delay_days",
    "settle_days",
    "reward_correct_decision",
    "penalty_incorrect_decision",
    "penalty_missed_case",
    "final_correct_bonus",
    "penalty_misdiagnosed",
    "penalty_untested_misdiagnosed",
    "penalty_unresolved",
    "shaping_conclude_bonus",
    "force_decision_after_tests",
    "render_mode",
    "randomize_outbreak",
}

# Chaves do YAML que selecionam o gerador (não vão para o __init__ do env).
_GENERATOR_KEYS = {"generator", "surfaces_path"}

_WRAPPER_BUILDERS = {
    "map_tensor": DengueWrapper,
    "case_by_case": CaseByCaseWrapper,
}

# Pipeline padrão de RL usado por DQN/PPO.
DEFAULT_WRAPPERS = ["map_tensor", "case_by_case"]


def _make_world_builder(env_cfg: dict) -> Optional[Callable]:
    """Retorna um ``world_builder(env)`` ou None (World sintético padrão)."""
    generator = env_cfg.get("generator", "synthetic")
    if generator in (None, "synthetic"):
        return None

    if generator == "kriging":
        from dengue_envs.data.kriging_generator import (
            DEFAULT_SURFACES_PATH,
            KrigingDensityGenerator,
            load_kriging_surfaces,
        )

        path = Path(env_cfg.get("surfaces_path", DEFAULT_SURFACES_PATH))
        surfaces = load_kriging_surfaces(path)

        def builder(env: DengueDiagnosticsEnv):
            gen = KrigingDensityGenerator(
                size=env.size,
                episize=env.episize,
                epilength=env.epilength,
                surfaces=surfaces,
            )
            return gen.build_world(
                random_state=env.np_random,
                dengue_r0=env.dengue_r0,
                chik_r0=env.chik_r0,
            )

        return builder

    raise ValueError(
        f"Gerador desconhecido: {generator!r}. "
        "Disponíveis: 'synthetic', 'kriging'."
    )


def make_raw_env(config: Optional[dict] = None, **kwargs) -> DengueDiagnosticsEnv:
    """Cria o ambiente base (sem wrappers).

    Parâmetros vêm de `config["env"]` e/ou kwargs diretos (kwargs têm prioridade).
    Chaves desconhecidas são ignoradas com segurança.
    """
    env_cfg = {}
    if config:
        env_cfg.update(config.get("env", {}))
    env_cfg.update(kwargs)

    world_builder = _make_world_builder(env_cfg)
    filtered = {k: v for k, v in env_cfg.items() if k in _ENV_KEYS}
    return DengueDiagnosticsEnv(**filtered, world_builder=world_builder)


def make_env(config: Optional[dict] = None, **kwargs):
    """Cria o ambiente base e aplica a pilha de wrappers.

    - `config["env"]`: parâmetros do ambiente.
    - `config["wrappers"]`: lista de wrappers a aplicar, na ordem
      (ex.: ["map_tensor", "case_by_case"]). Se ausente, usa DEFAULT_WRAPPERS.
    - `config["per_case_reward"]`: se True, o `case_by_case` entrega a
      recompensa de cada decisão no passo daquele caso, em vez de agregar o
      dia inteiro num único passo (ver CaseByCaseWrapper).
    - `config["context_features"]`: se True, o `case_by_case` acrescenta à
      observação a evidência acumulada sobre a competência do médico.
    - `config["map_size"]`: resolução do tensor de mapa (default: tamanho do
      mundo). Precisa dividir `env.size`. Ver `DengueWrapper` para a medição
      que motiva reduzi-la.
    - kwargs extras são repassados ao ambiente base.
    """
    env = make_raw_env(config, **kwargs)

    wrappers = DEFAULT_WRAPPERS
    if config and "wrappers" in config:
        wrappers = config["wrappers"]
    context_features = bool((config or {}).get("context_features", False))
    per_case_reward = bool((config or {}).get("per_case_reward", False))
    # Resolução da observação de mapa. `None` = mesma do mundo (comportamento
    # histórico). Ver DengueWrapper: o encoder reduz tudo a 6x6 de qualquer
    # forma, então resolução extra só encarece o caminho dos dados.
    map_size = (config or {}).get("map_size")

    for name in wrappers:
        if name not in _WRAPPER_BUILDERS:
            raise ValueError(
                f"Wrapper desconhecido: {name!r}. "
                f"Disponíveis: {sorted(_WRAPPER_BUILDERS)}"
            )
        if name == "case_by_case":
            env = CaseByCaseWrapper(
                env,
                context_features=context_features,
                per_case_reward=per_case_reward,
            )
        elif name == "map_tensor":
            env = DengueWrapper(env, map_size=map_size)
        else:
            env = _WRAPPER_BUILDERS[name](env)

    # Reescala a recompensa apenas se pedido explicitamente (uso: treino).
    # A avaliação/benchmark deve rodar SEM isto, para reportar a recompensa na
    # escala original e manter comparabilidade com resultados anteriores.
    scale = (config or {}).get("reward_scale")
    if scale:
        from dengue_envs.wrappers.reward_scale import RewardScaleWrapper

        env = RewardScaleWrapper(env, float(scale))

    return env
