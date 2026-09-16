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
    "epi_model",
    "initial_infected_fraction",
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
_GENERATOR_KEYS = {"generator", "surfaces_path", "mix", "augment_surfaces"}

_WRAPPER_BUILDERS = {
    "map_tensor": DengueWrapper,
    "case_by_case": CaseByCaseWrapper,
}

# Pipeline padrão de RL usado por DQN/PPO.
DEFAULT_WRAPPERS = ["map_tensor", "case_by_case"]


def _synthetic_world(env: DengueDiagnosticsEnv):
    """Constrói explicitamente o World sintético padrão.

    O env já faz isso quando `world_builder is None`; esta função existe para
    que o gerador `mixed` possa ESCOLHER o sintético em tempo de execução, o
    que exige construí-lo de dentro de um builder.
    """
    from dengue_envs.data.generator import World

    return World(
        env.size,
        env.episize,
        env.epilength,
        env.dengue_center,
        env.chik_center,
        env.dengue_radius,
        env.chik_radius,
        dengue_r0=env.dengue_r0,
        chik_r0=env.chik_r0,
        other_prevalence=env.other_prevalence,
        random_state=env.np_random,
    )


def _make_world_builder(env_cfg: dict) -> Optional[Callable]:
    """Retorna um ``world_builder(env)`` ou None (World sintético padrão)."""
    generator = env_cfg.get("generator", "synthetic")
    if generator in (None, "synthetic"):
        return None

    if generator == "mixed":
        # Sorteia a distribuição espacial A CADA MUNDO NOVO (ver `_create_world`).
        #
        # Motivação medida: hoje o mapa é decorativo — zerá-lo muda 0-2% das
        # decisões do agente, porque dentro de UMA distribuição ele não
        # acrescenta nada às features do caso. Misturando duas, o mapa passa a
        # ser a única forma de saber em que regime se está — e os regimes pedem
        # políticas opostas: no sintético a posição acerta a doença em 93,7% (o
        # exame é redundante); no kriging, 47,1% (o exame é tudo).
        #
        # Config:
        #   generator: mixed
        #   mix:
        #     - {generator: synthetic, weight: 0.5}
        #     - {generator: kriging,   weight: 0.5}
        entradas = env_cfg.get("mix")
        if not entradas:
            raise ValueError(
                "generator: mixed exige a chave `mix` com as distribuições. "
                "Ex.: mix: [{generator: synthetic}, {generator: kriging}]"
            )
        construtores, pesos = [], []
        for item in entradas:
            sub = dict(env_cfg)
            sub.pop("mix", None)
            sub.update(item)
            peso = float(sub.pop("weight", 1.0))
            if peso <= 0:
                continue
            b = _make_world_builder(sub)
            construtores.append(b if b is not None else _synthetic_world)
            pesos.append(peso)
        if not construtores:
            raise ValueError("`mix` não produziu nenhuma distribuição com peso > 0")
        pesos = [p / sum(pesos) for p in pesos]

        def mixed_builder(env: DengueDiagnosticsEnv):
            # `env.np_random` é o RNG semeado do episódio: a escolha é
            # reprodutível para uma dada seed.
            i = int(env.np_random.choice(len(construtores), p=pesos))
            env.mixture_component = i
            return construtores[i](env)

        return mixed_builder

    if generator == "kriging":
        from dengue_envs.data.kriging_generator import (
            DEFAULT_SURFACES_PATH,
            KrigingDensityGenerator,
            augment_surfaces,
            load_kriging_surfaces,
        )

        path = Path(env_cfg.get("surfaces_path", DEFAULT_SURFACES_PATH))
        surfaces = load_kriging_surfaces(path)
        # `augment` sorteia uma transformação rígida por episódio (rotação,
        # espelho, deslocamento). Sem isso a superfície é SEMPRE a mesma cidade
        # no mesmo surto, e o agente pode decorar a geografia. A transformação
        # é conjunta às duas doenças, então preserva a dificuldade — ver
        # `augment_surfaces`.
        augment = bool(env_cfg.get("augment_surfaces", False))

        def builder(env: DengueDiagnosticsEnv):
            sup = augment_surfaces(surfaces, env.np_random) if augment else surfaces
            gen = KrigingDensityGenerator(
                size=env.size,
                episize=env.episize,
                epilength=env.epilength,
                surfaces=sup,
            )
            return gen.build_world(
                random_state=env.np_random,
                dengue_r0=env.dengue_r0,
                chik_r0=env.chik_r0,
                epi_model=env.epi_model,
                initial_infected_fraction=env.initial_infected_fraction,
            )

        return builder

    raise ValueError(
        f"Gerador desconhecido: {generator!r}. "
        "Disponíveis: 'synthetic', 'kriging', 'mixed'."
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
    temporal_features = bool((config or {}).get("temporal_features", False))
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
                temporal_features=temporal_features,
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
