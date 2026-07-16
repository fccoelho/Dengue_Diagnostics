"""Carregamento e composição de configuração por YAML.

Permite descrever ambiente, wrappers e treino em arquivos YAML e construir o
ambiente via `dengue_envs.wrappers.factory` sem escrever código. Suporta
composição por `include:` (merge profundo), de modo que uma config de treino
possa reaproveitar uma config de ambiente.

Nada aqui modifica o pacote `dengue_envs`: apenas o consome.
"""
from __future__ import annotations

import copy
import os
from typing import Any, Callable, Dict, List, Optional

import yaml

# Diretório raiz das configs empacotadas (experiments/configs)
CONFIGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs")


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Merge recursivo de dicionários (override vence).

    Dicionários são combinados chave a chave; qualquer outro tipo (inclusive
    listas) é substituído pelo valor de `override`.
    """
    result = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _read_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return data or {}


def load_config(path: str) -> Dict[str, Any]:
    """Carrega um YAML resolvendo `include:` por merge profundo.

    `include` é uma lista de caminhos relativos ao arquivo atual. Os includes
    são mesclados na ordem informada e, por fim, as chaves do próprio arquivo
    têm prioridade sobre os includes.
    """
    path = os.path.abspath(path)
    raw = _read_yaml(path)

    includes: List[str] = raw.pop("include", []) or []
    if isinstance(includes, str):
        includes = [includes]

    merged: Dict[str, Any] = {}
    base_dir = os.path.dirname(path)
    for inc in includes:
        inc_path = inc if os.path.isabs(inc) else os.path.join(base_dir, inc)
        merged = deep_merge(merged, load_config(inc_path))

    return deep_merge(merged, raw)


def normalize_env_config(env_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Ajusta tipos vindos do YAML para o que o ambiente espera.

    - `clinical_specificity`: lista [min, max] -> tupla (min, max);
      lista [v] -> float v.
    - `dengue_center` / `chik_center`: lista [x, y] -> tupla (x, y).
    """
    cfg = copy.deepcopy(env_cfg)

    spec = cfg.get("clinical_specificity")
    if isinstance(spec, list):
        if len(spec) == 1:
            cfg["clinical_specificity"] = float(spec[0])
        elif len(spec) >= 2:
            cfg["clinical_specificity"] = (float(spec[0]), float(spec[1]))

    for key in ("dengue_center", "chik_center"):
        if isinstance(cfg.get(key), list):
            cfg[key] = tuple(cfg[key])

    return cfg


def _normalized_config(config: Dict[str, Any]) -> Dict[str, Any]:
    cfg = copy.deepcopy(config)
    if "env" in cfg:
        cfg["env"] = normalize_env_config(cfg["env"])
    return cfg


def build_env(config: Dict[str, Any], **overrides):
    """Constrói o ambiente (env + wrappers) a partir de um dicionário de config.

    `overrides` são repassados como kwargs diretos ao ambiente base (têm
    prioridade sobre `config["env"]`).
    """
    # Import tardio para manter este módulo leve e testável.
    from dengue_envs.wrappers.factory import make_env

    return make_env(_normalized_config(config), **overrides)


def build_raw_env(config: Dict[str, Any], **overrides):
    """Constrói apenas o ambiente base (sem wrappers)."""
    from dengue_envs.wrappers.factory import make_raw_env

    return make_raw_env(_normalized_config(config), **overrides)


def build_env_factory(config: Dict[str, Any], **overrides) -> Callable[[], Any]:
    """Retorna uma função sem argumentos que constrói um novo ambiente.

    Útil para bibliotecas de RL que exigem um `env_factory` (ex.: Tianshou).
    """
    frozen = copy.deepcopy(config)
    frozen_overrides = copy.deepcopy(overrides)

    def _factory():
        return build_env(frozen, **frozen_overrides)

    return _factory


def get_train_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Retorna a seção `train` com defaults seguros."""
    train = copy.deepcopy(config.get("train", {}))
    train.setdefault("algorithm", "random")
    train.setdefault("seed", None)
    return train


def load_named_config(name: str) -> Dict[str, Any]:
    """Carrega uma config empacotada por caminho relativo a `experiments/configs`.

    Ex.: load_named_config("train/random.yaml").
    """
    return load_config(os.path.join(CONFIGS_DIR, name))
