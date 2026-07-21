"""DEPRECATED — shim de compatibilidade.

Os wrappers canônicos agora vivem em `dengue_envs.wrappers`. Este módulo é
mantido apenas para não quebrar imports antigos (`from dengue_wrapper import
DengueWrapper, CaseByCaseWrapper`). Ele reexporta as versões canônicas.

A implementação legada original foi preservada em `dengue_wrapper_old.py`.
Novos scripts devem usar diretamente:

    from dengue_envs.wrappers import DengueWrapper, CaseByCaseWrapper, make_env
"""
from __future__ import annotations

import warnings

from dengue_envs.wrappers import CaseByCaseWrapper, DengueWrapper

warnings.warn(
    "agents.deepq.files.dengue_wrapper está obsoleto; use dengue_envs.wrappers "
    "(DengueWrapper, CaseByCaseWrapper, make_env).",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["DengueWrapper", "CaseByCaseWrapper"]
