"""Shim depreciado: use ``agents.deepq.network.DengueNet``.

Mantido para não quebrar imports legados / testes antigos.
"""
from __future__ import annotations

import warnings

from agents.deepq.network import DengueNet

warnings.warn(
    "agents.deepq.files.fcn_network está depreciado; "
    "importe DengueNet de agents.deepq.network.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["DengueNet"]
