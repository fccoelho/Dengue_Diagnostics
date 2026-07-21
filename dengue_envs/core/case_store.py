"""Sincronização incremental dos casos observados.

Extração fiel de `DengueDiagnosticsEnv._sync_obs_cases`: adiciona apenas os
casos novos ao `obs_cases`, preservando testes/decisões já registrados nos
casos existentes (correção de estado do P0).
"""
from __future__ import annotations

import pandas as pd

OBS_COLUMNS = ["t", "x", "y", "disease", "testd", "testc", "epiconf", "agent_diagnosis"]


def sync_obs_cases(obs_cases: pd.DataFrame, cases: pd.DataFrame, clinical_model, rng) -> pd.DataFrame:
    """Retorna o `obs_cases` atualizado com os casos novos de `cases`.

    - `obs_cases`: dataframe observado atual (pode ser vazio/None).
    - `cases`: dataframe verdadeiro até o tempo t (indexado por case_id).
    - `clinical_model`: instância de `ClinicalModel` (aplica incerteza).
    - `rng`: gerador aleatório (ex.: env.np_random).

    Casos já presentes em `obs_cases` NÃO são reprocessados, o que preserva
    resultados de teste e o `agent_diagnosis` ao longo do episódio.
    """
    if cases is None or cases.empty:
        if obs_cases is None or obs_cases.empty:
            return pd.DataFrame(columns=OBS_COLUMNS)
        return obs_cases

    if obs_cases is None or obs_cases.empty:
        known = set()
    else:
        known = set(obs_cases.index)

    new_cases = cases[~cases.index.isin(known)]
    if new_cases.empty:
        return obs_cases

    new_obs = clinical_model.apply_uncertainty(new_cases, rng)
    if obs_cases is None or obs_cases.empty:
        return new_obs
    return pd.concat([obs_cases, new_obs])
