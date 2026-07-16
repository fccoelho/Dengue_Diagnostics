"""Modelo clínico: incerteza diagnóstica, testes de laboratório e atualização de status.

Extração fiel da lógica de `DengueDiagnosticsEnv` (métodos
`_apply_clinical_uncertainty`, `_dengue_lab_test`, `_chik_lab_test`,
`_update_case_status`) para funções/métodos reutilizáveis e testáveis.
"""
from __future__ import annotations

import pandas as pd

# Rótulos de doença
DENGUE = 0
CHIK = 1
OTHER = 2

# Resultados de teste
NOT_TESTED = 0
NEGATIVE = 1
POSITIVE = 2
INCONCLUSIVE = 3


class ClinicalModel:
    """Encapsula a incerteza clínica e os testes de laboratório.

    Todos os métodos recebem explicitamente um gerador de números aleatórios
    (`rng`, tipicamente `env.np_random`), o que os torna determinísticos e
    facilmente testáveis.
    """

    def __init__(
        self,
        clinical_specificity: float,
        other_prob: float = 0.01,
        sensitivity: float = 0.9,
        specificity: float = 0.9,
        inconclusive_prob: float = 0.1,
    ):
        self.clinical_specificity = clinical_specificity
        self.other_prob = other_prob
        self.sensitivity = sensitivity
        self.specificity = specificity
        self.inconclusive_prob = inconclusive_prob

    # -- Incerteza clínica ---------------------------------------------------
    def apply_uncertainty(self, cases_df: pd.DataFrame, rng) -> pd.DataFrame:
        """Aplica incerteza clínica a casos recém-reportados.

        Espelha `_apply_clinical_uncertainty`: a amostragem acontece uma única
        vez por caso, de modo que o diagnóstico observado é estável pelo resto
        do episódio. Retorna uma cópia com a coluna `agent_diagnosis`.
        """
        obs = cases_df.copy()
        for idx in obs.index:
            true_disease = obs.at[idx, "disease"]
            if rng.uniform() < self.other_prob:
                obs.at[idx, "disease"] = OTHER
                continue
            if true_disease == DENGUE:
                if rng.uniform() > self.clinical_specificity:  # vira chik
                    obs.at[idx, "disease"] = CHIK
            elif true_disease == CHIK:
                if rng.uniform() > self.clinical_specificity:  # vira dengue
                    obs.at[idx, "disease"] = DENGUE

        obs["agent_diagnosis"] = obs["disease"]
        return obs

    # -- Testes de laboratório ----------------------------------------------
    def dengue_lab_test(self, true_disease: int, rng) -> int:
        """Resultado do teste de dengue condicionado à doença VERDADEIRA."""
        if rng.uniform() < self.inconclusive_prob:
            return INCONCLUSIVE
        if true_disease == DENGUE:
            return POSITIVE if rng.uniform() < self.sensitivity else NEGATIVE
        return NEGATIVE if rng.uniform() < self.specificity else POSITIVE

    def chik_lab_test(self, true_disease: int, rng) -> int:
        """Resultado do teste de chikungunya condicionado à doença VERDADEIRA."""
        if rng.uniform() < self.inconclusive_prob:
            return INCONCLUSIVE
        if true_disease == CHIK:
            return POSITIVE if rng.uniform() < self.sensitivity else NEGATIVE
        return NEGATIVE if rng.uniform() < self.specificity else POSITIVE


def update_case_status(obs_cases: pd.DataFrame, action: int, index, result: int) -> None:
    """Atualiza status do caso e `agent_diagnosis` conforme o resultado.

    Espelha `_update_case_status`. Modifica `obs_cases` in-place.
    action: 0 = teste dengue, 1 = teste chik, 2 = epi confirm.
    """
    if action == 0:  # Teste de Dengue
        obs_cases.loc[index, "testd"] = result
        if result == POSITIVE:
            obs_cases.loc[index, "agent_diagnosis"] = DENGUE
        elif result == NEGATIVE:
            if obs_cases.loc[index, "agent_diagnosis"] == DENGUE:
                obs_cases.loc[index, "agent_diagnosis"] = CHIK
        # INCONCLUSIVE não altera agent_diagnosis

    elif action == 1:  # Teste de Chik
        obs_cases.loc[index, "testc"] = result
        if result == POSITIVE:
            obs_cases.loc[index, "agent_diagnosis"] = CHIK
        elif result == NEGATIVE:
            if obs_cases.loc[index, "agent_diagnosis"] == CHIK:
                obs_cases.loc[index, "agent_diagnosis"] = DENGUE
        # INCONCLUSIVE não altera agent_diagnosis

    elif action == 2:  # Epi confirm
        obs_cases.loc[index, "epiconf"] = result
        # não altera agent_diagnosis
