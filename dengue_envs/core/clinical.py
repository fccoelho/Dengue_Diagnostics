"""Modelo clínico: incerteza diagnóstica, testes de laboratório e atualização de status.

Extração fiel da lógica de `DengueDiagnosticsEnv` (métodos
`_apply_clinical_uncertainty`, `_dengue_lab_test`, `_chik_lab_test`,
`_update_case_status`) para funções/métodos reutilizáveis e testáveis.
"""
from __future__ import annotations

from typing import Dict, Tuple

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

# --- Qualidade clínica ("o médico") -----------------------------------------
# Sensibilidade e especificidade do diagnóstico CLÍNICO (não do laboratório),
# tratando dengue como a classe positiva:
#   sensibilidade = P(clínico diz dengue | é dengue)
#   especificidade = P(clínico diz chik   | é chik)
#
# Cada "médico" tem sua própria competência, amostrada de uma distribuição
# Beta (suporte em [0,1], adequada para probabilidades). Os três níveis abaixo
# fixam a MÉDIA da Beta; a concentração controla a dispersão entre médicos.
CLINICAL_LEVELS: Dict[str, Dict[str, float]] = {
    "low": {"sensitivity": 0.60, "specificity": 0.60},
    "medium": {"sensitivity": 0.75, "specificity": 0.75},
    "high": {"sensitivity": 0.90, "specificity": 0.90},
}

# Concentração padrão (κ) da Beta. Maior κ => médicos mais parecidos entre si.
# Desvio-padrão ≈ sqrt(μ(1-μ)/(κ+1)); com μ=0.75 e κ=20 => ±0.09.
DEFAULT_CONCENTRATION = 20.0

# --- Desempenho do exame laboratorial ---------------------------------------
# Padrões representativos de RT-PCR para arbovírus (o "padrão-ouro" adotado no
# trabalho): teste molecular, de alta especificidade, com sensibilidade boa na
# fase virêmica (primeiros ~5 dias de sintomas) e taxa baixa de resultados
# indeterminados.
#
# ATENÇÃO: os valores abaixo são plausíveis para RT-PCR, mas ainda precisam ser
# ancorados em referência da literatura antes de irem para o artigo — a faixa
# reportada varia com o kit, o dia de coleta e o sorotipo. Todos são
# configuráveis por YAML (`lab_sensitivity`, `lab_specificity`,
# `lab_inconclusive_prob`), então trocar por valores citáveis não exige código.
#
# Antes usávamos 0,9 / 0,9 com 10% de inconclusivos, o que era pessimista para
# PCR e limitava o teto de acurácia do ambiente a ~88%.
PCR_SENSITIVITY = 0.95
PCR_SPECIFICITY = 0.98
PCR_INCONCLUSIVE_PROB = 0.02


def beta_params_from_mean(mean: float, concentration: float) -> Tuple[float, float]:
    """Converte (média, concentração) nos parâmetros ``(a, b)`` de uma Beta.

    Parametrização por média: ``a = μκ``, ``b = (1-μ)κ``, de modo que
    ``E[X] = μ`` e a dispersão cai conforme ``κ`` cresce. É a forma pedida para
    combinar "aleatoriedade entre médicos" com "níveis médios" controlados.
    """
    if not 0.0 < mean < 1.0:
        raise ValueError(f"média da Beta deve estar em (0,1); recebido {mean}")
    if concentration <= 0:
        raise ValueError(f"concentração deve ser > 0; recebido {concentration}")
    return mean * concentration, (1.0 - mean) * concentration


class ClinicalQualitySampler:
    """Amostra a competência (sensibilidade, especificidade) de um médico.

    Combina o pedido do orientador: uma Beta para a sensibilidade e outra para
    a especificidade, com três níveis médios (``low``/``medium``/``high``).

    Exemplos de configuração (YAML)::

        clinical_quality: medium                       # nível nomeado
        clinical_quality: {sensitivity: 0.8, specificity: 0.65}   # médias explícitas
        clinical_quality: {level: high, concentration: 50}        # nível + dispersão
    """

    def __init__(self, spec, concentration: float = DEFAULT_CONCENTRATION):
        if isinstance(spec, str):
            level = spec.lower()
            if level not in CLINICAL_LEVELS:
                raise ValueError(
                    f"Nível clínico desconhecido: {spec!r}. "
                    f"Disponíveis: {sorted(CLINICAL_LEVELS)}"
                )
            means = dict(CLINICAL_LEVELS[level])
            self.level = level
        elif isinstance(spec, dict):
            cfg = dict(spec)
            self.level = cfg.pop("level", None)
            concentration = float(cfg.pop("concentration", concentration))
            means = dict(CLINICAL_LEVELS[self.level]) if self.level else {}
            for key in ("sensitivity", "specificity"):
                if key in cfg:
                    means[key] = float(cfg[key])
            if not means:
                raise ValueError(
                    "clinical_quality precisa de 'level' e/ou "
                    "'sensitivity'/'specificity'."
                )
            means.setdefault("sensitivity", CLINICAL_LEVELS["medium"]["sensitivity"])
            means.setdefault("specificity", CLINICAL_LEVELS["medium"]["specificity"])
        else:
            raise TypeError(
                "clinical_quality deve ser str (nível) ou dict; "
                f"recebido {type(spec).__name__}"
            )

        self.mean_sensitivity = float(means["sensitivity"])
        self.mean_specificity = float(means["specificity"])
        self.concentration = float(concentration)
        self._sens_ab = beta_params_from_mean(self.mean_sensitivity, self.concentration)
        self._spec_ab = beta_params_from_mean(self.mean_specificity, self.concentration)

    def sample(self, rng) -> Tuple[float, float]:
        """Sorteia ``(sensibilidade, especificidade)`` de um médico."""
        sens = float(rng.beta(*self._sens_ab))
        spec = float(rng.beta(*self._spec_ab))
        return sens, spec

    def __repr__(self) -> str:  # pragma: no cover - only for logs
        return (
            f"ClinicalQualitySampler(level={self.level!r}, "
            f"mean_sens={self.mean_sensitivity:.2f}, "
            f"mean_spec={self.mean_specificity:.2f}, κ={self.concentration:g})"
        )


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
        sensitivity: float = PCR_SENSITIVITY,
        specificity: float = PCR_SPECIFICITY,
        inconclusive_prob: float = PCR_INCONCLUSIVE_PROB,
        clinical_sensitivity: float = None,
        other_recognition_prob: float = 0.15,
        arbovirus_confusion: float = 0.5,
    ):
        self.clinical_specificity = clinical_specificity
        # Sensibilidade clínica separada (P(diz dengue | é dengue)). Se não for
        # informada, cai no comportamento legado (mesma taxa nos dois sentidos).
        self.clinical_sensitivity = (
            clinical_specificity if clinical_sensitivity is None else clinical_sensitivity
        )
        self.other_prob = other_prob
        # Chance de o médico reconhecer, ainda na triagem, que um caso não é
        # arbovirose. Baixa por construção: se fosse alta, esses casos nem
        # entrariam na fila de suspeitos.
        self.other_recognition_prob = other_recognition_prob
        # Entre os não reconhecidos, como se repartem os rótulos dengue/chik.
        self.arbovirus_confusion = arbovirus_confusion
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

            if true_disease == OTHER:
                # Caso que NÃO é arbovirose, mas entrou na vigilância pelo mesmo
                # critério clínico (febre + sintomas inespecíficos). Na triagem
                # ele se parece com arbovirose: o médico o rotula como dengue ou
                # chik. Só o laboratório revela que não é nenhuma das duas — que
                # é justamente o que dá sentido à ação de descartar.
                # `arbovirus_confusion` reparte esses casos entre os dois
                # rótulos; `1 - other_recognition_prob` é a chance de o médico
                # reconhecer, já na triagem, que não é arbovirose.
                if rng.uniform() < self.other_recognition_prob:
                    obs.at[idx, "disease"] = OTHER
                elif rng.uniform() < self.arbovirus_confusion:
                    obs.at[idx, "disease"] = DENGUE
                else:
                    obs.at[idx, "disease"] = CHIK
                continue

            if rng.uniform() < self.other_prob:
                obs.at[idx, "disease"] = OTHER
                continue
            if true_disease == DENGUE:
                # Falso negativo de dengue: 1 - sensibilidade clínica.
                if rng.uniform() > self.clinical_sensitivity:  # vira chik
                    obs.at[idx, "disease"] = CHIK
            elif true_disease == CHIK:
                # Falso positivo de dengue: 1 - especificidade clínica.
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
            # Dengue descartada. Se chikungunya JÁ foi descartada por exame, o
            # caso não é arbovirose; caso contrário, chik é o palpite restante —
            # mas apenas tentativo, porque também pode ser outra etiologia.
            if int(obs_cases.loc[index, "testc"]) == NEGATIVE:
                obs_cases.loc[index, "agent_diagnosis"] = OTHER
            elif obs_cases.loc[index, "agent_diagnosis"] == DENGUE:
                obs_cases.loc[index, "agent_diagnosis"] = CHIK
        # INCONCLUSIVE não altera agent_diagnosis

    elif action == 1:  # Teste de Chik
        obs_cases.loc[index, "testc"] = result
        if result == POSITIVE:
            obs_cases.loc[index, "agent_diagnosis"] = CHIK
        elif result == NEGATIVE:
            # Simétrico ao teste de dengue (ver acima).
            if int(obs_cases.loc[index, "testd"]) == NEGATIVE:
                obs_cases.loc[index, "agent_diagnosis"] = OTHER
            elif obs_cases.loc[index, "agent_diagnosis"] == CHIK:
                obs_cases.loc[index, "agent_diagnosis"] = DENGUE
        # INCONCLUSIVE não altera agent_diagnosis

    elif action == 2:  # Epi confirm
        obs_cases.loc[index, "epiconf"] = result
        # não altera agent_diagnosis
