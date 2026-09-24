"""Baselines sequenciais: o segundo exame só quando o primeiro não resolve."""
from types import SimpleNamespace

import pandas as pd
import pytest

from agents.sequential.agent import ClinicalSequentialAgentRunner, SequentialAgentRunner

T_DENGUE, T_CHIK, CONCLUIR = 0, 1, 4
NAO, NEG, POS, INC = 0, 1, 2, 3


def _env(testd, testc, diagnostico):
    casos = pd.DataFrame([{"testd": testd, "testc": testc, "agent_diagnosis": diagnostico}])
    return SimpleNamespace(unwrapped=SimpleNamespace(obs_cases=casos), current_case=(0,))


@pytest.mark.parametrize("runner", [SequentialAgentRunner(), ClinicalSequentialAgentRunner()])
def test_positivo_encerra_sem_segundo_exame(runner):
    assert runner.choose_action(_env(POS, NAO, 0)) == CONCLUIR + 0
    assert runner.choose_action(_env(NAO, POS, 1)) == CONCLUIR + 1


@pytest.mark.parametrize("runner", [SequentialAgentRunner(), ClinicalSequentialAgentRunner()])
def test_negativo_ou_inconclusivo_pede_o_outro(runner):
    assert runner.choose_action(_env(NEG, NAO, 1)) == T_CHIK
    assert runner.choose_action(_env(INC, NAO, 0)) == T_CHIK
    assert runner.choose_action(_env(NAO, NEG, 0)) == T_DENGUE


@pytest.mark.parametrize("runner", [SequentialAgentRunner(), ClinicalSequentialAgentRunner()])
def test_dois_laudos_concluem_com_o_diagnostico_corrente(runner):
    assert runner.choose_action(_env(NEG, NEG, 2)) == CONCLUIR + 2


def test_ordem_do_primeiro_exame():
    assert SequentialAgentRunner().choose_action(_env(NAO, NAO, 1)) == T_DENGUE
    clinico = ClinicalSequentialAgentRunner()
    assert clinico.choose_action(_env(NAO, NAO, 1)) == T_CHIK    # médico suspeita de chik
    assert clinico.choose_action(_env(NAO, NAO, 0)) == T_DENGUE
    assert clinico.choose_action(_env(NAO, NAO, 2)) == T_DENGUE  # "outro": começa pela dengue
