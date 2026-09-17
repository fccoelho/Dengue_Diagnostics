"""Ferramentas de robustez: o que elas prometem medir, elas medem.

O risco aqui não é o código quebrar — é ele rodar e produzir um número errado
em silêncio. Uma ablação que zera a fatia errada, ou uma política de controle
que não testa a fração que diz testar, geram gráficos plausíveis e conclusões
falsas. Estes testes travam exatamente isso.
"""
from __future__ import annotations

import numpy as np
import pytest

from experiments import robustez as R


def _env():
    from dengue_envs.wrappers import make_env

    return make_env(R.config_ambiente("kriging_v8"))


# --- configuração -----------------------------------------------------------

def test_sobrescreve_sem_tocar_no_arquivo():
    base = R.config_ambiente("kriging_v8")
    mudado = R.config_ambiente("kriging_v8", test_cost=9.0, clinical_specificity=0.7)
    assert mudado["env"]["test_cost"] == 9.0
    assert mudado["env"]["clinical_specificity"] == 0.7
    # O original permanece intocado: as varreduras rodam em sequência no mesmo
    # processo, e um dicionário compartilhado vazaria de um nível para o outro.
    assert base["env"]["test_cost"] == R.config_ambiente("kriging_v8")["env"]["test_cost"]
    assert isinstance(base["env"]["clinical_specificity"], list)


def test_especificidade_fixa_chega_ao_ambiente():
    from dengue_envs.wrappers import make_env

    env = make_env(R.config_ambiente("kriging_v8", clinical_specificity=0.73))
    for seed in (1, 2, 3):
        env.reset(seed=seed)
        assert env.unwrapped.clinical_specificity == pytest.approx(0.73)
    env.close()


# --- ablação de entradas ----------------------------------------------------

def test_ablacao_zera_exatamente_o_bloco_declarado():
    """A fatia zerada é a que o bloco nomeia — nem uma posição a mais."""
    vistas = []

    class _Espiao:
        def _ensure_agent(self, env):
            return self

        def choose_action(self, obs):
            vistas.append({k: np.array(v, copy=True) for k, v in obs.items()})
            return 3

    env = _env()
    ablado = R._com_ablacao(_Espiao(), "resultado dos exames")
    env.reset(seed=9001)
    ablado.choose_action(env)

    ctx = vistas[-1]["context"]
    assert np.all(ctx[5:13] == 0.0), "a fatia dos exames deveria estar zerada"
    # As posições vizinhas continuam de pé — senão a ablação mediria outra coisa.
    assert vistas[-1]["map"].any(), "o mapa não deveria ter sido tocado"
    env.close()


def test_ablacao_do_mapa_nao_mexe_no_contexto():
    vistas = []

    class _Espiao:
        def _ensure_agent(self, env):
            return self

        def choose_action(self, obs):
            vistas.append({k: np.array(v, copy=True) for k, v in obs.items()})
            return 3

    env = _env()
    ablado = R._com_ablacao(_Espiao(), "mapa")
    env.reset(seed=9001)
    ablado.choose_action(env)
    assert not vistas[-1]["map"].any()
    assert vistas[-1]["context"].any()
    env.close()


def test_ablacao_recusa_agente_sem_rede():
    """Políticas fixas não olham a observação; ablar uma delas seria um no-op
    silencioso, e o gráfico mostraria 'nenhum efeito' por motivo errado."""
    with pytest.raises(TypeError, match="ppo|dqn|rede"):
        R._com_ablacao(R.cria_runner("testonce"), "mapa")


def test_blocos_cobrem_o_contexto_sem_sobreposicao():
    fatias = [f for spec in R.BLOCOS_OBS.values() for f in spec.get("fatias", [])]
    cobertas = sorted(i for a, b in fatias for i in range(a, b))
    assert cobertas == list(range(16)), "os blocos devem particionar o contexto"


# --- política de controle ---------------------------------------------------

def test_fracao_aleatoria_testa_aproximadamente_a_fracao_pedida():
    env = _env()
    runner = R.TestaFracaoAleatoria(0.5, semente=0)
    medido = R.roda_episodio(runner, env, 9001)
    env.close()
    assert medido["fracao_investigada"] == pytest.approx(0.5, abs=0.1)


def test_fracao_zero_nunca_testa_e_fracao_um_testa_tudo():
    env = _env()
    nenhum = R.roda_episodio(R.TestaFracaoAleatoria(0.0), env, 9002)
    todos = R.roda_episodio(R.TestaFracaoAleatoria(1.0), env, 9002)
    env.close()
    assert nenhum["exames"] == 0
    assert todos["fracao_investigada"] == pytest.approx(1.0, abs=0.01)
    # Mesmo surto, mesma quantidade de casos: a diferença é só a alocação.
    assert nenhum["casos"] == todos["casos"]


def test_decisao_por_caso_e_estavel_dentro_do_episodio():
    """Um caso sorteado para testar continua sorteado quando volta com o laudo.

    Sem isso a política de controle oscilaria a cada revisita e deixaria de ser
    'testar uma fração dos casos'.
    """
    runner = R.TestaFracaoAleatoria(0.5, semente=3)
    env = _env()
    env.reset(seed=9001)
    for _ in range(40):
        env.step(runner.choose_action(env))
    decididos = dict(runner._sorteados)
    for _ in range(40):
        env.step(runner.choose_action(env))
    env.close()
    for caso, escolha in decididos.items():
        assert runner._sorteados[caso] == escolha


# --- cache ------------------------------------------------------------------

def test_cache_reaproveita_e_recalcula_quando_pedido(tmp_path, monkeypatch):
    import pandas as pd

    monkeypatch.setattr(R, "CACHE", tmp_path)
    chamadas = []

    def calcula():
        chamadas.append(1)
        return pd.DataFrame({"x": [len(chamadas)]})

    assert R.cache("exemplo", calcula)["x"][0] == 1
    assert R.cache("exemplo", calcula)["x"][0] == 1      # veio do disco
    assert len(chamadas) == 1
    assert R.cache("exemplo", calcula, recalcula=True)["x"][0] == 2
