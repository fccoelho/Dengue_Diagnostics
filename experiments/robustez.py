"""Experimentos de robustez do agente treinado, para o notebook de demonstração.

Todos respondem à mesma pergunta por ângulos diferentes: **o resultado do
experimento v4 é uma propriedade da política aprendida, ou um acidente das
condições em que ela foi medida?**

Cada experimento devolve um `DataFrame` de uma linha por episódio e é cacheado
em `results/demo/<nome>.csv`. O cache existe por um motivo prático: uma varredura
completa custa ~1h de CPU, e o notebook precisa poder ser reexecutado na frente
de alguém. Apague o arquivo para recalcular.

Nada aqui altera o agente: os checkpoints são carregados como estão, e o que
muda é o AMBIENTE (qualidade do médico, custo do exame, distribuição espacial)
ou a OBSERVAÇÃO que lhe é entregue (ablações de entrada).
"""
from __future__ import annotations

import copy
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import yaml

_RAIZ = Path(__file__).resolve().parents[1]
if str(_RAIZ) not in sys.path:
    sys.path.insert(0, str(_RAIZ))

from dengue_envs.wrappers import make_env
from experiments.evaluate import AGENT_REGISTRY, _make_runner

CACHE = _RAIZ / "results" / "demo"
CONFIGS = _RAIZ / "experiments" / "configs" / "env"

# Seeds do benchmark oficial — reservadas para a comparação principal.
SEEDS_BENCHMARK = (100, 150, 200, 250, 300, 350, 400, 450, 500, 550)
# Seeds NUNCA usadas em avaliação nenhuma até aqui (nem treino, nem benchmark).
SEEDS_INEDITAS = tuple(range(9001, 9021))

CHECKPOINTS = {
    "ppo A": "results/ppo_v4a_seir_s{seed}/policy_final.pth",
    "ppo B": "results/ppo_v4b_credito_s{seed}/policy_final.pth",
    "ppo C": "results/ppo_v4c_credito_sem_tempo_s{seed}/policy_final.pth",
}

# Blocos da observação, para as ablações de entrada. O contexto tem 16 posições:
#   0-1   evidência sobre o médico (taxa de acerto observada, força da evidência)
#   2-4   diagnóstico clínico do caso (one-hot dengue/chik/outro)
#   5-8   resultado do exame de dengue (não feito / neg / pos / inconclusivo)
#   9-12  resultado do exame de chikungunya
#   13    confirmação epidemiológica já feita
#   14-15 densidade local de confirmados (dengue, chik)
BLOCOS_OBS: Dict[str, Dict] = {
    "mapa":               {"chave": "map"},
    "posição do caso":    {"chave": "case_coords"},
    "evidência do médico": {"chave": "context", "fatias": [(0, 2)]},
    "diagnóstico clínico": {"chave": "context", "fatias": [(2, 5)]},
    "resultado dos exames": {"chave": "context", "fatias": [(5, 13)]},
    "vizinhança confirmada": {"chave": "context", "fatias": [(13, 16)]},
}


# --------------------------------------------------------------------------
# infraestrutura
# --------------------------------------------------------------------------

def config_ambiente(nome: str = "kriging_v8", **sobrescreve) -> dict:
    """Carrega um YAML de ambiente e sobrescreve chaves de `env`.

    `sobrescreve` age dentro do bloco `env` (ex.: `test_cost=6.0`,
    `clinical_specificity=0.7`), que é onde vivem os parâmetros que variamos.
    """
    cfg = yaml.safe_load((CONFIGS / f"{nome}.yaml").read_text(encoding="utf-8"))
    cfg = copy.deepcopy(cfg)
    cfg["env"].update(sobrescreve)
    return cfg


DISPOSITIVO = "cpu"


def cria_runner(agente: str, seed_treino: int = 45, ablacao: Optional[str] = None):
    """Runner do benchmark. `ppo A/B/C` carregam o checkpoint daquela seed.

    O dispositivo é explícito (CPU por padrão) e não herdado da disponibilidade
    de CUDA: estas avaliações rodam ao lado de treinos que ocupam a GPU, e o
    laço é limitado pelo ambiente de qualquer forma (medido: 48 passos/s nos
    dois dispositivos).
    """
    if agente.startswith("ppo "):
        from agents.ppo.agent import PPOAgentRunner

        caminho = _RAIZ / CHECKPOINTS[agente].format(seed=seed_treino)
        if not caminho.exists():
            raise FileNotFoundError(caminho)
        runner = PPOAgentRunner(policy_path=str(caminho), device=DISPOSITIVO)
    elif agente == "dqn":
        from agents.deepq.agent import DQNAgentRunner

        runner = DQNAgentRunner(
            policy_path=str(_RAIZ / "results/dqn_v9_b6k_s44/policy_final.pth"),
            device=DISPOSITIVO,
        )
    else:
        if agente not in AGENT_REGISTRY:
            raise KeyError(f"agente desconhecido: {agente}")
        runner = _make_runner(agente, {})
    if ablacao:
        runner = _com_ablacao(runner, ablacao)
    return runner


def _com_ablacao(runner, bloco: str):
    """Zera um bloco da observação ANTES de o agente decidir.

    Mede de que entrada a política depende, sem retreinar: se zerar um bloco
    não muda o comportamento, aquele bloco não estava sendo usado.
    """
    spec = BLOCOS_OBS[bloco]
    from agents.deepq.agent import observation_from_env

    agente_interno = runner

    class _Ablado:
        name = f"{getattr(runner, 'name', 'agente')} (sem {bloco})"

        def choose_action(self, env) -> int:
            obs = observation_from_env(env)
            chave = spec["chave"]
            if chave in obs:
                valor = np.array(obs[chave], copy=True)
                if "fatias" in spec:
                    for a, b in spec["fatias"]:
                        valor[a:b] = 0.0
                else:
                    valor[...] = 0.0
                obs = dict(obs)
                obs[chave] = valor
            return agente_interno._ensure_agent(env).choose_action(obs)

    if not hasattr(runner, "_ensure_agent"):
        raise TypeError("ablação só se aplica a agentes com rede (ppo/dqn)")
    return _Ablado()


class TestaFracaoAleatoria:
    """Testa uma fração `p` dos casos, sorteada — sem olhar para o caso.

    É o controle que separa as duas competências possíveis do agente: gastar a
    quantidade certa de exames, e gastá-los nos casos certos. Esta política
    gasta a mesma quantidade, distribuída ao acaso. A diferença entre as duas é
    o valor da SELEÇÃO.

    O fluxo é o do `testonce` nos casos sorteados (pede exame, conclui quando o
    laudo volta) e o do `confirmall` nos demais (conclui pelo palpite clínico).
    """

    def __init__(self, p: float, semente: int = 0):
        self.p = float(p)
        self.name = f"aleatório {p:.0%}"
        self._rng = np.random.default_rng(semente)
        self._sorteados: Dict[int, bool] = {}

    def choose_action(self, env) -> int:
        base = env.unwrapped
        case_id = env.current_case[0]
        if case_id not in base.obs_cases.index:
            return 4
        linha = base.obs_cases.loc[case_id]
        if case_id not in self._sorteados:
            self._sorteados[case_id] = bool(self._rng.random() < self.p)
        if self._sorteados[case_id] and int(linha["testd"]) == 0:
            return 0
        return 4 + int(linha["agent_diagnosis"])


def orcamento_aleatorio(fracoes=(0.25, 0.50, 0.70, 0.85),
                        seeds: Sequence[int] = SEEDS_INEDITAS[:5]) -> pd.DataFrame:
    """Mesma quantidade de exames do agente, distribuída ao acaso.

    Se o agente ganhar de uma alocação aleatória de mesmo tamanho, o ganho está
    em ESCOLHER os casos. Se empatar, estaria apenas calibrando o volume.
    """
    cfg = config_ambiente("kriging_v8")
    partes = []
    for p in fracoes:
        env = make_env(cfg)
        runner = TestaFracaoAleatoria(p)
        linhas = []
        for s in seeds:
            runner._sorteados.clear()
            linha = {"agente": f"aleatório {p:.0%}", "fracao_alvo": p,
                     "seed": int(s), "seed_treino": -1}
            linha.update(roda_episodio(runner, env, s))
            linhas.append(linha)
        env.close()
        partes.append(pd.DataFrame(linhas))
    partes.append(avalia("ppo C", cfg, seeds, extras={"fracao_alvo": np.nan}))
    # Referências medidas nas MESMAS seeds, para a comparação ser interna.
    for fixa in ("testonce", "confirmall"):
        partes.append(avalia(fixa, cfg, seeds, extras={"fracao_alvo": np.nan}))
    return pd.concat(partes, ignore_index=True)


def roda_episodio(runner, env, seed: int) -> dict:
    """Um episódio completo; devolve métricas + como o agente se comportou."""
    env.reset(seed=int(seed))
    env.action_space.seed(int(seed))
    terminou = truncou = False
    while not (terminou or truncou):
        _, _, terminou, truncou, _ = env.step(runner.choose_action(env))

    u = env.unwrapped
    casos = u.obs_cases
    real = u.real_cases.loc[casos.index, "disease"].to_numpy()
    acertou = casos["agent_diagnosis"].to_numpy() == real
    investigado = (casos["testd"].to_numpy() != 0) | (casos["testc"].to_numpy() != 0)
    m = u.get_episode_metrics()
    return {
        "recompensa": float(m["Recompensa Total"]),
        "acuracia": float(m["Acurácia Multiclasse"]),
        "exames": float(m["Testes Realizados"]),
        "casos": int(len(casos)),
        "fracao_investigada": float(investigado.mean()),
        "acerto_investigados": float(acertou[investigado].mean()) if investigado.any() else np.nan,
        "acerto_nao_investigados": float(acertou[~investigado].mean()) if (~investigado).any() else np.nan,
        "especificidade_medico": float(u.clinical_specificity),
    }


def avalia(agente: str, cfg: dict, seeds: Sequence[int], *,
           seed_treino: int = 45, ablacao: Optional[str] = None,
           extras: Optional[dict] = None) -> pd.DataFrame:
    """Roda `agente` em `seeds` sob a configuração `cfg`."""
    env = make_env(cfg)
    runner = cria_runner(agente, seed_treino=seed_treino, ablacao=ablacao)
    linhas = []
    for s in seeds:
        linha = {"agente": agente, "seed": int(s), "seed_treino": seed_treino}
        linha.update(extras or {})
        linha.update(roda_episodio(runner, env, s))
        linhas.append(linha)
    env.close()
    return pd.DataFrame(linhas)


def cache(nome: str, calcula: Callable[[], pd.DataFrame], *, recalcula: bool = False) -> pd.DataFrame:
    """Lê `results/demo/<nome>.csv` ou calcula e grava."""
    CACHE.mkdir(parents=True, exist_ok=True)
    caminho = CACHE / f"{nome}.csv"
    if caminho.exists() and not recalcula:
        return pd.read_csv(caminho)
    t0 = time.time()
    df = calcula()
    df.to_csv(caminho, index=False)
    print(f"[{nome}] {len(df)} episódios em {time.time() - t0:.0f}s -> {caminho}")
    return df


# --------------------------------------------------------------------------
# experimentos
# --------------------------------------------------------------------------

def seeds_ineditas(seeds_treino: Sequence[int] = (45, 46, 47),
                   seeds: Sequence[int] = SEEDS_INEDITAS[:12]) -> pd.DataFrame:
    """Surtos nunca vistos: nem no treino, nem no benchmark oficial.

    O benchmark usa sempre as mesmas 10 seeds. Isso é bom para comparabilidade e
    ruim para uma pergunta: e se as conclusões valessem só para aqueles 10
    surtos? Aqui as seeds são inéditas (9001+), e cada checkpoint é avaliado em
    todas elas.
    """
    cfg = config_ambiente("kriging_v8")
    partes = []
    for agente in ("ppo C", "ppo A"):
        for st in seeds_treino:
            partes.append(avalia(agente, cfg, seeds, seed_treino=st))
    for fixa in ("testonce", "testtwice", "clinical", "confirmall"):
        partes.append(avalia(fixa, cfg, seeds))
    return pd.concat(partes, ignore_index=True)


def qualidade_do_medico(valores=(0.50, 0.60, 0.70, 0.80, 0.90),
                        seeds: Sequence[int] = SEEDS_INEDITAS[:5]) -> pd.DataFrame:
    """Varre a especificidade do médico, que normalmente é sorteada por episódio.

    Fixando-a, a pergunta fica direta: o agente aloca exame conforme o valor da
    informação? A política ótima varia com esse parâmetro — com médico perfeito
    não há o que investigar; com médico aleatório, investigar tudo compensa.
    """
    partes = []
    for espec in valores:
        cfg = config_ambiente("kriging_v8", clinical_specificity=float(espec))
        for agente in ("ppo C", "ppo A"):
            partes.append(avalia(agente, cfg, seeds, extras={"especificidade": espec}))
        for fixa in ("testonce", "clinical"):
            partes.append(avalia(fixa, cfg, seeds, extras={"especificidade": espec}))
    return pd.concat(partes, ignore_index=True)


def custo_do_exame(custos=(2.0, 4.0, 6.0, 8.0),
                   seeds: Sequence[int] = SEEDS_INEDITAS[:5]) -> pd.DataFrame:
    """Muda a economia DEPOIS do treino: o agente treinou com exame a 4,0.

    Teste de generalização econômica. O agente não observa o custo — ele o
    aprendeu implicitamente. Se a política continuar boa quando o custo muda,
    é porque aprendeu a ordenar casos por valor de informação, não a executar
    uma quantidade fixa de exames. Se degradar, mede-se o quanto.
    """
    partes = []
    for custo in custos:
        cfg = config_ambiente("kriging_v8", test_cost=float(custo))
        for agente in ("ppo C",):
            partes.append(avalia(agente, cfg, seeds, extras={"custo": custo}))
        for fixa in ("testonce", "testtwice", "clinical"):
            partes.append(avalia(fixa, cfg, seeds, extras={"custo": custo}))
    return pd.concat(partes, ignore_index=True)


def atraso_do_laudo(atrasos=(0, 2, 5, 10, 15),
                    seeds: Sequence[int] = SEEDS_INEDITAS[:5]) -> pd.DataFrame:
    """Varre o tempo até o laudo voltar — o parâmetro que cria o problema.

    É ele que separa a decisão de investigar da sua consequência, e é por causa
    dele que o crédito precisa atravessar as decisões de outros pacientes. Com
    atraso zero o problema de atribuição quase desaparece; com 15 dias, cada
    exame atravessa ainda mais casos alheios.

    A política foi treinada com 5 dias e não observa esse parâmetro.
    """
    partes = []
    for atraso in atrasos:
        cfg = config_ambiente("kriging_v8", lab_delay_days=int(atraso))
        for agente in ("ppo C", "ppo A"):
            partes.append(avalia(agente, cfg, seeds, extras={"atraso": atraso}))
        for fixa in ("testonce", "clinical"):
            partes.append(avalia(fixa, cfg, seeds, extras={"atraso": atraso}))
    return pd.concat(partes, ignore_index=True)


def ablacao_de_entradas(seeds: Sequence[int] = SEEDS_INEDITAS[:6]) -> pd.DataFrame:
    """Zera um bloco da observação por vez, sem retreinar.

    É atribuição de importância por intervenção: a política é a mesma, só deixa
    de enxergar aquele bloco. Duas leituras possíveis — o bloco não era usado
    (desempenho intacto) ou era essencial (desempenho desaba).
    """
    cfg = config_ambiente("kriging_v8")
    partes = [avalia("ppo C", cfg, seeds, extras={"ablacao": "nenhuma"})]
    for bloco in BLOCOS_OBS:
        partes.append(avalia("ppo C", cfg, seeds, ablacao=bloco,
                             extras={"ablacao": bloco}))
    return pd.concat(partes, ignore_index=True)


def transferencia_espacial(seeds: Sequence[int] = SEEDS_INEDITAS[:6]) -> pd.DataFrame:
    """O agente treinado no mapa real, avaliado na distribuição sintética.

    As duas distribuições pedem políticas fixas OPOSTAS (testonce é ótimo no
    kriging, testtwice no sintético) e a posição prediz a doença em 50,7% contra
    93,7%. É a troca de ambiente mais severa que o projeto tem.
    """
    partes = []
    for nome_cfg, rotulo in [("kriging_v8", "kriging (treino)"),
                             ("synthetic_v8", "sintético (inédito)")]:
        cfg = config_ambiente(nome_cfg)
        for agente in ("ppo C", "ppo A"):
            partes.append(avalia(agente, cfg, seeds, extras={"distribuicao": rotulo}))
        for fixa in ("testonce", "testtwice", "clinical"):
            partes.append(avalia(fixa, cfg, seeds, extras={"distribuicao": rotulo}))
    return pd.concat(partes, ignore_index=True)


# Superfícies do Rio por ano (`build_kriging_surfaces --years-dengue/--years-chik`).
# A chikungunya quase não circulou em 2015 (70 notificações), então o cenário
# "2015" troca só a dengue; a chik é a de 2016 nos dois.
SUPERFICIES_POR_ANO = {
    "dengue 2015 + chik 2016": "results/kriging/rio_d2015_c2016_kriging_surfaces.npz",
    "dengue 2016 + chik 2016": "results/kriging/rio_2016_kriging_surfaces.npz",
}


def _superficie_por_ano(rotulo: str, seeds_treino: Sequence[int] = (45, 46, 47, 48),
                        seeds: Sequence[int] = SEEDS_BENCHMARK) -> pd.DataFrame:
    """Os agentes do v4, sem retreino, sobre a superfície de um único ano.

    A superfície de referência junta 2015 e 2016, então nenhum dos dois anos é
    inédito para o agente: isto mede CONSISTÊNCIA entre anos, não
    generalização. Medido antes de rodar: a posição sozinha acerta a doença em
    53-54% nas três superfícies (Bayes, prior igual), e a dengue de 2015 tem
    correlação 0,90 com a de 2016. As seeds são as do benchmark oficial, para
    que o bootstrap pareie estes episódios com os da referência.
    """
    cfg = config_ambiente("kriging_v8", surfaces_path=str(_RAIZ / SUPERFICIES_POR_ANO[rotulo]))
    extras = {"superficie": rotulo}
    partes = []
    for agente in ("ppo C", "ppo A"):
        for st in seeds_treino:
            partes.append(avalia(agente, cfg, seeds, seed_treino=st, extras=extras))
    for fixa in ("testonce", "testtwice", "clinical", "confirmall"):
        partes.append(avalia(fixa, cfg, seeds, extras=extras))
    return pd.concat(partes, ignore_index=True)


def dengue_2015(**kw) -> pd.DataFrame:
    return _superficie_por_ano("dengue 2015 + chik 2016", **kw)


def rio_2016(**kw) -> pd.DataFrame:
    return _superficie_por_ano("dengue 2016 + chik 2016", **kw)


# Varredura de dificuldade espacial (`transform_surfaces`): τ -> acerto da
# posição sozinha (Bayes, prior igual), medido na superfície de referência.
TEMPERATURAS = {0.0: 0.500, 0.5: 0.522, 1.0: 0.544, 2.0: 0.586,
                4.0: 0.651, 8.0: 0.740, 16.0: 0.852, 32.0: 0.939}


def temperatura_espacial(tau: float, seeds_treino: Sequence[int] = (45, 46, 47, 48),
                         seeds: Sequence[int] = SEEDS_BENCHMARK) -> pd.DataFrame:
    """Os agentes do v4, sem retreino, com a geografia mais ou menos informativa.

    τ = 1 é o ambiente de treino; τ = 0 apaga a geografia; τ = 32 separa as
    doenças tanto quanto o sintético (0,94). Diferente da `transferencia_espacial`,
    que troca de gerador inteiro, aqui só a concentração das MESMAS superfícies
    muda: os focos continuam onde estão.
    """
    cfg = config_ambiente("kriging_v8", surface_temperature=float(tau))
    extras = {"temperatura": float(tau), "acerto_posicao": TEMPERATURAS.get(float(tau))}
    partes = []
    for agente in ("ppo C", "ppo A"):
        for st in seeds_treino:
            partes.append(avalia(agente, cfg, seeds, seed_treino=st, extras=extras))
    for fixa in ("testonce", "testtwice", "clinical"):
        partes.append(avalia(fixa, cfg, seeds, extras=extras))
    return pd.concat(partes, ignore_index=True)


EXPERIMENTOS = {
    "seeds_ineditas": seeds_ineditas,
    "qualidade_do_medico": qualidade_do_medico,
    "custo_do_exame": custo_do_exame,
    "atraso_do_laudo": atraso_do_laudo,
    "ablacao_de_entradas": ablacao_de_entradas,
    "orcamento_aleatorio": orcamento_aleatorio,
    "transferencia_espacial": transferencia_espacial,
    "dengue_2015": dengue_2015,
    "rio_2016": rio_2016,
    **{f"temperatura_{t:g}": (lambda t=t: temperatura_espacial(t)) for t in TEMPERATURAS},
}


def main(nomes: Optional[List[str]] = None) -> None:
    """Pré-calcula os caches (uso: `python -m experiments.robustez [nome ...]`)."""
    for nome in (nomes or list(EXPERIMENTOS)):
        cache(nome, EXPERIMENTOS[nome])


if __name__ == "__main__":
    main(sys.argv[1:] or None)
