"""Quais casos o agente testa duas vezes? Análise por caso do braço C contra a regra.

O resultado agregado (v6) é que o C treinado no Rio empata com o
`sequencial_clinico` em recompensa, com ~8% menos exames. Isto responde ONDE
estão esses exames a menos: registra cada decisão, caso a caso, nos mesmos
surtos de teste do v6 (sementes 3001-3030), e resume por caso.

Uma linha por decisão: surto, caso, dia, ação, suspeita do médico, doença
real, estado dos dois exames e da confirmação epidemiológica ANTES da ação,
e a densidade local de confirmados (as mesmas features que o agente vê).

Uso:
    python -m experiments.analise_casos --workers 3     # roda (retomável por unidade)
    python -m experiments.analise_casos --analisa       # tabelas e figura
    python -m experiments.analise_casos --limiar        # o 2º exame é sempre ótimo?
"""
from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

_RAIZ = Path(__file__).resolve().parents[1]
if str(_RAIZ) not in sys.path:
    sys.path.insert(0, str(_RAIZ))

SAIDA = _RAIZ / "results" / "analise_casos"
TREINOS = _RAIZ / "results" / "ppo_v6"
SEEDS = tuple(range(3001, 3031))
# (agente, teste): o C do Rio nas três cidades reais, o C do sintético no
# sintético (onde a posição informa) e a regra em todos.
UNIDADES = (
    [(f"rio_C_s{s}", t) for s in (45, 46, 47) for t in ("teste_rio", "teste_recife_2016", "teste_recife_2021")]
    + [(f"sintetico_C_s{s}", "teste_sintetico") for s in (45, 46, 47)]
    + [("sequencial_clinico", t) for t in ("teste_rio", "teste_recife_2016", "teste_recife_2021", "teste_sintetico")]
    # Recife corrigido (casos "outro" só na área habitada): para a conta do 2º exame.
    + [("sequencial_clinico", t) for t in ("teste_recife_2016_sup", "teste_recife_2021_sup")]
)
ACOES = {0: "exame_dengue", 1: "exame_chik", 2: "confirma_epi", 3: "espera",
         4: "conclui_dengue", 5: "conclui_chik", 6: "conclui_outro"}
DOENCA = {0: "dengue", 1: "chik", 2: "outro"}
LAUDO = {0: "nao_feito", 1: "neg", 2: "pos", 3: "inconc"}


def _destino(agente: str, teste: str) -> Path:
    return SAIDA / f"{teste}__{agente}.csv.gz"


def roda_unidade(agente: str, teste: str) -> str:
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    from experiments import robustez as R
    from experiments.evaluate import _make_runner
    from dengue_envs.wrappers import make_env

    if agente.startswith(("rio_", "sintetico_")):
        from agents.ppo.agent import PPOAgentRunner
        runner = PPOAgentRunner(policy_path=str(TREINOS / agente / "policy_final.pth"), device="cpu")
    else:
        runner = _make_runner(agente, {})
    env = make_env(R.config_ambiente(teste))
    u = env.unwrapped
    linhas = []
    for seed in SEEDS:
        env.reset(seed=seed)
        env.action_space.seed(seed)
        visto = {}
        fim = False
        while not fim:
            cid = int(env.current_case[0])
            registro = None
            # (0, 0, 0) é o sentinela de "sem caso", mas 0 também é um id válido.
            if cid in u.obs_cases.index and (cid != 0 or env.active_cases):
                row = u.obs_cases.loc[cid]
                nd, nc = u.local_confirmed_density(cid)
                registro = {
                    "seed": seed, "caso": cid, "dia": int(u.t), "rodada": visto.get(cid, 0),
                    "suspeita": int(row["disease"]), "real": int(u.real_cases.loc[cid, "disease"]),
                    "testd": int(row["testd"]), "testc": int(row["testc"]), "epi": int(row["epiconf"]),
                    "dens_d": float(nd), "dens_c": float(nc),
                }
                visto[cid] = visto.get(cid, 0) + 1
            acao = int(runner.choose_action(env))
            if registro is not None:
                registro["acao"] = acao
                linhas.append(registro)
            _, _, term, trunc, _ = env.step(acao)
            fim = term or trunc
    env.close()
    destino = _destino(agente, teste)
    destino.parent.mkdir(parents=True, exist_ok=True)
    tmp = destino.with_name(destino.name + ".tmp")
    pd.DataFrame(linhas).assign(agente=agente, teste=teste).to_csv(tmp, index=False, compression="gzip")
    tmp.replace(destino)
    return f"{agente} @ {teste}: {len(linhas)} decisões"


def roda(workers: int) -> None:
    faltam = [u for u in UNIDADES if not _destino(*u).exists()]
    print(f"{len(faltam)} unidades a rodar", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futuros = {pool.submit(roda_unidade, *u): u for u in faltam}
        for f in as_completed(futuros):
            try:
                print("OK", f.result(), flush=True)
            except Exception as e:  # noqa: BLE001
                print("FALHOU", futuros[f], repr(e), flush=True)


# --------------------------------------------------------------------------
# análise
# --------------------------------------------------------------------------

def por_caso(dec: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por caso: quantos exames, em que ordem, o que o 1º laudo disse, como terminou."""
    dec = dec.sort_values(["agente", "teste", "seed", "caso", "dia", "rodada"])
    chave = ["agente", "teste", "seed", "caso"]
    exames = dec[dec.acao.isin([0, 1])]
    primeiro = exames.groupby(chave).first()
    conclusao = dec[dec.acao >= 4].groupby(chave).last()
    g = dec.groupby(chave)
    casos = pd.DataFrame({
        "suspeita": g.suspeita.first(), "real": g.real.first(),
        "n_exames": g.acao.apply(lambda a: int(a.isin([0, 1]).sum())),
        "n_epi": g.acao.apply(lambda a: int((a == 2).sum())),
        "dia": g.dia.first(),
        "dens_d": g.dens_d.first(), "dens_c": g.dens_c.first(),
    })
    casos["primeiro_exame"] = primeiro.acao.map({0: "dengue", 1: "chik"})
    casos["primeiro_eh_suspeita"] = (primeiro.acao == primeiro.suspeita)
    # O laudo do 1º exame, lido na decisão seguinte ao seu retorno.
    casos["laudo1"] = pd.Series(pd.NA, index=casos.index, dtype="object")
    depois = dec.merge(primeiro[["acao"]].rename(columns={"acao": "a1"}), left_on=chave, right_index=True)
    depois = depois[depois.rodada > 0]
    depois = depois.assign(l1=np.where(depois.a1 == 0, depois.testd, depois.testc))
    depois = depois[depois.l1 != 0].groupby(chave).l1.first()
    casos.loc[depois.index, "laudo1"] = depois.map(LAUDO)
    casos["conclusao"] = (conclusao.acao - 4).reindex(casos.index)
    casos["acertou"] = casos.conclusao == casos.real
    return casos.reset_index()


def carrega() -> pd.DataFrame:
    partes = [pd.read_csv(f) for f in sorted(SAIDA.glob("*.csv.gz"))]
    return pd.concat(partes, ignore_index=True)


def analisa() -> dict:
    dec = carrega()
    casos = por_caso(dec)
    casos["politica"] = np.where(casos.agente == "sequencial_clinico", "regra",
                                 casos.agente.str.replace(r"_s\d+$", "", regex=True))
    casos["suspeita_n"] = casos.suspeita.map(DOENCA)
    casos["real_n"] = casos.real.map(DOENCA)
    casos.to_csv(SAIDA / "casos.csv.gz", index=False, compression="gzip")

    tabelas = {}
    # 1. Distribuição de exames por caso.
    t = (casos.groupby(["teste", "politica"]).n_exames.value_counts(normalize=True)
         .unstack(fill_value=0).round(3))
    t["media"] = casos.groupby(["teste", "politica"]).n_exames.mean().round(3)
    t["acuracia"] = casos.groupby(["teste", "politica"]).acertou.mean().round(3)
    tabelas["exames_por_caso"] = t
    # 2. Depois de um 1º laudo negativo, pede o 2º? Por suspeita e doença real.
    neg = casos[casos.laudo1 == "neg"]
    t = (neg.assign(segundo=neg.n_exames >= 2)
         .groupby(["teste", "politica", "suspeita_n", "real_n"])
         .agg(casos=("segundo", "size"), p_segundo=("segundo", "mean"), acerto=("acertou", "mean")).round(3))
    tabelas["apos_negativo"] = t
    # 3. Primeiro exame = suspeita do médico?
    t = (casos[casos.n_exames >= 1].groupby(["teste", "politica", "suspeita_n"])
         .agg(casos=("primeiro_eh_suspeita", "size"), p_segue_medico=("primeiro_eh_suspeita", "mean"),
              p_zero=("n_exames", lambda s: 0.0))).round(3)
    tabelas["primeiro_exame"] = t
    # 4. Quem não é testado: suspeita, real, conclusão.
    t = (casos.groupby(["teste", "politica", "suspeita_n"])
         .agg(casos=("n_exames", "size"), p_sem_exame=("n_exames", lambda s: (s == 0).mean()),
              p_dois=("n_exames", lambda s: (s >= 2).mean()), acerto=("acertou", "mean"),
              exames=("n_exames", "mean"))).round(3)
    tabelas["por_suspeita"] = t
    for nome, t in tabelas.items():
        t.to_csv(SAIDA / f"tab_{nome}.csv")
        print(f"\n== {nome}\n{t.to_string()}")
    return tabelas


def limiar_segundo_exame(testes=("teste_rio", "teste_recife_2016_sup", "teste_recife_2021_sup",
                                  "teste_recife_2016", "teste_recife_2021")) -> pd.DataFrame:
    """Pedir sempre o 2º exame depois de um negativo é ótimo aqui? Conta de decisão com a posterior medida.

    No momento em que o 1º laudo negativo volta, o agente pode concluir já ou
    pedir o outro exame. Concluir a classe c vale `10·p_c + e_c·(1 − p_c)`,
    com e_c = −30 para "outro" (falso negativo de vigilância) e −20 para uma
    arbovirose. Pedir o 2º exame vale o que a regra de fato obtém depois dele
    (recompensa média de conclusão medida, menos o custo 4 do exame). O bônus
    de concluir caso investigado vale nos dois ramos e se cancela.

    A posterior p_c é estimada por regressão logística multinomial, com
    validação cruzada, sobre tudo o que o agente observa do caso nesse momento:
    rótulo do médico, qual exame deu negativo, densidade local de confirmados
    das duas doenças e dia da epidemia. Se nenhum caso tiver posterior alta o
    bastante para que concluir já valha mais que o 2º exame, "sempre pedir o
    2º exame" é a política ótima e não há o que selecionar.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict

    casos = pd.read_csv(SAIDA / "casos.csv.gz", low_memory=False)
    erro = np.array([-20.0, -20.0, -30.0])  # concluir dengue, chik, outro errado
    linhas = []
    for teste in testes:
        g = casos[(casos.politica == "regra") & (casos.teste == teste) & (casos.laudo1 == "neg")
                  & (casos.suspeita_n != "outro")]
        if g.empty:
            continue
        # O que o 2º exame rendeu, de fato, na regra (que sempre o pede).
        r_conc = np.where(g.acertou, 10.0, np.where((g.conclusao == 2) & (g.real != 2), -30.0, -20.0))
        v_teste = float(r_conc.mean()) - 4.0
        X = pd.get_dummies(g[["suspeita_n", "primeiro_exame"]]).astype(float)
        X["dens_d"], X["dens_c"], X["dia"] = np.log1p(g.dens_d), np.log1p(g.dens_c), g.dia / 300
        post = cross_val_predict(LogisticRegression(max_iter=2000), X, g.real.to_numpy(), cv=5,
                                 method="predict_proba")
        v_parar = (10.0 * post + erro * (1 - post)).max(axis=1)
        ganho = v_parar - v_teste
        linhas.append({
            "teste": teste, "casos": len(g), "valor_2o_exame": round(v_teste, 2),
            "posterior_max_mediana": round(float(np.median(post.max(axis=1))), 3),
            "posterior_max_p99": round(float(np.quantile(post.max(axis=1), 0.99)), 3),
            "frac_parar_melhor": round(float((ganho > 0).mean()), 4),
            "ganho_max_por_caso": round(float(ganho.max()), 2),
        })
    t = pd.DataFrame(linhas)
    t.to_csv(SAIDA / "tab_limiar_segundo_exame.csv", index=False)
    print(t.to_string(index=False))
    return t


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--analisa", action="store_true")
    ap.add_argument("--limiar", action="store_true", help="o 2º exame é sempre ótimo? (conta de decisão)")
    args = ap.parse_args(argv)
    if args.limiar:
        limiar_segundo_exame()
    elif args.analisa:
        analisa()
    else:
        roda(args.workers)


if __name__ == "__main__":
    main()
