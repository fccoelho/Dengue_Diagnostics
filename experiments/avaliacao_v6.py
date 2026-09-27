"""Avaliação cruzada do v6: cada agente treinado em cada cenário de teste (mobilidade).

Unidade = (agente, cenário de teste), com 30 surtos pareados inéditos
(sementes 3001-3030, nunca usadas em treino nem em avaliações anteriores).
Cada unidade grava `results/v6_avaliacao/<teste>/<agente>.csv` e é pulada se
já existir — dá para rodar aos poucos, em paralelo com os treinos, e retomar.
Agentes treinados só entram quando o `policy_final.pth` existe.

Uso:
    python -m experiments.avaliacao_v6 --workers 4        # roda o que estiver pronto, espera o resto
    python -m experiments.avaliacao_v6 --analisa           # matriz de transferência + bootstrap
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple

import pandas as pd

_RAIZ = Path(__file__).resolve().parents[1]
if str(_RAIZ) not in sys.path:
    sys.path.insert(0, str(_RAIZ))

SAIDA = _RAIZ / "results" / "v6_avaliacao"
TREINOS = _RAIZ / "results" / "ppo_v6"
SEEDS = tuple(range(3001, 3031))
TESTES = ("teste_sintetico", "teste_rio", "teste_recife_2016", "teste_recife_2021")
CENARIOS = ("sintetico", "rio", "recife", "misto", "mistolongo", "recifepos", "riofis", "recifefis", "recifesup")
EXTRAS = ("mistolongo", "recifepos", "riofis", "recifefis", "recifesup")  # grupos extras: só o braço C
# Escala física e casos "outro" na área habitada: ambientes de teste próprios
# (ver experiments/cenarios.py). Os demais grupos são testados em TESTES.
TESTES_FIS = ("teste_rio_fis", "teste_recife_2016_fis", "teste_recife_2021_fis")
TESTES_SUP = ("teste_rio", "teste_recife_2016_sup", "teste_recife_2021_sup")
TESTES_DO_GRUPO = {"riofis": TESTES_FIS, "recifefis": TESTES_FIS, "recifesup": TESTES_SUP}
BRACOS = ("C", "A")
SEEDS_TREINO = (45, 46, 47)
FIXAS = ("sequencial_clinico", "sequencial_confia_outro", "sequencial", "testtwice", "testonce", "clinical")
RAM_LIVRE = 4.0  # GB: acima da margem da fila de treinos, que tem prioridade


def unidades() -> List[Tuple[str, str]]:
    """(agente, teste). Agente: nome de política fixa ou '<cenario>_<braco>_s<seed>'."""
    todos_testes = TESTES + tuple(dict.fromkeys(TESTES_FIS + TESTES_SUP))
    un = [(a, t) for a in FIXAS for t in dict.fromkeys(todos_testes)]
    for s in SEEDS_TREINO:
        for c in CENARIOS:
            for b in BRACOS:
                if c in EXTRAS and b == "A":
                    continue
                un += [(f"{c}_{b}_s{s}", t) for t in TESTES_DO_GRUPO.get(c, TESTES)]
    return un


def _destino(agente: str, teste: str) -> Path:
    return SAIDA / teste / f"{agente}.csv"


def _pronta(agente: str) -> bool:
    return agente in FIXAS or (TREINOS / agente / "policy_final.pth").exists()


def roda_unidade(agente: str, teste: str) -> str:
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    from experiments import robustez as R
    from experiments.evaluate import AGENT_REGISTRY, _make_runner
    from dengue_envs.wrappers import make_env

    if agente in FIXAS:
        runner = _make_runner(agente, {})
    else:
        from agents.ppo.agent import PPOAgentRunner
        runner = PPOAgentRunner(policy_path=str(TREINOS / agente / "policy_final.pth"), device="cpu")
    env = make_env(R.config_ambiente(teste))
    linhas = []
    for s in SEEDS:
        linha = {"agente": agente, "teste": teste, "seed": s}
        linha.update(R.roda_episodio(runner, env, s))
        linhas.append(linha)
    env.close()
    destino = _destino(agente, teste)
    destino.parent.mkdir(parents=True, exist_ok=True)
    tmp = destino.with_suffix(".tmp")
    pd.DataFrame(linhas).to_csv(tmp, index=False)
    tmp.replace(destino)
    return f"{agente} @ {teste}"


def roda(workers: int) -> None:
    feitas = set()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        em_voo = {}
        while True:
            faltam = [u for u in unidades() if not _destino(*u).exists() and u not in em_voo.values()]
            prontas = [u for u in faltam if _pronta(u[0])]
            # Guarda de memória: a avaliação nunca disputa RAM com os treinos.
            import psutil
            while prontas and len(em_voo) < workers and psutil.virtual_memory().available / 1e9 >= RAM_LIVRE:
                u = prontas.pop(0)
                em_voo[pool.submit(roda_unidade, *u)] = u
                time.sleep(30)
            if not em_voo and not faltam:
                print(time.strftime("%d/%m %H:%M"), "FIM da avaliação", flush=True)
                return
            if em_voo:
                for fut in as_completed(list(em_voo), timeout=None):
                    u = em_voo.pop(fut)
                    try:
                        print(time.strftime("%d/%m %H:%M"), "OK", fut.result(), flush=True)
                    except Exception as e:  # noqa: BLE001 - registra e segue
                        print(time.strftime("%d/%m %H:%M"), "FALHOU", u, repr(e), flush=True)
                        feitas.add(u)
                    break
            else:
                time.sleep(300)  # nada pronto: espera treinos terminarem


# --------------------------------------------------------------------------
# análise
# --------------------------------------------------------------------------

def carrega() -> pd.DataFrame:
    partes = [pd.read_csv(f) for f in SAIDA.glob("*/*.csv")]
    df = pd.concat(partes, ignore_index=True)
    treinado = df["agente"].str.match(r"^(" + "|".join(CENARIOS) + r")_[AC]_s\d+$")
    partes = df.loc[treinado, "agente"].str.extract(r"^(?P<cenario>\w+?)_(?P<braco>[AC])_s(?P<seed_treino>\d+)$")
    df.loc[treinado, ["cenario", "braco", "seed_treino"]] = partes.values
    df.loc[~treinado, "cenario"] = "fixa"
    df.loc[~treinado, "braco"] = df.loc[~treinado, "agente"]
    df.loc[~treinado, "seed_treino"] = 0
    df["seed_treino"] = df["seed_treino"].astype(int)
    return df


def analisa(n_replicas: int = 10_000) -> pd.DataFrame:
    """Para cada cenário de teste: todos os agentes (braço treinado em cada cenário) e contrastes."""
    from experiments import bootstrap as B

    df = carrega()
    linhas = []
    for teste, g in df.groupby("teste"):
        g = g.assign(braco_nome=lambda d: d.apply(
            lambda r: r["braco"] if r["cenario"] == "fixa" else f"{r['braco']}|{r['cenario']}", axis=1))
        tab = g.rename(columns={"braco_nome": "braco_", "seed": "seed_aval"})[
            ["braco_", "seed_treino", "seed_aval", "recompensa", "acuracia", "exames"]].rename(
            columns={"braco_": "braco"})
        comps = []
        for c in CENARIOS:
            if c not in EXTRAS:
                comps.append((f"C|{c}", f"A|{c}"))
            comps.append((f"C|{c}", "sequencial_clinico"))
            comps.append((f"C|{c}", "sequencial_confia_outro"))
        comps += [("sequencial_confia_outro", "sequencial_clinico"),
                  ("C|mistolongo", "C|misto"), ("C|recifepos", "C|recife"),
                  ("C|recifefis", "C|riofis"), ("C|recifesup", "C|rio")]
        presentes = set(tab.braco)
        comps = [c for c in comps if c[0] in presentes and c[1] in presentes]
        bracos, cmp_ = B.analisa(tab, comparacoes=comps, n_replicas=n_replicas)
        bracos["teste"], cmp_["teste"] = teste, teste
        bracos.to_csv(SAIDA / f"bracos_{teste}.csv", index=False)
        cmp_.to_csv(SAIDA / f"comparacoes_{teste}.csv", index=False)
        linhas.append(bracos)
    tudo = pd.concat(linhas, ignore_index=True)
    rec = tudo[tudo.metrica == "recompensa"]
    matriz = rec[rec.braco.str.startswith(("C|", "A|"))].pivot_table(index="braco", columns="teste", values="media")
    matriz.to_csv(SAIDA / "matriz_transferencia.csv")
    print(matriz.round(0).to_string())
    return matriz


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--analisa", action="store_true")
    args = ap.parse_args(argv)
    if args.analisa:
        analisa()
    else:
        roda(args.workers)


if __name__ == "__main__":
    main()
