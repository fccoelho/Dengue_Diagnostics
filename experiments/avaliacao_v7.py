"""Avaliação do v7: matriz treino × teste no ambiente corrigido, com 5 sementes, e varreduras de sensibilidade.

O que muda em relação ao v6 (`experiments/avaliacao_v6.py`):

- **Ambiente corrigido em tudo.** Os casos "outro" caem só na área habitada.
  No Recife isso é o `teste_recife_*_sup`; no Rio e no sintético a correção
  não muda nada (o Rio ocupa o grid todo, o sintético não usa superfícies),
  então os testes são os mesmos do v6 e os resultados já calculados valem.
- **5 sementes de treino** (45-49) por grupo e braço.
- **Varreduras** no Rio, para responder "tudo é simulado": custo do exame
  (2, 3, 6, 8, além do 4 da matriz), sensibilidade do RT-PCR (0,80, 0,90,
  0,99) e acurácia do médico fixa (0,55, 0,70, 0,85, 0,95). Os agentes são
  os treinados no cenário padrão, avaliados sem retreino; na varredura de
  custo entram também agentes treinados com custo 2 e 8.

Cada unidade (agente, condição) grava `results/v6_avaliacao/<condição>/<agente>.csv`
e é pulada se já existir. Mesmos 30 surtos (sementes 3001-3030) do v6.

Uso:
    python -m experiments.avaliacao_v7 --workers 3     # roda o que estiver pronto, espera o resto
    python -m experiments.avaliacao_v7 --analisa       # bootstrap + tabelas
    python -m experiments.avaliacao_v7 --conjunto confirmacao --workers 10   # surtos 4001-4030
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

_RAIZ = Path(__file__).resolve().parents[1]
if str(_RAIZ) not in sys.path:
    sys.path.insert(0, str(_RAIZ))

SAIDA = _RAIZ / "results" / "v6_avaliacao"
TREINOS = _RAIZ / "results" / "ppo_v6"
SEEDS = tuple(range(3001, 3031))
SEEDS_TREINO = (45, 46, 47, 48, 49)
RAM_LIVRE = 4.0

# condição -> (ambiente de teste, parâmetros sobrescritos)
MATRIZ = ("teste_rio", "teste_recife_2016_sup", "teste_recife_2021_sup", "teste_sintetico")
CONDICOES: Dict[str, Tuple[str, dict]] = {t: (t, {}) for t in MATRIZ}
CONDICOES.update({f"teste_rio__custo{c}": ("teste_rio", {"test_cost": float(c), "epi_confirm_cost": float(c)})
                  for c in (2, 3, 6, 8)})
CONDICOES.update({f"teste_rio__sens{s:.2f}": ("teste_rio", {"lab_sensitivity": s}) for s in (0.80, 0.90, 0.99)})
CONDICOES.update({f"teste_rio__medico{a:.2f}": ("teste_rio", {"clinical_specificity": [a, a]})
                  for a in (0.55, 0.70, 0.85, 0.95)})

GRUPOS = ("sintetico", "rio", "recifesup", "mistosup")
FIXAS = ("sequencial_clinico", "sequencial_confia_outro", "sequencial", "testtwice", "testonce", "clinical")
FIXAS_VARREDURA = ("sequencial_clinico", "sequencial_confia_outro", "testtwice", "testonce")

# Conjuntos de surtos. O "principal" (3001-3030) foi OLHADO para desenhar a regra
# derivada e a correção do simulador; o de "confirmacao" (4001-4030) nunca foi
# usado e serve para confirmar as comparações principais sem esse viés de
# seleção: a matriz e os custos 6 e 8 (onde a regra derivada parecia vencer).
CONJUNTOS = {
    "principal": (SEEDS, SAIDA, tuple(CONDICOES)),
    "confirmacao": (tuple(range(4001, 4031)), SAIDA / "confirmacao",
                    MATRIZ + ("teste_rio__custo6", "teste_rio__custo8")),
}


def unidades(conjunto: str = "principal") -> List[Tuple[str, str]]:
    un = []
    for cond in CONJUNTOS[conjunto][2]:
        if cond in MATRIZ:
            agentes = list(FIXAS) + [f"{g}_{b}_s{s}" for s in SEEDS_TREINO for g in GRUPOS for b in "CA"]
        else:
            agentes = list(FIXAS_VARREDURA) + [f"rio_C_s{s}" for s in SEEDS_TREINO]
            if "custo" in cond:
                agentes += [f"riocusto{c}_C_s{s}" for c in (2, 8) for s in (45, 46, 47)]
        un += [(a, cond) for a in agentes]
    return un


def _destino(agente: str, cond: str, conjunto: str = "principal") -> Path:
    return CONJUNTOS[conjunto][1] / cond / f"{agente}.csv"


def _pronta(agente: str) -> bool:
    return agente in FIXAS or (TREINOS / agente / "policy_final.pth").exists()


def roda_unidade(agente: str, cond: str, conjunto: str = "principal") -> str:
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    from experiments import robustez as R
    from experiments.evaluate import _make_runner
    from dengue_envs.wrappers import make_env

    if agente in FIXAS:
        runner = _make_runner(agente, {})
    else:
        from agents.ppo.agent import PPOAgentRunner
        runner = PPOAgentRunner(policy_path=str(TREINOS / agente / "policy_final.pth"), device="cpu")
    teste, sobrescreve = CONDICOES[cond]
    env = make_env(R.config_ambiente(teste, **sobrescreve))
    linhas = []
    for s in CONJUNTOS[conjunto][0]:
        linha = {"agente": agente, "teste": cond, "seed": s}
        linha.update(R.roda_episodio(runner, env, s))
        linhas.append(linha)
    env.close()
    destino = _destino(agente, cond, conjunto)
    destino.parent.mkdir(parents=True, exist_ok=True)
    tmp = destino.with_suffix(".tmp")
    pd.DataFrame(linhas).to_csv(tmp, index=False)
    tmp.replace(destino)
    return f"{agente} @ {cond} ({conjunto})"


PARADA_MAX_S = 40 * 60  # sem nenhuma unidade concluída nesse tempo, algo travou


def _impede_suspensao() -> None:
    """Pede ao Windows para não suspender a máquina enquanto este processo roda.

    Em 29/09 a avaliação ficou ~30 h parada depois de a máquina dormir: os
    workers voltaram vivos, mas sem trabalhar. A requisição vale só para este
    processo (não altera configurações de energia) e cai quando ele termina.
    """
    if os.name == "nt":
        import ctypes

        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)


def _aborta_parada(em_voo: dict) -> None:
    """Vigia: registra as unidades presas, mata os workers e sai com erro (relançar retoma)."""
    import psutil

    print(time.strftime("%d/%m %H:%M"), f"PARADO há {PARADA_MAX_S // 60} min; em voo:",
          sorted(em_voo.values()), flush=True)
    for filho in psutil.Process().children(recursive=True):
        filho.kill()
    os._exit(3)


def roda(workers: int, conjunto: str = "principal") -> None:
    import psutil
    from concurrent.futures import TimeoutError as Esgotado

    _impede_suspensao()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        em_voo = {}
        while True:
            faltam = [u for u in unidades(conjunto) if not _destino(*u, conjunto).exists()
                      and u not in em_voo.values()]
            prontas = [u for u in faltam if _pronta(u[0])]
            while prontas and len(em_voo) < workers and psutil.virtual_memory().available / 1e9 >= RAM_LIVRE:
                u = prontas.pop(0)
                em_voo[pool.submit(roda_unidade, *u, conjunto)] = u
                time.sleep(20)
            if not em_voo and not faltam:
                print(time.strftime("%d/%m %H:%M"), "FIM da avaliação", flush=True)
                return
            if em_voo:
                try:
                    for fut in as_completed(list(em_voo), timeout=PARADA_MAX_S):
                        u = em_voo.pop(fut)
                        try:
                            print(time.strftime("%d/%m %H:%M"), "OK", fut.result(), flush=True)
                        except Exception as e:  # noqa: BLE001
                            print(time.strftime("%d/%m %H:%M"), "FALHOU", u, repr(e), flush=True)
                        break
                except Esgotado:
                    _aborta_parada(em_voo)
            else:
                time.sleep(300)


# --------------------------------------------------------------------------
# análise
# --------------------------------------------------------------------------

def _tabela(cond: str, conjunto: str = "principal") -> pd.DataFrame:
    """Todas as unidades v7 da condição, no formato do `experiments.bootstrap`."""
    agentes = {a for a, c in unidades(conjunto) if c == cond}
    partes = []
    for f in (CONJUNTOS[conjunto][1] / cond).glob("*.csv"):
        if f.stem not in agentes:
            continue
        d = pd.read_csv(f)
        if f.stem in FIXAS:
            d["braco"], d["seed_treino"] = f.stem, 0
        else:
            grupo, braco, seed = f.stem.rsplit("_", 2)
            d["braco"], d["seed_treino"] = f"{braco}|{grupo}", int(seed[1:])
        partes.append(d)
    df = pd.concat(partes, ignore_index=True).rename(columns={"seed": "seed_aval"})
    return df[["braco", "seed_treino", "seed_aval", "recompensa", "acuracia", "exames"]]


def analisa(n_replicas: int = 10_000, conjunto: str = "principal") -> None:
    from experiments import bootstrap as B

    saida = SAIDA / ("v7" if conjunto == "principal" else f"v7_{conjunto}")
    saida.mkdir(exist_ok=True)
    for cond in CONJUNTOS[conjunto][2]:
        tab = _tabela(cond, conjunto)
        presentes = set(tab.braco)
        comps = [("sequencial_confia_outro", "sequencial_clinico")]
        for g in GRUPOS + ("riocusto2", "riocusto8"):
            comps += [(f"C|{g}", f"A|{g}"), (f"C|{g}", "sequencial_clinico"), (f"C|{g}", "sequencial_confia_outro")]
        comps = [c for c in comps if c[0] in presentes and c[1] in presentes]
        bracos, cmp_ = B.analisa(tab, comparacoes=comps, n_replicas=n_replicas)
        bracos["condicao"], cmp_["condicao"] = cond, cond
        bracos.to_csv(saida / f"bracos_{cond}.csv", index=False)
        cmp_.to_csv(saida / f"comparacoes_{cond}.csv", index=False)
        n = tab.groupby("braco").seed_treino.nunique()
        print(f"{cond}: {len(presentes)} braços; sementes por braço treinado: "
              f"{sorted(set(n[n.index.str.contains('|', regex=False)]))}", flush=True)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--analisa", action="store_true")
    ap.add_argument("--conjunto", choices=tuple(CONJUNTOS), default="principal",
                    help="principal: surtos 3001-3030; confirmacao: 4001-4030, nunca olhados")
    args = ap.parse_args(argv)
    if args.analisa:
        analisa(conjunto=args.conjunto)
    else:
        roda(args.workers, args.conjunto)


if __name__ == "__main__":
    main()
