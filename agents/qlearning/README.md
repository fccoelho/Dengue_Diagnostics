# Q-Learning tabular

Agente integrado ao ambiente atual (`make_env` + `map_tensor` + `case_by_case`).
Uma ação por caso (`Discrete(6)`), no mesmo formato do random.

## Arquivos

| Arquivo | Papel |
|---------|-------|
| `agent.py` | `QLearningAgent`, `QLearningAgentRunner`, save/load |
| `state.py` | `StateEncoder` (`rich_v1` padrão, `compact_v1` opcional) |
| `train.py` | Treino CLI |
| `watch.py` | Visualização Pygame |

Cópias antigas do agente (env bruto) foram removidas do working tree — continuam no histórico do git.

## Estado (`rich_v1`)

Chave discretizada (sem `case_id`, sem `(x,y)` brutos, sem mapa inteiro). Exemplo:

```
dp=1|cl=0|adx=0|td=0|tc=0|epi=0|tdn=0|lp=0|cf=0|q=2|dd=1|dc=0|cd=1|ld=2|lc=0|lpd=0|lpc=0|la=1|ct=2|wk=1
```

Campos: fase temporal, clínica, diagnóstico, testes, epi, conflito, quadrante, distâncias aos focos, vizinhança local, carga do dia/episódio. Detalhe dos bins no changelog abaixo / código em `state.py`.

`compact_v1` (`day|clinical|testd|testc`) existe só como baseline mínimo no YAML.

## Treinar

```bash
# Sintético
poetry run python agents/qlearning/train.py --config experiments/configs/train/qlearning_default.yaml

# Kriging (requer results/kriging/*.npz)
poetry run python agents/qlearning/train.py --config experiments/configs/train/qlearning_kriging.yaml
```

Saídas em `results/qlearning/` (ou `results/qlearning_kriging/`): `q_table.pkl`, `q_table.txt`, `training_log.csv`.

## Benchmark / watch

```bash
# Descomente qlearning em experiments/configs/benchmark.yaml
poetry run python experiments/evaluate.py --config experiments/configs/benchmark.yaml

poetry run python agents/qlearning/watch.py --q-table results/qlearning/q_table.pkl --seed 100
```

## Limitações

Tabular: o número de estados cresce com o treino; não escala como DQN em grids muito grandes.
