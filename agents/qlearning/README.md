# Q-Learning tabular

Agente de **Q-Learning tabular** integrado ao ambiente novo (`make_env` +
wrappers `map_tensor` + `case_by_case`). Decide **uma ação por caso**, no mesmo
formato do random e do DQN.

## Arquivos

| Arquivo | Papel |
|---------|-------|
| `agent.py` | `QLearningAgent`, `QLearningAgentRunner`, codificação de estado |
| `train.py` | Script de treino (CLI) |
| `watch.py` | Visualização com Pygame |
| `qlearning_agent_old.py` | Legado (env bruto, ação por dia inteiro) |

## Estado discretizado

A Q-table indexa estados pela chave:

```
{day_bucket}|{clinical}|{testd}|{testc}
```

- `day_bucket` = `t // day_bucket_size` (default 5 dias por bucket)
- `clinical`, `testd`, `testc` = valores dos canais 0–2 do mapa tensor na célula
  do caso corrente (`CaseByCaseWrapper.current_case`)

Isso captura o contexto clínico e de laboratório do caso sem explodir o espaço
de estados com coordenadas brutas.

## Treinar

```bash
poetry run python agents/qlearning/train.py
poetry run python agents/qlearning/train.py --config experiments/configs/train/qlearning_default.yaml
poetry run python agents/qlearning/train.py --episodes 100 --output-dir results/qlearning
```

Hiperparâmetros no YAML (`experiments/configs/train/qlearning_default.yaml`):

| Parâmetro | Default | Descrição |
|-----------|---------|-----------|
| `episodes` | 200 | Episódios de treino |
| `alpha` | 0.5 | Taxa de aprendizado |
| `gamma` | 0.5 | Fator de desconto |
| `epsilon_start` | 0.30 | ε inicial (exploração) |
| `epsilon_final` | 0.05 | ε final |
| `epsilon_decay_episodes` | 150 | Episódios para decair ε |
| `day_bucket_size` | 5 | Granularidade do dia no estado |

Saídas em `results/qlearning/`:

- `q_table.pkl` — checkpoint usado pelo benchmark e pelo watch
- `q_table.txt` — dump legível
- `training_log.csv` — recompensa e tamanho da tabela por episódio

## Avaliar no benchmark

1. Treine e gere `results/qlearning/q_table.pkl`.
2. Descomente `qlearning` em `experiments/configs/benchmark.yaml`.
3. Rode:

```bash
poetry run python experiments/evaluate.py --config experiments/configs/benchmark.yaml
```

O caminho da Q-table pode ser customizado em `checkpoints.qlearning` no YAML.

## Assistir

```bash
poetry run python agents/qlearning/watch.py --q-table results/qlearning/q_table.pkl --seed 100
```

## Limitações

- Tabular: o espaço de estados cresce com o treino; não escala como DQN para
  grids grandes ou `epilength` longo.
- O estado não inclui densidade espacial ao redor do caso (só a célula do caso).
  Para políticas mais ricas, prefira DQN/PPO.
