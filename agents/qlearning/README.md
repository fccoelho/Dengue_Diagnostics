# Q-Learning tabular

Agente de **Q-Learning tabular** integrado ao ambiente novo (`make_env` +
wrappers `map_tensor` + `case_by_case`). Decide **uma ação por caso**, no mesmo
formato do random e do DQN.

## Arquivos

| Arquivo | Papel |
|---------|-------|
| `agent.py` | `QLearningAgent`, `QLearningAgentRunner`, save/load com metadados |
| `state.py` | `StateEncoder` — codificação compacta ou **rich** (padrão) |
| `train.py` | Script de treino (CLI) |
| `watch.py` | Visualização com Pygame |
| `qlearning_agent_old.py` | Legado (env bruto, ação por dia inteiro) |

## Estado (`rich_v1`, padrão)

A Q-table indexa estados por uma chave **discretizada e generalizável** (sem
`case_id`, sem `(x,y)` exatos, sem string de `obs` inteiro, sem mapa 400×400).

Formato (campos separados por `|`):

```
dp=1|cl=0|adx=0|td=0|tc=0|epi=0|tdn=0|lp=0|cf=0|q=2|dd=1|dc=0|cd=1|ld=2|lc=0|lpd=0|lpc=0|la=1|ct=2|wk=1
```

| Campo | Significado | Valores (bins) |
|-------|-------------|----------------|
| `dp` | Fase temporal do episódio | 0=início (t≤5), 1=meio (≤15), 2=fim do surto, 3=cauda pós-surto |
| `cl` | Suspeita clínica | 0=dengue, 1=chik, 2=outro |
| `adx` | Diagnóstico atual do agente | 0/1/2 |
| `td`, `tc` | Status teste dengue/chik | 0=não testado, 1=neg, 2=pos, 3=inconclusivo |
| `epi` | Confirmação epidemiológica | 0/1 |
| `tdn` | Quantos testes já pedidos | 0/1/2 |
| `lp` | Laudo positivo em algum teste | 0/1 |
| `cf` | Conflito clínico vs laboratório | 0/1 |
| `q` | Quadrante do mapa | 0=NW, 1=NE, 2=SW, 3=SE |
| `dd`, `dc` | Distância ao foco dengue/chik | 0=perto, 1=médio, 2=longe |
| `cd` | Mais perto do foco dengue que chik | 0/1 |
| `ld`, `lc` | Suspeitas clínicas dengue/chik na vizinhança | 0/1/2/3+ (janela `(2r+1)²`, r=2) |
| `lpd`, `lpc` | Positivos de lab na vizinhança | buckets |
| `la` | Casos ativos hoje na vizinhança | buckets |
| `ct` | Casos reportados hoje (global) | 0 / 1–2 / 3–5 / 6+ |
| `wk` | Carga acumulada do episódio | 0=baixa, 1=média, 2=alta (% de `episize`) |

### Variante legada (`compact_v1`)

Ainda disponível para comparação:

```
{day_bucket}|{clinical}|{testd}|{testc}
```

Configure `state.version: compact_v1` no YAML. Costuma gerar ~10 estados — útil
só como baseline mínimo.

## Treinar

```bash
poetry run python agents/qlearning/train.py
poetry run python agents/qlearning/train.py --config experiments/configs/train/qlearning_default.yaml
```

Hiperparâmetros em `experiments/configs/train/qlearning_default.yaml`:

| Seção | Parâmetro | Default |
|-------|-----------|---------|
| `state` | `version` | `rich_v1` |
| `state` | `local_radius` | 2 |
| `state` | `day_early_max` / `day_mid_max` | 5 / 15 |
| `train` | `episodes` | 200 |
| `train` | `alpha`, `gamma` | 0.5, 0.5 |
| `train` | `epsilon_start` → `epsilon_final` | 0.30 → 0.05 |

Saídas em `results/qlearning/`:

- `q_table.pkl` — inclui `state_version`, `encoder_config` e a tabela
- `q_table.txt` — dump legível
- `training_log.csv` — recompensa e `states_seen` por episódio

Se retreinar com `state.version` diferente do checkpoint existente, o treino
**reinicia a Q-table** (evita misturar chaves incompatíveis).

## Avaliar no benchmark

1. Treine e gere `results/qlearning/q_table.pkl`.
2. Descomente `qlearning` em `experiments/configs/benchmark.yaml`.
3. Rode:

```bash
poetry run python experiments/evaluate.py --config experiments/configs/benchmark.yaml
```

Checkpoint opcional:

```yaml
checkpoints:
  qlearning: results/qlearning/q_table.pkl
```

## Assistir

```bash
poetry run python agents/qlearning/watch.py --q-table results/qlearning/q_table.pkl --seed 100
```

O encoder é lido automaticamente dos metadados do `.pkl`.

## Limitações

- Tabular: mesmo com `rich_v1`, o espaço de estados cresce com o treino; não
  escala como DQN para grids muito grandes ou `epilength` longo.
- Vizinhança local resume o mapa, mas não substitui uma rede convolucional.

---

## Changelog (log de evolução)

### 2026-07-08 — `rich_v1` (estado completo tabular)

- Novo módulo `state.py` com `StateEncoder`.
- Estado padrão passa de 4 campos (~9 estados) para **20 features discretas**:
  clínica, decisão do agente, laboratório, fase temporal, quadrante, distância
  aos focos, contexto espacial local e carga do dia/episódio.
- Checkpoints v2: `q_table.pkl` guarda `state_version` + `encoder_config`.
- Treino detecta mismatch de versão e reinicia tabela se necessário.
- `compact_v1` mantido como opção no YAML.

### 2026-07-08 — Integração inicial ao ambiente novo

- `QLearningAgentRunner` + benchmark + watch.
- Estado mínimo `day|clinical|testd|testc` (posteriormente renomeado `compact_v1`).
- Scripts `train.py` / `watch.py` e config YAML.

### Legado — `qlearning_agent_old.py`

- Env bruto, ação por **dia inteiro** (lista de `(case_id, action)`).
- Estado = `str(obs[-1]) + str(case_id)` — instável e acoplado ao env antigo.
