# Experiments

Ponto único para avaliar agentes no ambiente `DengueDiag`, com configuração por
YAML e resultados em um formato comum (comparável entre todos os agentes).

## Estrutura

```
experiments/
├── configs/
│   ├── env/
│   │   ├── synthetic_default.yaml   # ambiente sintético padrão
│   │   └── synthetic_large.yaml     # grid/epidemia maiores
│   └── benchmark.yaml               # quais agentes/seeds rodar e onde salvar
├── evaluate.py                      # roda o benchmark e salva os CSVs
└── README.md
```

## Como rodar os baselines

Pré-requisito: ambiente instalado (`poetry install` com Python 3.12).

Da raiz do repositório:

```bash
poetry run python experiments/evaluate.py --config experiments/configs/benchmark.yaml
```

Isso avalia os agentes listados em `benchmark.yaml` (hoje `clinical` e `random`),
usando o ambiente de `configs/env/synthetic_default.yaml`, e salva os resultados
em `results/baseline/`.

Baselines disponíveis:

- `clinical` — **clínico puro**: aceita sempre o diagnóstico clínico (ação "não
  fazer nada"), sem pedir testes. É o piso de referência: qualquer agente deve
  superá-lo para justificar seu custo. A diferença de recompensa para ele é a
  **contribuição marginal** do agente.
- `random` — escolhe uma ação uniforme por caso.

Para trocar o ambiente, edite `env_config` no `benchmark.yaml` (ex.:
`env/synthetic_large.yaml`) ou ajuste os YAMLs em `configs/env/`.

### Ranking e leitura dos resultados

O benchmark ranqueia por **`Recompensa Total` (métrica primária)** — é a que
mais reflete o objetivo real. As demais métricas são **diagnóstico secundário**:
úteis para entender o comportamento, mas não para ranquear. Ao final, o script
também imprime a contribuição marginal de cada agente sobre o clínico puro.

> Atenção às duas acurácias: a `Acurácia` é **binária** (dengue vs. resto) e é
> leniente — ela pode subir mesmo quando o agente confunde chik↔outro. A
> `Acurácia Multiclasse` exige acertar a classe exata (dengue/chik/outro) e
> revela essa confusão. No baseline, o `random` chega a ter acurácia binária
> maior que o clínico, mas multiclasse menor e recompensa muito pior.

## Saídas (padrão comparável entre agentes)

Todos os agentes produzem **as mesmas colunas**, então os CSVs são diretamente
comparáveis. São gravados em `output_dir` (default `results/baseline/`):

| Arquivo | Conteúdo |
|---------|----------|
| `benchmark_raw.csv` | uma linha por `(agente, seed)` |
| `benchmark_mean.csv` | média das métricas por agente |
| `benchmark_std.csv` | desvio-padrão das métricas por agente |

Colunas (ver `agents/base.py::RESULT_COLUMNS`):

- Identificação: `agent`, `seed`, `clinical_specificity`
- Métrica primária: `Recompensa Total`
- Métricas secundárias (de `env.get_episode_metrics()`): `Acurácia Multiclasse`,
  `Acurácia`, `Sensibilidade (Dengue)`, `Especificidade`, `F1-Score`,
  `Precisão`, `Custo Total de Testes`, `Testes Realizados`, `Custo por Acerto`,
  `Redução de Testes (%)`

Para a mesma `seed`, todos os agentes enxergam o mesmo cenário (mundo e
incerteza clínica reprodutíveis), garantindo comparação justa.

> A pasta `results/` é ignorada pelo git (dados gerados).

## Visualizar um agente agindo (renderização)

O benchmark **não** abre janela (só gera CSVs). Para *assistir* a um agente
decidindo caso a caso, com o mapa e os gráficos de recompensa/acurácia, use os
scripts de watch. Eles constroem o mesmo ambiente (mesmos wrappers), mas com
`render_mode="human"`.

```bash
# Agente aleatório
poetry run python agents/random/watch.py                 # config/seed/fps padrão
poetry run python agents/random/watch.py --seed 100 --fps 8
poetry run python agents/random/watch.py --config experiments/configs/env/synthetic_large.yaml

# Policy DQN treinada (aponte para o .pth gerado no treino)
poetry run python agents/deepq/watch.py --policy caminho/para/policy.pth --seed 100
```

Usando a mesma `--seed` do `benchmark.yaml` (ex.: `100`), você vê exatamente o
cenário avaliado. Ao final, um resumo do episódio é impresso (recompensa,
acurácia multiclasse/binária, testes). Feche a janela ou use Ctrl+C para sair.

O baseline **clínico** não tem script de watch: ele existe apenas como piso de
referência das métricas (não faz ações para observar).

Por baixo, tudo compartilha `agents/watch.py::run_watch`. Qualquer agente que
herde de `EpisodeRunner` ganha `.watch(env_config, seed=..., render_fps=...)` de
graça; agentes com política (DQN) usam `run_watch(..., setup=lambda env: act_fn)`.

## Adicionando novos agentes

1. Implemente um runner com o contrato `agents.base.AgentRunner`
   (método `evaluate(make_env, seeds, env_config) -> list[dict]`).
   - Para agentes sem treino que decidem uma ação por caso, herde de
     `agents.base.EpisodeRunner` e implemente apenas `name` e
     `choose_action(env)` (é o que `random` e `clinical` fazem). Bônus: já
     ganham `.watch(...)` para visualização.
2. Registre-o em `AGENT_REGISTRY` no `experiments/evaluate.py`.
3. Adicione o nome dele em `agents:` no `benchmark.yaml`.
