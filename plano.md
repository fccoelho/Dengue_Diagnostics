# Confirmação do estado atual + plano de reorganização

---

# ✅ Rodada atual — Recompensa reformulada + ambiente rodando (Python 3.12)

Esta rodada resolveu o **P1 de recompensa** e destravou o ambiente oficial:

- **Ambiente instala e roda**: `poetry env use python3.12 && poetry install`.
  A suíte oficial passa: **80 testes** (`poetry run pytest`), incluindo os novos
  testes do `RewardEngine`, da `LabResultQueue` e do horizonte de término.
- **`RewardEngine` reescrito** (`dengue_envs/core/reward.py`) em **3 camadas**
  claras e configuráveis:
  1. **Custo imediato** por ação (teste=1.0, epi=0.5, nada=0.1, confirm/discard=0.0).
  2. **Desfecho de decisão com delay** (`reward_delay_days`, default 5) para
     `confirm`/`discard`.
  3. **Placar final** no fim do episódio: `+final_correct_bonus` por acerto
     (equivale a `episize - misdiagnosed`) e `penalty_untested_misdiagnosed`
     (-10) por caso errado e nunca testado — **modelo do README**.
- **Bug do `discard` corrigido**: a decisão agora é avaliada pela **doença
  verdadeira** (acerto quando `true == OTHER`), não pelo `agent_diagnosis` que o
  `step` já havia mutado. Descartar um doente real virou falso negativo com
  penalidade mais pesada (`penalty_missed_case`, -30).
- **Pesos expostos** em `DengueDiagnosticsEnv.__init__` e na factory `make_env`
  (`reward_correct_decision`, `penalty_incorrect_decision`, `penalty_missed_case`,
  `final_correct_bonus`, `penalty_untested_misdiagnosed`) → ajustáveis por config.
- **Delay do laboratório implementado** (`dengue_envs/core/lab_queue.py`,
  `LabResultQueue`): ao pedir um teste a amostra é colhida hoje, mas o laudo
  (e a atualização de `testd`/`testc`/`agent_diagnosis`) só chega após
  `lab_delay_days` (default = `reward_delay_days`). No fim do episódio, os
  laudos pendentes são liberados (`flush`). `lab_delay_days=0` volta ao
  comportamento imediato.
- **Delay ligado no treino**: `agents/deepq/files/dqn_new_reward.py` agora usa
  `REWARD_DELAY_DAYS = 5` e `N_STEP = 5` (o lab delay acompanha automaticamente).
- **Término do episódio (`terminated`) finalizado**: o antigo `epilength + 10`
  arbitrário virou um horizonte derivado — `horizon = (epilength - 1) + settle_days`,
  com `settle_days = max(reward_delay, lab_delay)` por padrão. Ou seja, o episódio
  roda até o último dia com casos + o tempo necessário para os atrasos
  amadurecerem. `settle_days` é configurável.
- **README atualizado** com o modelo de recompensa, o delay de laboratório,
  observação (`tnot`), instalação (Python 3.12) e roadmap.

**P1 concluído.** Próxima prioridade real: **Fase 0** — congelar baseline
reprodutível (Random + Q-Learning) agora que a recompensa e o término fazem
sentido.

---

# ✅ Rodada — Baseline aleatório + `experiments/` (Fases 2 e 3 iniciadas)

Estruturação do agente **random** no novo modelo e criação do pipeline de
avaliação por YAML, com resultados em formato comum (comparável entre agentes).

### Estratégia `_old` mantida (nada apagado)

| Antes | Agora |
|-------|-------|
| `agents/random/agent_random.py` | `agents/random/agent_random_old.py` (legado) |
| `agents/random/random_agent.py` | `agents/random/random_agent_old.py` (legado) |
| `COMPARACAO.py` | `COMPARACAO_old.py` (import ajustado p/ `agent_random_old`) |

### Novo (contrato + agente + experimentos)

- **`agents/base.py`**: contrato `AgentRunner` (`evaluate(make_env, seeds, env_config)`)
  e o **esquema de resultados canônico** `RESULT_COLUMNS` (id + métricas de
  `get_episode_metrics`). Garante que TODOS os agentes gerem as mesmas colunas.
- **`agents/random/agent.py`** (`RandomAgentRunner`): baseline que roda sobre o
  MESMO env dos outros (bruto + `map_tensor` + `case_by_case`), amostrando o
  `action_space` discreto do wrapper. Reprodutível por seed (semeia o RNG global
  do numpy antes de criar o `World`).
- **`experiments/`**:
  - `configs/env/synthetic_default.yaml`, `configs/env/synthetic_large.yaml`
  - `configs/benchmark.yaml` (agentes + seeds + `env_config` + `output_dir`)
  - `evaluate.py` (CLI com `AGENT_REGISTRY`; salva `benchmark_raw/mean/std.csv`)
  - `README.md` (como rodar + esquema de saída)
- **Resultados** salvos em `results/baseline/` (pasta ignorada no git).
- **`clinical_specificity`** no env passou a aceitar lista (vinda do YAML), não só tupla.

### Como rodar (baselines)

```bash
poetry run python experiments/evaluate.py --config experiments/configs/benchmark.yaml
```

### Melhorias no benchmark e nas métricas (endurecimento)

- **Baseline clínico puro** (`agents/clinical/agent.py`, `ClinicalOnlyAgentRunner`):
  aceita sempre o diagnóstico clínico (ação "não fazer nada"), sem testes. É o
  **piso de referência**; a diferença de recompensa para ele é a **contribuição
  marginal** de cada agente. Registrado no `AGENT_REGISTRY` e no `benchmark.yaml`.
- **Base compartilhada `EpisodeRunner`** (`agents/base.py`): laço de avaliação
  padronizado (reprodutível por seed, mesmo env, métricas de episódio). `random`
  e `clinical` agora só definem `name` + `choose_action(env)`.
- **Acurácia multiclasse (3 classes)** adicionada ao lado da binária em
  `episode_metrics`. A binária (dengue vs. resto) é leniente e mascara a confusão
  chik↔outro; a multiclasse exige a classe exata. No baseline isso ficou nítido:
  o `random` tem acurácia binária até maior que o clínico, mas multiclasse menor.
- **Alinhamento por índice** em `env.get_episode_metrics`: verdade e predição são
  casadas por `case_id` (`real_cases.loc[obs_cases.index]`), não mais por posição.
  A função delega o cálculo para `dengue_envs.metrics.episode_metrics` (que valida
  tamanhos iguais), eliminando a implementação duplicada no env.
- **Recompensa como métrica primária de ranking** (`PRIMARY_METRIC` em
  `agents/base.py`): `evaluate.py` ordena por `Recompensa Total`, imprime o
  ranking e a contribuição marginal sobre o clínico; as demais métricas são
  diagnóstico secundário.

Resultado observado (10 seeds, env default): **clínico −613.7 > random −2260.0**
— confirmando que o aleatório fica **abaixo** do clínico puro.

### Renderização modular (Fase 1 concluída)

A renderização saiu do ambiente e virou um pacote isolado `dengue_envs/rendering/`:

- **Assets no lugar certo**: os PNGs foram movidos (`git mv`) de
  `dengue_envs/envs/` para `dengue_envs/rendering/assets/`. Ninguém mais os
  carrega via `os.path.dirname(__file__)` do env.
- **`rendering/assets.py`**: `ASSETS_DIR`, `asset_path(name)` e `load_image(name,
  size)` com cache (`lru_cache`) + mapa `ACTION_ICONS` (ação → ícone).
- **`rendering/sprites.py`**: `CaseSprite` / `CaseGroup` extraídos do env; o
  sprite troca de ícone via `ACTION_ICONS`, sem referência ao env.
- **`rendering/pygame_renderer.py`**: `PygameRenderer` dono da janela, das
  superfícies, dos grupos e do desenho (mapa + gráfico de recompensa + gráfico de
  acurácia + legenda). Gera os plots internamente.
- **Env agora delega**: `_render_init` cria o `PygameRenderer`; `render`,
  `_create_sprites` e `update_sprites` são finas delegações; `reset` chama
  `renderer.reset()`. Removidos `import os`/`import time` e todo o código pygame
  inline do env.
- **Gráfico de acurácia = multiclasse**: novo `multiclass_accuracy_history`
  (match exato das 3 classes) é o que alimenta o gráfico, no lugar da média
  binária dengue/chik (que mascarava a confusão chik↔outro).

Validação: 80 testes passando + smoke test headless (`SDL_VIDEODRIVER=dummy`)
rodando `step`/`render` sem abrir janela.

Próximo agente a migrar para este mesmo pipeline: **Q-Learning** (basta um novo
runner + registrá-lo no `AGENT_REGISTRY`).

---

## 1. Problemas que o código **já resolveu**

### P0 — corrigidos e testados (72 testes passando no ambiente oficial)

| Problema | Status | O que mudou |
|----------|--------|-------------|
| **Estado apagado a cada dia** | ✅ Resolvido | `_sync_obs_cases()` mantém testes, `agent_diagnosis` e `epiconf` entre timesteps |
| **Lab tests ignoravam doença real** | ✅ Resolvido | `_dengue_lab_test(case_id)` / `_chik_lab_test(case_id)` usam `real_cases` |
| **`_epi_confirm` quebrado** | ✅ Resolvido | Usa `self.dmap` / `self.cmap`; ação 2 chama o método de verdade |
| **`action_space` com IDs inválidos** | ✅ Resolvido | `Discrete(self.num_cases)` em vez de `2*episize` |
| **Comentário errado do discard** | ✅ Resolvido | Documentado como `agent_diagnosis = 2` (outro) |
| **`reset()` dessincronizado** | ✅ Resolvido | `_load_episode_state(start_day)` alinha `t`, `cases_t`, mapas |
| **Treino sem casos / análise OK** | ✅ Resolvido | `start_day=1` + `CaseByCaseWrapper` avança dias vazios |

### Melhorias adjacentes (não eram P0, mas já estão no código)

- `CaseByCaseWrapper` propaga `info` do env na troca de dia
- Coordenadas normalizadas `[0,1]` no wrapper
- Mapa em `uint8` para economizar RAM no replay buffer

---

## 2. Delay de recompensa — **implementado e completo** (atualizado nesta rodada)

> **Atualização:** o modelo de recompensa foi completado. As três lacunas
> listadas abaixo em "O que ainda NÃO está no delay" **foram implementadas** no
> `RewardEngine` (bônus final `episize - misdiagnosed` e penalidade -10 por caso
> não testado e mal diagnosticado). A subseção histórica é mantida para registro.

### O que **já existe** no ambiente

No `DengueDiagnosticsEnv`, o parâmetro `reward_delay_days` (default **5**) ativa este fluxo em `_calc_reward()`:

```289:352:dengue_envs/envs/dengue_diagnostics.py
    def _calc_reward(self, true, estimated, action, terminated=False):
        ...
        # Custo imediato: toda ação paga hoje
        immediate_reward -= self.costs[action_id]

        # Confirm (4) / Discard (5): agenda bônus/penalidade para t + reward_delay
        if delayed_reward_accum != 0:
            target_t = self.t + self.reward_delay
            self.pending_rewards[target_t] += delayed_reward_accum

        # Hoje: resgata recompensas que "venceram"
        matured_reward = self.pending_rewards.pop(self.t, 0.0)

        # Fim do episódio: liquida o que sobrou na fila
        if terminated:
            matured_reward += sum(self.pending_rewards.values())
```

Resumo do comportamento:

| Componente | Quando paga |
|------------|-------------|
| Custo de teste / epi / nada | **Imediato** (mesmo dia) |
| Confirm (+10) / Discard (-20) | **Atrasado** em `reward_delay_days` |
| Fila pendente no fim | **Liquida** no último step |

### O que **ainda NÃO** está no delay (README original) — ✅ em grande parte resolvido

- ✅ Bônus `episize - casos_errados` no fim do episódio (`final_correct_bonus`)
- ✅ Penalidade **-10** por caso não testado e diagnosticado errado
  (`penalty_untested_misdiagnosed`)
- ✅ Delay para resultados de **laboratório** (`LabResultQueue` / `lab_delay_days`):
  o laudo do exame agora respeita o atraso, além do confirm/discard.

### Atenção no script de treino

No `dqn_new_reward.py` **atual** (atualizado nesta rodada):

```python
REWARD_DELAY_DAYS = 5   # delay ligado no treino
N_STEP = 5              # >= delay para cobrir a janela no bootstrap do DQN
```

Ou seja: o mecanismo existe no env **e o experimento agora usa delay** (5 dias).
O README foi atualizado e não lista mais o delay como TODO.

### Bug do discard — ✅ CORRIGIDO nesta rodada

Antes, o `discard` era avaliado por `agent_diagnosis` **depois** de o `step()` já
ter setado `agent_diagnosis = 2`, tornando `discarded = 1 if agent_diagnosis == 0`
degenerado. Agora o `RewardEngine._decision_outcome` avalia o `discard` pela
**doença verdadeira**: acerto quando `true == OTHER`; descartar um doente real é
falso negativo e recebe `penalty_missed_case` (-30). Coberto por testes
(`test_discard_correct_when_true_other`, `test_discard_missed_real_case_is_heavily_penalized`).

---

## 3. O que **ainda falta** (fora do P0)

| Item | Prioridade | Status |
|------|------------|--------|
| Completar modelo de recompensa (README) | P1 | ✅ feito nesta rodada |
| Corrigir semântica de reward do discard | P1 | ✅ feito nesta rodada |
| README desatualizado | P2 | ✅ feito nesta rodada |
| Delay para resultados de laboratório | P1 | ✅ feito nesta rodada |
| `terminated = epilength + 10` arbitrário | P1 | ✅ feito nesta rodada (horizonte derivado) |
| Distribuições realistas (2016, Kriging) | P2 | ⬜ pendente |
| Unificar scripts de treino duplicados | P2 | ⬜ pendente |
| PPO integrado ao env | P2 | ⬜ pendente |
| `World` sem seed reprodutível | P2 | ⬜ pendente |

---

# Passo a passo para reorganizar o repositório (modular + multi-algoritmo + multi-distribuição)

Objetivo: **separar simulação, ambiente, wrappers, recompensa, agentes e experimentos**, para trocar distribuição epidemiológica e algoritmo RL sem reescrever código.

---

# ✅ Integração da Fase 1 — CONCLUÍDA (o que foi feito nesta rodada)

Até então a Fase 1 existia apenas como **esqueleto em paralelo** ao código legado
(módulos novos em `dengue_envs/core`, `wrappers`, `data`, mas o env e os scripts
de treino ainda usavam o caminho antigo). Esta rodada **ligou o núcleo ao caminho
de execução**. Nada foi apagado: todo código legado foi preservado com sufixo
`_old` como backup.

### Estratégia adotada: `[nome-atual]` = novo (integrado) · `[nome-atual]_old` = legado intacto

### 1. `DengueDiagnosticsEnv` agora DELEGA para o núcleo `dengue_envs/core`

Arquivo `dengue_envs/envs/dengue_diagnostics.py` reescrito para delegar
(backup fiel em `dengue_diagnostics_old.py`):

| Método do env | Agora delega para |
|---------------|-------------------|
| `_apply_clinical_uncertainty` | `ClinicalModel.apply_uncertainty` |
| `_dengue_lab_test` / `_chik_lab_test` | `ClinicalModel.dengue_lab_test` / `chik_lab_test` |
| `_update_case_status` | `core.clinical.update_case_status` |
| `_sync_obs_cases` | `core.case_store.sync_obs_cases` |
| `_epi_confirm` | `core.epi_confirm.epi_confirm` |
| `_calc_reward` | `core.reward.RewardEngine.compute` |

- Novo atributo `self.reward_engine` (`RewardEngine`) detém a fila de delay;
  `self.pending_rewards` virou **propriedade de leitura** apontando para o engine.
- Novo atributo `self.clinical_model` (`ClinicalModel`), **recriado no `reset()`**
  sempre que a especificidade é reamostrada.
- A **API pública e os nomes dos métodos foram mantidos** (compatibilidade total
  com os testes e com scripts existentes).

### 2. Scripts de treino/avaliação usam `dengue_envs.wrappers.make_env`

Todos deixaram de instanciar `DengueDiagnosticsEnv` + wrappers na mão e passaram
a chamar a **fábrica única** `make_env` (backup de cada um em `*_old.py`):

- `agents/deepq/files/dqn_new_reward.py`
- `agents/deepq/files/agent_train.py`
- `agents/deepq/files/watch_agent.py`
- `agents/deepq/files/watch_agent_large.py`
- `agents/deepq/files/confusion.py`

### 3. Cópia duplicada de wrappers foi deprecada

`agents/deepq/files/dengue_wrapper.py` virou um **shim** que reexporta os wrappers
canônicos de `dengue_envs.wrappers` e emite `DeprecationWarning`. A implementação
original ficou em `dengue_wrapper_old.py`. Fonte única da verdade agora é
`dengue_envs/wrappers/{map_tensor,case_by_case}.py`.

### 4. Validação

Suite de testes não pôde rodar neste shell (sem `poetry`/venv; `pygame` não tem
wheel para Python 3.14). Em vez disso, rodou-se um **script de verificação de
runtime** (pygame stubado) cobrindo os mesmos casos dos testes + o mecanismo de
delay: **19/19 checks OK** (persistência de estado, testes de lab condicionados,
epi confirm binário, término de episódio, agendamento de recompensa com delay=5,
pagamento imediato com delay=0). Recomenda-se rodar `poetry run pytest` no
ambiente oficial para confirmar os 31 testes.

### Arquivos `_old` criados (backup, podem ser removidos no futuro)

```
dengue_envs/envs/dengue_diagnostics_old.py
agents/deepq/files/dengue_wrapper_old.py
agents/deepq/files/dqn_new_reward_old.py
agents/deepq/files/agent_train_old.py
agents/deepq/files/watch_agent_old.py
agents/deepq/files/watch_agent_large_old.py
agents/deepq/files/confusion_old.py
```

### Mudanças de plano decorrentes desta rodada

- **Fase 1 deixa de ser "esqueleto" e passa a estar integrada** ao runtime. A
  próxima prioridade real é **P1 de recompensa** (semântica do discard + bônus
  final/penalidade por não testar + revisar `terminated = epilength + 10`),
  agora que há um `RewardEngine` único onde fazer isso sem tocar no laço do env.
- **Bug do discard segue preservado de propósito** (integração ≠ mudança de
  semântica): `RewardEngine.compute` replica `discarded = 1 if agent_diagnosis == 0`.
  Corrigir isso é o primeiro item da próxima fase.
- **Fase 0 (congelar baseline + limpar raiz)** continua pendente e sobe de
  prioridade: agora que o pipeline usa `make_env`, dá para gerar um
  `benchmark_baseline.csv` reprodutível antes de mexer na recompensa.
- Sugestão de faxina futura: mover os `*_old.py` para uma pasta `legacy/` ou
  removê-los após o `pytest` oficial passar, para não confundir os imports.

---

## Fase 0 — Preparação (1–2 dias)

1. **Congelar baseline**
   - Rode `pytest` e salve um `benchmark_baseline.csv` com Random + Q-Learning + DQN atual.
   - Isso vira referência para não quebrar comportamento na refatoração.

2. **Definir contratos**
   - Env bruto: `DengueDiagnosticsEnv` (Dict obs, Sequence action).
   - Env RL: `make_rl_env(config)` → `CaseByCaseWrapper(DengueWrapper(env))`.
   - Todo algoritmo consome o **mesmo** env RL.

3. **Limpar a raiz**
   ```
   results/          # CSVs, PNGs, logs (gitignore)
   notebooks/        # .ipynb
   experiments/        # scripts de treino/avaliação
   ```

---

## Fase 1 — Pacote `dengue_envs` (núcleo estável)

Estrutura alvo:

```
dengue_envs/
├── __init__.py              # register DengueDiag-v0
├── data/
│   ├── __init__.py
│   ├── base.py              # Protocol: WorldGenerator
│   ├── synthetic.py         # World atual (SIR + truncnorm) ← mover generator.py
│   ├── rio2016.py           # futuro: dados reais
│   └── kriging.py           # futuro: densidade espacial
├── core/
│   ├── case_store.py        # _sync_obs_cases, obs_cases
│   ├── clinical.py          # incerteza clínica, lab tests
│   ├── epi_confirm.py       # confirmação epidemiológica
│   └── reward.py            # RewardEngine + pending_rewards
├── envs/
│   └── dengue_diagnostics.py  # só orquestra step/reset/render
├── wrappers/
│   ├── map_tensor.py        # DengueWrapper
│   ├── case_by_case.py      # CaseByCaseWrapper
│   └── factory.py           # make_env(config)
├── rendering/
│   └── pygame_renderer.py
├── metrics/
│   └── episode_metrics.py   # get_episode_metrics, confusion map
└── tests/
```

**Passos concretos:**

1. Extrair `RewardEngine` de `dengue_diagnostics.py` → `core/reward.py`.
2. Mover `DengueWrapper` / `CaseByCaseWrapper` de `agents/deepq/files/` → `dengue_envs/wrappers/`.
3. Criar interface de distribuição:

```python
# dengue_envs/data/base.py
from typing import Protocol
import pandas as pd

class EpidemicGenerator(Protocol):
    def generate(self, seed: int | None) -> pd.DataFrame:
        """Retorna casedf com colunas t,x,y,disease,testd,testc,epiconf."""
        ...
```

4. Refatorar `World` para implementar `EpidemicGenerator`; depois adicionar `Rio2016Generator`, `KrigingGenerator` sem mudar o env.

---

## Fase 2 — Configuração por YAML (trocar distribuição sem código)

```
experiments/
├── configs/
│   ├── env/
│   │   ├── synthetic_default.yaml
│   │   ├── synthetic_large.yaml
│   │   └── rio2016.yaml          # futuro
│   ├── train/
│   │   ├── dqn_delay5.yaml
│   │   ├── dqn_no_delay.yaml
│   │   └── ppo.yaml
│   └── benchmark.yaml
├── train.py                      # CLI único
├── evaluate.py                   # ex-COMPARACAO.py
└── watch.py                      # visualização
```

Exemplo `synthetic_default.yaml`:

```yaml
env:
  generator: synthetic
  size: 400
  episize: 150
  epilength: 60
  start_day: 1
  reward_delay_days: 5
  clinical_specificity: [0.5, 0.95]

wrappers:
  - map_tensor
  - case_by_case

train:
  algorithm: dqn
  seed: 42
  epochs: 50
```

`train.py`:

```bash
poetry run python experiments/train.py --config experiments/configs/train/dqn_delay5.yaml
```

Trocar distribuição = trocar YAML (`generator: rio2016`).

---

## Fase 3 — Pasta `agents/` (um algoritmo = um módulo)

Estrutura alvo:

```
agents/
├── __init__.py
├── base.py                    # Protocol AgentRunner
├── random/
│   └── agent.py               # AleatoryAgent (unificar agent_random.py)
├── qlearning/
│   └── agent.py
├── dqn/
│   ├── network.py             # DengueNet
│   ├── policy.py              # build_dqn_policy
│   └── train.py               # lógica Tianshou (sem make_env duplicado)
├── ppo/
│   ├── network.py
│   └── train.py               # adaptar CleanRL ao CaseByCaseWrapper
└── baselines/
    ├── test_all.py            # testar todos os casos
    └── clinical_only.py       # aceitar sempre clínica
```

**Contrato comum** (`agents/base.py`):

```python
class AgentRunner(Protocol):
    def train(self, env_factory, config) -> Path: ...  # retorna checkpoint
    def evaluate(self, env_factory, checkpoint) -> dict: ...
```

Cada algoritmo implementa isso; `experiments/train.py` só faz:

```python
runner = get_runner(config["train"]["algorithm"])  # "dqn" | "ppo" | "random"
runner.train(make_env_factory(config), config)
```

**O que fazer com o que você já tem:**

| Arquivo atual | Destino |
|---------------|---------|
| `agents/deepq/files/dengue_wrapper.py` | `dengue_envs/wrappers/` |
| `agents/deepq/files/fcn_network.py` | `agents/dqn/network.py` |
| `agents/deepq/files/agent_train.py` + `dqn_new_reward.py` | `agents/dqn/train.py` (um só) |
| `agents/random/agent_random.py` | `agents/random/agent.py` |
| `agents/random/random_agent.py` | remover ou marcar deprecated |
| `agents/qlearning/qlearning_agent.py` | `agents/qlearning/agent.py` |
| `agents/ppo/ppo.py` | `agents/ppo/train.py` (reescrever para o env) |
| `COMPARACAO.py` | `experiments/evaluate.py` |

---

## Fase 4 — Distribuições mais realistas (seu plano de pesquisa)

Ordem sugerida:

1. **`SyntheticGenerator`** (atual) — baseline reprodutível.
2. **`CalibratedSyntheticGenerator`** — SIR com `R0`, centros e raios calibrados a um surto real.
3. **`Rio2016Generator`** — casos reais ou semi-reais (TODO do README).
4. **`KrigingDensityGenerator`** — superfície de densidade para `epi_confirm` (TODO Kriging).

Cada generator só produz `casedf`; o env não muda.

```python
# dengue_envs/envs/dengue_diagnostics.py (futuro)
def __init__(self, generator: EpidemicGenerator, ...):
    self.generator = generator
    self.world = generator.build_world(...)  # ou injeta casedf direto
```

---

## Fase 5 — Treino multi-algoritmo

Fluxo unificado:

```mermaid
flowchart LR
    CFG[YAML config]
    FACTORY[make_env_factory]
    ENV[dengue_envs + wrappers]
    ALG[agents/dqn | ppo | qlearning | random]
    RES[results/models + tensorboard]

    CFG --> FACTORY --> ENV --> ALG --> RES
```

Checklist por algoritmo:

| Algoritmo | Env necessário | Status hoje |
|-----------|----------------|-------------|
| Random | `CaseByCaseWrapper` ou env bruto | ✅ Funciona |
| Q-Learning | Env bruto (ação por dia) | ⚠️ Baseline fraco |
| DQN | `DengueWrapper` + `CaseByCaseWrapper` | ✅ Tianshou |
| PPO | Mesmo wrapper do DQN | ❌ Template CartPole |

Para PPO: reutilizar `DengueNet` ou uma política mais leve; GAE sobre episódios completos no wrapper.

---

## Fase 6 — Qualidade e documentação

1. Atualizar `README.md` (obs `tnot`, delay, `start_day`, instalação).
2. `pyproject.toml`: incluir `experiments` como scripts ou pacote opcional.
3. CI: `pytest dengue_envs/tests agents/*/tests`.
4. `.gitignore`: `results/`, `logs/`, `*.pth`, CSVs gerados.

---

## Ordem de execução recomendada (cronograma prático)

| Semana | Entrega |
|--------|---------|
| 1 | Fase 0 + mover wrappers para `dengue_envs/wrappers/` + `factory.py` |
| 2 | Extrair `RewardEngine` + completar delay (README) + YAML básico |
| 3 | Unificar DQN + `experiments/train.py` + `evaluate.py` |
| 4 | Interface `EpidemicGenerator` + primeiro generator calibrado |
| 5 | PPO no mesmo pipeline + baselines (`test_all`, `clinical_only`) |
| 6 | Rio2016 / Kriging + benchmark multi-distribuição |

---

## Resposta direta às suas três perguntas

1. **Problemas resolvidos?** Sim — todos os **P0** de correção do ambiente (estado, lab tests, epi confirm, action space, reset/start_day, wrapper de treino) **e o P1 de recompensa** (semântica do discard + modelo completo do README). **72 testes passando** no ambiente oficial (Python 3.12).

2. **Delay implementado?** **Sim, completo.** Custo imediato + fila `pending_rewards` (delay para confirm/discard) + **delay do laboratório** (`LabResultQueue`, o laudo do exame chega após `lab_delay_days`) + **placar final** (bônus `episize - misdiagnosed` e -10 por caso não testado e errado). No treino (`dqn_new_reward.py`) está com **`REWARD_DELAY_DAYS = 5`**, ou seja, ligado (o lab delay acompanha automaticamente).

3. **Plano modular?** Seguir as 6 fases acima: núcleo `dengue_envs` + generators plugáveis + wrappers únicos + `agents/*` por algoritmo + `experiments/` com YAML. Isso permite treinar DQN, PPO, Q-Learning e baselines na mesma distribuição (ou trocar distribuição só mudando config).

Se quiser, no próximo passo posso **executar a Fase 1** (mover wrappers, criar `factory.py` e `RewardEngine`) diretamente no repositório.