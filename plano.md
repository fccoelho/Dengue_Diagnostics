# Confirmação do estado atual + plano de reorganização

## 1. Problemas que o código **já resolveu**

### P0 — corrigidos e testados (31 testes passando)

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

## 2. Delay de recompensa — **implementado parcialmente**

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

### O que **ainda NÃO** está no delay (README original)

- Bônus `episize - casos_errados` no fim do episódio
- Penalidade **-10** por caso não testado e diagnosticado errado
- Delay para resultados de **laboratório** (hoje só confirm/discard têm delay)

### Atenção no script de treino

No seu `dqn_new_reward.py` atual:

```python
REWARD_DELAY_DAYS = 0   # delay desligado no treino
```

Ou seja: o **mecanismo existe no env**, mas o experimento que você está rodando **não usa delay** (`0` dias). Para treinar com delay, use por exemplo `REWARD_DELAY_DAYS = 5` e ajuste `N_STEP >= delay` no DQN.

O README ainda lista delay como TODO — vale atualizar depois da refatoração.

### Bug residual na lógica de discard (P1, não P0)

Em `_calc_reward`, o discard avalia `agent_diagnosis` **depois** de `step()` já ter setado `agent_diagnosis = 2`. A lógica binária `discarded = 1 if agent_diagnosis == 0 else 0` fica degenerada. Funciona com delay, mas a semântica de recompensa precisa de um `RewardEngine` dedicado (próxima fase).

---

## 3. O que **ainda falta** (fora do P0)

| Item | Prioridade |
|------|------------|
| Completar modelo de recompensa (README) | P1 |
| Corrigir semântica de reward do discard | P1 |
| `terminated = epilength + 10` arbitrário | P1 |
| Distribuições realistas (2016, Kriging) | P2 |
| Unificar scripts de treino duplicados | P2 |
| PPO integrado ao env | P2 |
| README desatualizado | P2 |
| `World` sem seed reprodutível | P2 |

---

# Passo a passo para reorganizar o repositório (modular + multi-algoritmo + multi-distribuição)

Objetivo: **separar simulação, ambiente, wrappers, recompensa, agentes e experimentos**, para trocar distribuição epidemiológica e algoritmo RL sem reescrever código.

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

1. **Problemas resolvidos?** Sim — todos os **P0** de correção do ambiente (estado, lab tests, epi confirm, action space, reset/start_day, wrapper de treino). **31 testes passando.**

2. **Delay implementado?** **Parcialmente no env** (custo imediato + fila `pending_rewards` para confirm/discard). **Não está completo** conforme o README (bônus final, penalidade por não testar). No treino atual (`dqn_new_reward.py`) está com **`REWARD_DELAY_DAYS = 0`**, ou seja, desligado na prática.

3. **Plano modular?** Seguir as 6 fases acima: núcleo `dengue_envs` + generators plugáveis + wrappers únicos + `agents/*` por algoritmo + `experiments/` com YAML. Isso permite treinar DQN, PPO, Q-Learning e baselines na mesma distribuição (ou trocar distribuição só mudando config).

Se quiser, no próximo passo posso **executar a Fase 1** (mover wrappers, criar `factory.py` e `RewardEngine`) diretamente no repositório.