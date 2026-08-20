# Refatoração do ambiente — documento de handoff

> **STATUS: implementada e avaliada.** As seções §5 e §6 descrevem o
> diagnóstico e o plano como escritos *antes* da implementação; o que foi feito
> está em **§10**, e os resultados em **§11**–**§15**.
>
> ⚠️ **Leia §15 antes de §12, §13 e §14.** Rodar a melhor configuração com um
> seed diferente derrubou o resultado de −81,90 para −3272,25 — um espalhamento
> **maior que qualquer efeito de parâmetro que eu havia medido**. As conclusões
> daquelas três seções foram tiradas de uma execução cada e estão **retiradas**.
> §10 (a refatoração do ambiente) não é afetada: é código, com testes.
>
> **Resumo dos resultados, em ordem:** a atribuição por caso (§10) levou o DQN
> de −3717 para −1703; ajustar `n_step` de 5 para 30 (§12) levou de −1703 para
> **−81,90**, à frente do `confirmall` e com o menor desvio entre seeds de
> todos os agentes. Restam +1037 pontos até o `testtwice` (+955), e eles têm
> uma causa identificada: o agente aprendeu a usar `epi_confirm` (evidência
> barata, retorno em ~6 passos) mas **nunca** exames de laboratório (retorno em
> ~30 passos). Mas isso não sobreviveu à verificação: encurtar a cadeia do
> laudo não fez os exames aparecerem (§14), e trocar apenas o seed moveu o
> resultado em 3190 pontos (§15). **Nenhum efeito de hiperparâmetro medido
> nesta sessão é distinguível da variância de treino.**
>
> **O achado robusto sobre o agente é comportamental, não numérico:** em todas
> as seis execuções — quatro configurações, dois seeds — o DQN faz **zero
> exames de laboratório**. Os números de recompensa oscilam 3190 pontos; esse
> comportamento não oscila nada.
>
> **Melhor artefato: `results/dqn_v13/policy_best.pth`** — −81,90 ± 120,1 sobre
> 10 seeds de avaliação, acurácia multiclasse 0,699, 3º/7. É um checkpoint bom
> e utilizável, mas **não reproduzível a partir da config** (§15.1), então não
> serve como evidência sobre nenhum parâmetro.
>
> **Antes de qualquer nova conclusão sobre o DQN: protocolo multi-seed (§15.4).**

Este documento é autossuficiente: reúne o que foi implementado, os números
medidos e o diagnóstico da causa raiz que impede o DQN de aprender no
ambiente v3, para que a refatoração possa ser retomada em outro momento sem
precisar reconstruir o raciocínio da sessão em que foi feito o diagnóstico.

Histórico cronológico completo (todas as rodadas, uma por uma): `ACHADOS.md`.
Este documento aqui é o **resumo executivo + plano técnico**, não o diário.

---

## 1. Resumo em uma tela

- O ambiente evoluiu de um problema binário (dengue × chik) para um problema
  de 3 classes com investigação sequencial genuína (dengue × chik × não-arbovirose).
- Ao longo dessa evolução, **o DQN parou de superar políticas fixas simples**.
  Ele ia bem (posição 2/4–2/6, ~50% do caminho entre o piso e o teto) enquanto
  existia um atalho de **uma única ação** com recompensa positiva. Quando esse
  atalho desapareceu (v3), nenhuma configuração testada (7 versões, currículo,
  ajuste de γ, rede reequilibrada) chegou perto da política fixa.
- **Causa raiz identificada e medida:** o ambiente atribui a recompensa **por
  dia**, não por caso. Um dia pode ter dezenas de casos; a decisão do agente
  sobre UM caso específico retorna recompensa **zero**, e o resultado agregado
  de todos os casos do dia cai inteiro no último passo. Medido: **90% da
  variância da recompensa de um passo vêm de decisões tomadas dias antes,
  sobre outros casos** (§5).
- **A correção** — separar "aplicar a ação de um caso" de "avançar o dia" no
  ambiente bruto, e usar `reward_delay_days: 0` (mantendo o atraso do **laudo**,
  `lab_delay_days`, que é o que sustenta a investigação sequencial) — foi
  **implementada**: plano em §6, execução em §10, resultado em §11. Efeito
  medido no sinal: passos com sinal de 2,3% → 100%, desvio por passo de
  31,0 → 10,2, soma do episódio inalterada. Um vazamento de recompensa
  pré-existente foi descoberto no caminho (§10.2).
- **Resultado dos treinos que testaram a correção (§11 e §12).** A correção
  de §10 levou o DQN de −3717 para −1703 (§11) — melhora real, mas ainda
  abaixo do `confirmall` e com zero exames. O diagnóstico seguinte (§11.4)
  apontou que o valor de investigar precisa propagar por ~30 transições
  pertencentes a **outros casos**, e que `n_step: 5` não alcançava isso.
  Confirmado em §12: `n_step: 30` levou o agente a **−81,90** (3º/7), com
  acurácia multiclasse 0,552 → 0,699 e sensibilidade para dengue empatando com
  o `testtwice` — **sem gastar um exame**. O que falta para fechar a lacuna
  está isolado em §12.4.
- **Tudo mais no ambiente está validado e funcionando:** o espaço de 7 ações
  conclusivas, o modelo clínico Beta, RT-PCR, o retorno do caso após o laudo, a
  máscara de decisão forçada, as features de contexto, o rebalanceamento da
  rede. Nenhum desses precisa ser refeito — só a atribuição de recompensa por
  caso, que é ortogonal a todo o resto.

---

## 2. Como reproduzir o estado atual

```bash
poetry run pytest                          # 161 testes, todos passando

# Ambiente v4 (atual): 3 classes + atribuição de recompensa POR CASO.
poetry run python agents/deepq/train.py --config experiments/configs/train/dqn_v12.yaml

# Ambiente v3 (agregado por dia) — mantido para reproduzir o histórico:
poetry run python experiments/evaluate.py --config experiments/configs/benchmark_v11.yaml

# Diagnóstico de casos órfãos / coerência (scripts ad-hoc usados nesta sessão,
# não fazem parte do pacote — ver §8 para recriá-los se necessário).
```

> ℹ️ O vazamento corrigido em §10.2 atingia **o sinal de treino, não as
> métricas reportadas** — estas vêm de `total_reward`, acumulado dentro do
> ambiente cru. Os números históricos de `ACHADOS.md` e `RESULTS.md` estão
> corretos; verificado re-rodando os baselines (idênticos, §10.2).

---

## 3. Linha do tempo de desempenho do DQN

**Métrica normalizada:** `(recompensa_dqn − recompensa_clínico) / (recompensa_melhor_política_fixa − recompensa_clínico)`.
`0,0` = empata com o clínico puro (piso); `1,0` = empata com a melhor política
fixa (teto). Usar recompensa bruta entre experimentos não é válido — as
escalas de custo/recompensa mudaram várias vezes.

| Experimento | Agentes no benchmark | DQN normalizado | Posição | Recompensa bruta (dp) | Acurácia | Exames |
|---|---:|---:|---|---:|---:|---:|
| v1 (só clínico/random/dqn) | 3 | — (sem 2º piso) | 1/3 | −395,9 (827) | 0,753 | 103,4 |
| v1 + baseline `testall` | 4 | **0,46** | 2/4 | −395,9 (827) | 0,753 | 103,4 |
| v2 (recompensa recalibrada) | 4 | — | 1/4 | +68,6 (732) | 0,699 | 7,7 |
| **v2_final** (PCR + retorno do caso + exame caro) | 6 | **0,51** | **2/6** | **+330,3 (751)** | 0,784 | 98,6 |
| v3 = `dqn_v8` (3 classes, sem currículo) | 7 | 0,23 | 4/7 | −2910,3 (580) | 0,601 | 75,9 |
| `dqn_v9` (currículo, γ=0,999) | 7 | 0,26 | 4/7 | −3711,7 (281) | 0,565 | 0,0 |
| `dqn_v10` (currículo, γ=0,99) | 7 | 0,26 | 4/7 | −2751,9 (333) | 0,521 | 202,8 |
| `dqn_v11` (rede reequilibrada) | 7 | 0,06 | 5/7 | −3717,0 (176) | 0,534 | 1,0 |

**Leitura:** o pico real foi `v2_final` (+330,3), mas mesmo ali o DQN nunca
superou a melhor política fixa (`testonce`, +739,2) — chegou a ~51% do caminho.
No v3, a distância cresceu e nenhuma correção estrutural (currículo, γ, rede)
recuperou o patamar anterior; cada tentativa converge para uma **degeneração
diferente** (não testar nada / testar sem concluir / concluir ignorando a
evidência).

### 3.1 Por que a queda coincide exatamente com a introdução da 3ª classe

| Política de **uma única decisão por caso** | v2 | v3 |
|---|---:|---:|
| `confirmall` (aceita o palpite clínico) | **+128,8** | **−1365,2** |
| `clinical` (não faz nada) | −95,2 | −4036,5 |

No v2 existia um atalho de uma ação com recompensa **positiva** — o agente
podia aprender "aceite o palpite" e já ficar acima do zero, refinando a partir
daí. No v3, a queda da acurácia clínica (0,690 → 0,571, efeito da 3ª classe)
tornou **toda** política de uma ação profundamente negativa. A única política
positiva (`testtwice`, +955,0) exige encadear 3 decisões corretas — testar
dengue, testar chik, concluir a classe certa — com o retorno de cada uma
maturando dias depois. O agente nunca descobriu essa cadeia.

---

## 4. O que está implementado e validado (não mexer sem necessidade)

Cada item abaixo foi implementado, medido e coberto por teste nesta ou em
sessões anteriores. `ACHADOS.md` tem o racional completo de cada um; aqui só a
referência rápida.

| Feature | Onde | Parâmetro/flag | Status |
|---|---|---|---|
| RT-PCR (sens. 0,95 / espec. 0,98 / 2% inconclusivo) | `core/clinical.py` | `lab_sensitivity`, `lab_specificity`, `lab_inconclusive_prob` | ✅ validado; **valores ainda sem citação da literatura** |
| Retorno do caso após o laudo | `envs/dengue_diagnostics.py` (`_schedule_revisit`, `take_pending_revisits`) | `max_case_revisits` (padrão 2) | ✅ |
| Casos não-arbovirose (3ª classe) | `data/generator.py` (`_build_other_cases`), `core/clinical.py` | `other_prevalence`, `other_recognition_prob` | ✅ espacialmente uniforme, proporcional no tempo |
| Modelo clínico Beta (sensib./especif. por médico) | `core/clinical.py` (`ClinicalQualitySampler`) | `clinical_quality: low\|medium\|high` ou dict explícito | ✅ 3 níveis + concentração configurável |
| Exame caro (cria escassez) | `envs/dengue_diagnostics.py` | `test_cost` (calibrado por varredura empírica) | ✅ v2: 7,0 · v3: 4,0 (recalibrar se `episize`/`other_prevalence` mudar) |
| Ações conclusivas por classe (substituíram confirm/discard) | `core/reward.py`, `envs/dengue_diagnostics.py` | ações 4/5/6 = `conclude_dengue/chik/other` | ✅ ver §7.1 — mudança de espaço de ações, `Discrete(7)` |
| Penalidade por caso não concluído | `core/reward.py` (`_terminal_score`) | `penalty_unresolved` | ✅ evita que "não decidir" seja refúgio |
| Reward shaping (crédito por investigar) | `core/reward.py` | `shaping_conclude_bonus` | ✅ desligado por padrão |
| Máscara de decisão forçada | `envs/dengue_diagnostics.py` (`action_mask`), `wrappers/case_by_case.py` | `force_decision_after_tests` | ✅ dispara após os 2 exames **ou** na última apresentação do caso (ver §7.2) |
| Features de contexto (evidência sobre o médico + atributos do caso) | `envs/dengue_diagnostics.py` (`clinical_evidence`, `case_features`) | `context_features: true` | ✅ vetor de 14 posições (2 + 12) |
| Rede: mapa projetado, contexto expandido | `agents/deepq/network.py` | `map_proj_dim`, `context_out_dim` (padrão 128/128) | ✅ ver §7.3 — contexto foi de 1,3% para 40% da entrada da cabeça |
| Escala de recompensa no treino | `wrappers/reward_scale.py` | `reward_scale` (só no YAML de treino) | ✅ não altera a política ótima; benchmark roda sem escala |
| Warm-start entre treinos (currículo) | `agents/deepq/train.py` | `train.init_from: <checkpoint>.pth` | ✅ funciona (compatibilidade de arquitetura confirmada), mas não resolveu o gap no v3 |

---

## 5. Causa raiz: recompensa atribuída por DIA, não por CASO

### 5.1 O mecanismo exato

`CaseByCaseWrapper.step()` (`dengue_envs/wrappers/case_by_case.py`):

```python
def step(self, action: int):
    if self.current_case[0] == 0 and not self.active_cases:
        return self._make_obs(), 0.0, True, False, {}
    self.pending_actions.append((int(self.current_case[0]), int(action)))
    return self._next_case()
```

`_next_case()` só chama `self.env.step(action_tuple)` (o ambiente bruto)
quando o iterador de casos do dia se esgota (`StopIteration`). Até lá, cada
`step()` do wrapper devolve **recompensa 0.0**. Quando o dia acaba, o ambiente
bruto processa **todas** as ações do dia de uma vez
(`DengueDiagnosticsEnv.step`, que recebe `action` como uma tupla de
`(case_id, action_id)` — ver `_calc_reward` → `RewardEngine.compute`) e
devolve **um único escalar** para o dia inteiro, que vira a recompensa do
**último** `step()` do wrapper naquele dia.

Ou seja: se o dia tem 50 casos, os primeiros 49 `step()` retornam 0,0 e o
50º recebe a soma de custos + desfechos atrasados que maturaram naquele dia,
somados sobre os 50 casos.

### 5.2 Medição do efeito

Rodando uma política fixa (`confirmall`) e decompondo a recompensa de cada
passo em "custo da ação atual" vs. "o resto":

| Componente | Média | Desvio-padrão |
|---|---:|---:|
| Custo da ação atual (o que o agente escolheu agora) | −1,34 | 1,89 |
| Desfechos que maturaram (de decisões de **outros** casos, dias antes) | −0,89 | **17,07** |

**90% da variância** (17,07 / (1,89+17,07)) do retorno de um passo vem de
decisões tomadas sobre **outros casos**, dias antes. O agente decide sobre o
caso X e o TD-target que ele recebe é dominado por ruído de decisões alheias.

Adicionalmente, a maior parte dos passos tem recompensa **exatamente zero**:
em uma medição de 370 passos (episódio completo), apenas 7 (1,9%) tiveram
recompensa não-nula. Isso por si só não impede o aprendizado (é esparsidade
normal em RL), mas combinado com o ponto acima — quando a recompensa não é
zero, ela é ruído de outros casos — o sinal fica irrecuperável pela função Q.

### 5.3 Por que isso explica todas as tentativas de correção que falharam

| Tentativa | Por que não resolveu, à luz desta causa |
|---|---|
| γ (desconto) | Ajusta o **horizonte** sobre o qual o ruído é integrado, não a fonte do ruído |
| Currículo (v2 → v3) | Transfere pesos, mas o problema de atribuição existe igualmente nas duas fases (o v2 tinha o atalho de 1 ação que mascarava o problema — §3.1) |
| Rede reequilibrada (mapa/contexto) | Corrige o que a rede **consegue ver**, não o que ela **aprende a partir do sinal recebido** |
| `penalty_unresolved` / `shaping_conclude_bonus` | Mudam a *magnitude* do sinal, mas ele continua sendo entregue no dia errado, misturado a outros casos |

---

## 6. Plano técnico da refatoração (não implementado)

**Objetivo:** cada `step()` do `CaseByCaseWrapper` deve devolver a recompensa
**daquela decisão específica**, não uma fatia arbitrária do agregado do dia.

### 6.1 Mudança central

Separar, no ambiente bruto (`DengueDiagnosticsEnv`), duas operações hoje
acopladas dentro de um único `step()`:

1. **Aplicar a ação de UM caso** — atualiza `obs_cases`, agenda exame/laudo,
   calcula o custo imediato + desfecho de decisão **daquele caso**.
2. **Avançar o dia** — recalcula `obs_cases` a partir do mundo, libera laudos
   que maturaram (`lab_queue`), atualiza `t`.

Hoje o `step()` faz as duas coisas juntas, recebendo a lista de ações do dia
inteiro. A refatoração propõe expor a operação (1) de forma que o wrapper
possa chamá-la **por caso**, e reservar (2) para quando o dia realmente vira.

### 6.2 `reward_delay_days: 0`

Com `reward_delay_days` zerado, o desfecho de uma decisão conclusiva
(`conclude_dengue/chik/other`) deixa de ser agendado para `t + delay` e passa
a ser resolvido **no mesmo step** em que a ação foi tomada — que já será o
step daquele caso específico, não do dia. Isso elimina a fila
`pending_rewards` como fonte de mistura entre casos (hoje ela já agrega por
dia: `self.pending_rewards[target_t] += delayed_reward_accum`, sobre **todos**
os casos que maturam naquele dia).

**Manter `lab_delay_days` como está.** É o atraso do laudo (resultado do
exame), que é a peça que sustenta a investigação sequencial (testar hoje,
decidir quando o resultado voltar — via `_schedule_revisit`). Esse mecanismo
não depende de `reward_delay_days` e deve continuar existindo.

### 6.3 O que muda em cada arquivo

- **`dengue_envs/core/reward.py`**: `RewardEngine.compute()` hoje recebe uma
  lista de `(case_id, action_id)` e devolve um escalar agregado. Precisa
  aceitar (ou ser chamado com) uma ação por vez, retornando a recompensa
  daquele caso isoladamente. A fila `pending_rewards` só continua fazendo
  sentido se `reward_delay_days > 0` for mantido como opção (mas o modo
  recomendado para treino é 0).
- **`dengue_envs/envs/dengue_diagnostics.py`**: `step()` precisa de um modo
  (ou um método novo) que processe uma ação de caso isoladamente e retorne a
  recompensa correspondente, sem esperar o dia fechar. O avanço de dia
  (`self.t += 1`, `self.world.get_maps_up_to_t`, `_sync_obs_cases`,
  `_flush_lab_results`/`_apply_matured_lab_results`) precisa ser reorganizado
  para acontecer uma vez por dia, não uma vez por ação.
- **`dengue_envs/wrappers/case_by_case.py`**: `_next_case()` hoje só invoca
  `self.env.step()` no `StopIteration`. Passaria a invocar a nova operação
  por-caso a cada `step()` do wrapper, e reservar a chamada de fim-de-dia só
  para o avanço temporal (sem recompensa agregada de decisão associada).
- **Placar final (`_terminal_score`)**: continua fazendo sentido como está —
  é intencionalmente pago uma vez, no fim do episódio, e não faz parte do
  problema de atribuição por dia.

### 6.4 Riscos e o que precisa ser revalidado depois

- **Todos os checkpoints e benchmarks atuais ficam obsoletos** — a semântica
  da recompensa por passo muda, então nada é comparável diretamente com o
  que está em `ACHADOS.md`/`RESULTS.md` sem re-rodar.
- **Os testes que verificam a fila `pending_rewards`**
  (`dengue_envs/tests/test_core_reward.py`, ex.:
  `test_conclude_dengue_correct_is_delayed`,
  `test_terminated_settles_pending_queue`) assumem `reward_delay_days > 0` e
  precisam ser mantidos cobrindo esse modo (não remover — é um modo válido,
  só não o recomendado para treino de RL).
- **Conferir que a recompensa TOTAL do episódio não muda** entre o modo
  antigo (delay agregado por dia) e o novo (delay zero, por caso) — a soma
  deve ser idêntica para a mesma sequência de ações; só a **distribuição
  temporal** do crédito muda. Vale um teste de regressão específico para isso.
- **O `RewardScaleWrapper`** (`reward_scale`) continua válido e provavelmente
  ainda necessário — a escala por passo pode mudar com a nova atribuição,
  então recalibrar o fator (hoje 0,05) medindo o novo desvio-padrão por passo.
- **γ e horizonte de desconto** também merecem nova calibração: com
  recompensa por caso (não mais por dia), o "horizonte" relevante deixa de
  ser o comprimento do episódio inteiro e passa a ser mais próximo do atraso
  do laudo (`lab_delay_days`) — provavelmente um γ bem mais baixo volta a
  fazer sentido.

---

## 7. Três bugs estruturais já corrigidos nesta sessão (para não reintroduzir)

### 7.1 Ações conclusivas por classe (substituíram `confirm`/`discard`)

Antes: `confirm` (ação 4) só podia aceitar o `agent_diagnosis` corrente;
trocar dengue↔chik exigia sempre um exame. `discard` (ação 5) e `confirm`
concorriam de forma degenerada quando o diagnóstico era "outro" (mesma
recompensa se certo, `discard` só perdia se errado) — na prática `discard`
nunca era usado.

Agora: `conclude_dengue` (4) / `conclude_chik` (5) / `conclude_other` (6)
alegam a classe diretamente, independente do `agent_diagnosis`. Regra de
recompensa em `RewardEngine._decision_outcome`:
`claimed = action_id - CONCLUDE_DENGUE`; acerto → `reward_correct_decision`;
alegar `OTHER` num caso real → `penalty_missed_case` (pior erro); qualquer
outro erro → `penalty_incorrect_decision`.

**Validado:** um classificador simples (sklearn) usando as ações novas para
concluir direto sem testar rende +131,4 contra −3823,6 de confirmar cego —
confirma que a expressividade nova tem valor real quando bem usada.

### 7.2 Dois becos sem saída que geravam casos "órfãos"

Medido no `dqn_v10`: 35,7% dos casos terminavam o episódio **sem nenhuma ação
conclusiva**, pagando `penalty_unresolved` (−10) cada — 47% do déficit de
recompensa. Duas causas:

1. **`epi_confirm` (ação 2) não agendava revisita.** Adicionava evidência
   epidemiológica ao caso, mas o caso nunca voltava ao agente para usá-la —
   37% dos órfãos. Corrigido: `epi_confirm` agora chama `_schedule_revisit`,
   igual aos exames.
2. **A máscara de decisão forçada só disparava com os DOIS exames feitos.**
   Se o agente nunca testasse (ou testasse só uma vez), o caso esgotava o
   orçamento de revisitas (`max_case_revisits`) e desaparecia da fila sem
   jamais ser forçado a concluir — 52% dos órfãos. Corrigido: `action_mask`
   agora também força conclusão na **última apresentação** do caso
   (`self.revisit_counts.get(case_id, 0) >= self.max_case_revisits`), não só
   quando os dois exames voltaram.

Resultado das duas correções: órfãos caíram de 35,7% para 3,8% (os
remanescentes são `nada` explícito, comportamento legítimo).

### 7.3 A rede ignorava o diagnóstico do próprio caso

Mesmo com os órfãos corrigidos, a recompensa não melhorou como esperado — a
causa era **qualidade**, não cobertura: reavaliando o checkpoint antigo sob o
ambiente corrigido, a acurácia **caiu** (0,626 → 0,521), porque o agente
passou a concluir sempre (não mais órfão), mas errando.

Medição de sensibilidade dos Q-values do modelo treinado:

| Perturbação | \|ΔQ\| | Muda a ação escolhida? |
|---|---:|---|
| Diagnóstico do próprio caso (vetor de contexto) | 0,0 – 0,63 | **Não** |
| Ruído aleatório no mapa | 207 – 731 | **Sim** |

Causa: a cabeça da rede recebia 2304 dimensões de mapa (achatado) contra 32
de contexto — 96% × 1,3% da capacidade de entrada. Consequência medida:
48,6% das conclusões contrariavam a evidência do próprio exame, acertando
apenas 20,7% (contra 78,6% quando coerentes). Corrigido em
`agents/deepq/network.py`: o mapa passa por uma projeção linear
(`map_proj_dim=128`) antes de entrar na cabeça, e o encoder de contexto
cresceu para `context_out_dim=128` — o contexto foi de 1,3% para 40% da
entrada.

**Nota importante:** o problema em 7.3 foi introduzido pela própria mudança
em 7.1 — o `confirm` antigo era coerente com o diagnóstico *por construção*;
as ações por classe deram liberdade para o agente discordar do médico (o
objetivo), mas isso significa que a coerência deixou de ser automática e
precisa ser **aprendida**, o que exige capacidade de rede suficiente no ramo
certo.

**Resultado líquido dos três fixes:** nenhum deles, isolado ou em conjunto,
fechou a lacuna até a política fixa (§3). Isso é o que aponta para §5/§6 como
a causa remanescente.

---

## 8. Inventário de arquivos

### 8.1 Ambientes (`experiments/configs/env/`)

| Arquivo | O que é |
|---|---|
| `synthetic_default.yaml` | Ambiente original, 2 classes, recompensa v1 |
| `synthetic_v2.yaml` | + RT-PCR, retorno do caso, exame caro (`test_cost: 7`) |
| `synthetic_v3.yaml` | + `other_prevalence: 0,25` (3 classes), `test_cost: 4`, `force_decision_after_tests: true`, `penalty_unresolved`, `shaping_conclude_bonus`, `context_features: true` — **ambiente-alvo atual** |
| `clinical_low/medium/high.yaml` | Nível clínico fixo (Beta), sem sorteio por episódio |
| `kriging_rio.yaml`, `rio2016.yaml` | Geradores espaciais alternativos (dados reais do Rio) |

### 8.2 Treinos DQN (`experiments/configs/train/dqn_v*.yaml`)

| Config | Ambiente | Nota |
|---|---|---|
| `dqn_v2*.yaml` | synthetic_v2 | níveis clínicos fixos |
| `dqn_v4.yaml` | synthetic_v2 | **melhor resultado histórico** (+330,3) |
| `dqn_v5.yaml`–`dqn_v8.yaml` | synthetic_v3 | evolução incremental (γ, case_features) — ver `ACHADOS.md` §7–§8 |
| `dqn_v9_phase{1,2}.yaml` | v2 → v3 | currículo, γ=0,999 (fase 2 colapsou) |
| `dqn_v10_phase{1,2}.yaml` | v2 → v3 | currículo, γ=0,99 (corrigiu colapso, não fechou o gap) |
| `dqn_v11.yaml` | synthetic_v3 direto | rede reequilibrada, sem currículo (pior que v10) |

### 8.3 Checkpoints (`results/dqn_v*/policy_best.pth`)

Todos listados existem em disco (ver saída de `ls` no início desta sessão).
**Nenhum é compatível entre si em arquitetura** salvo dentro da mesma família
de mudanças (ex.: v9/v10 usam a rede antiga de 32 dims de contexto; v11 usa a
rede reequilibrada de 128/128 — carregar um checkpoint v9/v10 na arquitetura
v11 dá erro de shape, como já ocorreu nesta sessão).

### 8.4 Benchmarks (`experiments/configs/benchmark_v*.yaml`)

Cada `benchmark_vN.yaml` aponta para o checkpoint `dqn_vN` correspondente e
roda os mesmos 10 seeds (`[100,150,200,250,300,350,400,450,500,550]`) contra
os baselines fixos (`clinical`, `random`, `confirmall`, `testonce`,
`testtwice`, `qlearning`). Resultados salvos em `results/baseline_vN/`.

### 8.5 Scripts de diagnóstico (não fazem parte do pacote)

Os scripts ad-hoc usados para medir órfãos, coerência e sensibilidade da rede
ficaram no diretório de scratch da sessão (fora do repositório) e não foram
promovidos a testes formais. Se a refatoração precisar deles, os trechos
relevantes estão citados em §5 e §7 — são curtos e fáceis de recriar:

- **Órfãos por ação**: percorrer um episódio guardando a última ação aplicada
  a cada `case_id`; ao final, filtrar `case_id not in env.unwrapped.finalized_cases`
  e contar a última ação desses.
- **Coerência**: nas conclusões (`action in (4,5,6)`), comparar
  `claimed = action - 4` com `obs_cases.loc[case_id, "agent_diagnosis"]`
  (que já reflete os laudos); separar em coerente/discorda e medir acerto
  contra `real_cases`.
- **Sensibilidade da rede**: perturbar `obs["context"][2:5]` (one-hot do
  diagnóstico) e `obs["map"]` (ruído aditivo) separadamente, medir
  `|Q_perturbado − Q_base|` e se `argmax` muda.

---

## 9. Perguntas em aberto para quem retomar

1. **A refatoração de §6 deve ser feita antes ou depois de decidir o
   enquadramento do artigo?** Se o resultado "RL não supera heurísticas
   simples, e aqui está o porquê" já for suficiente para a entrega atual, a
   refatoração pode ficar para uma versão futura do trabalho.
2. **Vale a pena testar `reward_delay_days: 0` isoladamente**, sem a
   refatoração completa de separar ação-por-caso de avanço-de-dia? Não
   resolveria o problema (a agregação por dia continuaria existindo dentro de
   um único `env.step()`), mas é um experimento de ~5 minutos de código que
   pode dar sinal adicional antes de investir na refatoração maior.
3. **Papel do `lab_delay_days`** na versão refatorada: hoje ele e
   `reward_delay_days` são parâmetros independentes; confirmar que a
   refatoração não precisa deles ficarem sincronizados de alguma forma nova.
4. **Custo de manter os dois modos** (recompensa por dia vs. por caso) no
   mesmo `RewardEngine`, para não quebrar retrocompatibilidade com os
   experimentos já documentados em `ACHADOS.md`/`RESULTS.md`.

---

## 10. A refatoração, como foi feita

### 10.1 O que mudou, arquivo por arquivo

**`dengue_envs/core/reward.py`** — `compute()` foi dividido em duas metades que
estavam fundidas, e reescrito como composição delas (nenhuma lógica duplicada):

| Método | Escopo | Conteúdo |
|---|---|---|
| `case_reward(case_id, action_id, t, ...)` | **caso** | custo da ação + desfecho da decisão (imediato se `reward_delay_days == 0`, senão vai para a fila) |
| `settle_day(t, ..., terminated, concluded)` | **dia** | desfechos que venceram hoje + placar final |
| `compute(action, ...)` | dia inteiro | `sum(case_reward) + settle_day` — modo histórico, preservado |

**`dengue_envs/envs/dengue_diagnostics.py`** — o corpo do laço de ações foi
extraído para `_apply_action_effects(case_id, action_id)`, usado tanto pelo
`step()` agregado quanto pelo novo:

```python
def apply_case_action(self, case_id, action_id) -> float:
    """Aplica UMA ação e devolve a recompensa atribuível a ela."""
```

Não avança o tempo. O avanço do dia continua sendo `step(())` — tupla vazia,
que já existia e era usada em `_advance_empty_days`. Foi essa descoberta que
reduziu bastante o escopo: não foi preciso reescrever o `step()`.

Também foi acrescentado `self._day_actions`, que acumula as ações do dia
apenas para o renderer (que espera receber a tupla completa do dia).

**`dengue_envs/wrappers/case_by_case.py`** — flag `per_case_reward`
(padrão `False`):

```python
def step(self, action: int):
    ...
    if self.per_case_reward:
        case_r = self.unwrapped.apply_case_action(case_id, int(action))
        obs, day_r, term, trunc, info = self._next_case()
        return obs, case_r + day_r, term, trunc, info
    self.pending_actions.append((case_id, int(action)))   # modo histórico
    return self._next_case()
```

**`dengue_envs/wrappers/factory.py`** — repassa `config["per_case_reward"]`.

**`experiments/configs/env/synthetic_v4.yaml`** — v3 + `per_case_reward: true`
+ `reward_delay_days: 0` + `lab_delay_days: 5` (explícito).

### 10.2 Bug adicional encontrado durante a validação

O teste de invariância acusou uma divergência de 250 pontos entre modos que
deveriam somar igual. A investigação isolou a variável e mostrou que **não era
o modo** (796 = 796, 1046 = 1046) e sim o `reward_delay` — o que não deveria
alterar a soma do episódio.

A causa era um vazamento pré-existente no wrapper:

```python
# ANTES (bug)
while not self.active_cases:
    obs_tensor, reward, terminated, truncated, info = self.env.step(tuple())
    #             ^^^^^^ descartado
```

`_advance_empty_days` avançava dias sem casos ativos e **jogava fora a
recompensa devolvida** — justamente nos dias em que vencem os desfechos
agendados. Quanto maior o `reward_delay_days`, mais se perdia: medido em um
único episódio com atraso de 5 dias, **114 de 444 pontos (~25%)** sumiam
silenciosamente.

Corrigido: `_advance_empty_days` agora devolve a recompensa acumulada, somada
em `_next_case`. No `reset()` o retorno é descartado de propósito (a fila está
vazia ali, não há o que perder) — com comentário explicando.

**Impacto (corrigido após medição — a primeira versão desta seção errava
o alvo).** O vazamento atingia **o sinal de treino, não as métricas
reportadas**. São dois caminhos distintos:

- **Métrica do benchmark** (`Recompensa Total`): vem de `self.total_reward`,
  acumulado *dentro* do ambiente cru
  (`dengue_diagnostics.py:409`, lido em `:1177`). O wrapper descartar o valor
  devolvido não afeta essa contabilidade. **Os números históricos de
  `ACHADOS.md` e `RESULTS.md` estão corretos.**
- **Sinal visto pelo agente durante o treino**: vem do retorno de `step()`
  atravessando a cadeia de wrappers — exatamente o que se perdia.

Verificação empírica: no config real do benchmark (v3, atraso 5), **406 de
1016 pontos (40%) do retorno passam por dias sem casos ativos** — era isso que
o treino não via. E os baselines re-rodados no v4 (pós-correção) vieram
**bit-a-bit idênticos** aos do v11 (pré-correção): `testtwice` 955,00 em
ambos, `confirmall` −1365,20 em ambos, e assim por diante — confirmando que a
métrica reportada nunca dependeu do trecho com bug.

A leitura correta, portanto, é *pior* para o diagnóstico de §5, não melhor: o
DQN não só recebia o crédito no passo errado (§5.1) como **nunca recebia 40%
dele**.

### 10.3 Validação

Suíte: **161 testes passando** (154 anteriores + 7 novos em
`dengue_envs/tests/test_per_case_reward.py`).

| Teste | O que trava |
|---|---|
| `test_total_reward_is_identical_between_modes` | a soma do episódio não muda entre modos |
| `test_total_reward_identical_for_several_policies` | idem, para 6 políticas diferentes |
| `test_total_reward_identical_with_reward_delay` | idem, com atraso > 0 |
| `test_reward_delay_does_not_change_episode_total` | **regressão do vazamento** de §10.2 |
| `test_aggregated_mode_leaves_most_steps_at_zero` | caracteriza o problema original |
| `test_per_case_mode_pays_on_the_decision_step` | o crédito chega no passo certo |
| `test_per_case_reward_matches_decision_outcome` | concluir certo rende positivo naquele passo |

Nota sobre tolerância: os dois modos somam as mesmas parcelas em **ordem
diferente**, então a igualdade é exata em teoria e aproximada em ponto
flutuante (divergências observadas na casa de 1e-14). Os testes usam
`pytest.approx` com `abs=1e-6`.

Nota sobre `settle_days`: ele é derivado de `max(reward_delay, lab_delay)` e
define o `horizon`. Ao comparar diferentes `reward_delay_days`, é preciso
**fixar `settle_days` explicitamente**, senão o episódio muda de tamanho e a
comparação deixa de ser válida — foi o que fez a primeira versão do teste de
regressão falhar por um motivo legítimo.

### 10.4 Efeito medido

Mesma política, mesmas seeds, v3 (agregado) vs v4 (por caso):

| Métrica | v3 (agregado por dia) | v4 (por caso) |
|---|---:|---:|
| Passos com recompensa **zero** | **97,7%** | **0,0%** |
| Desvio-padrão por passo | 31,04 | **10,19** (−67%) |
| Faixa por passo | [−380, +338] | **[−28, +46]** |
| Soma do episódio | 1119,0 | **1119,0** (idêntica ✓) |

O sinal passou de chegar em 2,3% dos passos para chegar em 100% deles, com um
terço da dispersão — e sem alterar a economia do problema, como o teste de
invariância garante.

### 10.5 Recalibração aplicada

- **`reward_scale`: 0,05 → 0,10.** O alvo é desvio por passo ≈ 1,0; com o
  desvio caindo de 31,0 para 10,2, a escala que atinge esse alvo mudou
  (`1/10,19 ≈ 0,10`).
- **γ mantido em 0,99.** Com crédito por caso, o horizonte relevante deixa de
  ser o episódio inteiro (~373 passos) e passa a ser o tempo até o laudo voltar
  — ~30 passos (5 dias × ~6 casos/dia). γ=0,99 dá horizonte efetivo de 100
  passos e desconto de 0,74 em 30 passos, cobrindo isso com folga. (γ=0,95
  daria 0,21 em 30 passos, apertado demais.)

### 10.6 Estado atual

Treino `dqn_v12` concluído (20 épocas × 5000 passos) e avaliado em
`benchmark_v12.yaml`. Resultado em §11.


---

## 11. Resultado do `dqn_v12` — a correção ajudou, mas não resolveu

**Veredito curto:** a refatoração de §10 melhorou muito o DQN (+2.014 pontos,
54% do déficit) mas **não** produziu um agente competitivo. Ele continua
perdendo para políticas fixas triviais, e colapsa na mesma política degenerada
de sempre: **nunca testar**.

### 11.1 Números

Benchmark `benchmark_v12.yaml`, ambiente v4, 10 seeds:

| # | Agente | Recompensa Total | Acurácia multiclasse | Testes |
|---|---|---:|---:|---:|
| 1 | `testtwice` | **+955,00** | 0,938 | 746,4 |
| 2 | `testonce` | +247,40 | 0,756 | 373,2 |
| 3 | `confirmall` | −1365,20 | 0,571 | 0,0 |
| 4 | **`dqn` (v12)** | **−1703,30** | 0,552 | **0,0** |
| 5 | `qlearning` | −2915,76 | 0,605 | 173,9 |
| 6 | `clinical` | −4036,52 | 0,571 | 0,0 |
| 7 | `random` | −5348,71 | 0,444 | 163,5 |

Progresso real sobre o v11 (mesmo agente, ambiente anterior):

| | v11 (v3, agregado) | v12 (v4, por caso) |
|---|---:|---:|
| Recompensa Total | −3716,97 | **−1703,30** (+2013,67) |
| Acurácia multiclasse | 0,534 | 0,552 |
| Posição no ranking | 5º de 7 | 4º de 7 |

A melhora é substancial e consistente com a hipótese de §5 — o crédito por
caso *é* um sinal melhor. Mas o agente segue **abaixo do `confirmall`**, que é
a política trivial "conclua tudo imediatamente, nunca teste".

### 11.2 A política aprendida

1858 decisões em 3 episódios (`policy_best.pth`, ε=0):

| Ação | Frequência |
|---|---:|
| `epi_confirm` | 39,7% |
| concluir CHIK | 31,5% |
| concluir DENGUE | 28,8% |
| testar dengue / testar chik | **0,0%** |
| concluir OUTRO | **0,0%** |
| nada | 0,0% |

Dois fatos que definem o fracasso:

1. **Zero exames.** O agente nunca compra informação. Isso reduz o problema a
   "adivinhar a partir do palpite clínico", que é exatamente o `confirmall` —
   e explica a acurácia de 0,552 contra 0,938 do `testtwice`.
2. **Zero conclusões "OUTRO"**, embora 25% dos casos sejam não-arbovirose
   (`other_prevalence: 0.25`). A terceira classe — a que motivou o v3 e
   iniciou toda a queda de desempenho (§3.1) — continua invisível para ele.

O `epi_confirm` em 39,7% é o que o torna *pior* que o `confirmall`: agenda
revisita, custa, e não resolve nada. É procrastinação com custo.

### 11.3 Curva de treino

Recompensa de teste (escala bruta = escalada × 10, `reward_scale: 0.10`):

| Época | 0 | 2 | 4 | 9 | 14 | 19 | 20 |
|---|---:|---:|---:|---:|---:|---:|---:|
| bruta | −3819 | **−1669** | −5163 | −2310 | −2372 | −1814 | −1735 |

O melhor checkpoint é da **época 2** (de 20). Depois disso, 90.000 passos de
treino não produziram ganho — oscilação entre −1700 e −3200 sem tendência. O
benchmark (−1703,30) bate com o `test_reward` do melhor checkpoint (−1669),
o que valida a leitura da escala.

### 11.4 O que isso diz

A hipótese de §5 estava **certa mas incompleta**. O ruído de atribuição era
real, foi medido (§10.4) e removido, e o agente melhorou 54% do déficit. Mas
sobrou uma barreira independente, e os dados desta rodada já a identificam.

**Não é a economia.** A suspeita óbvia seria que investigar deixou de
compensar. O benchmark refuta: `testtwice` (+955) supera `confirmall` (−1365)
por **2320 pontos** no v4. Investigar não só compensa como é, de longe, a
melhor coisa a fazer — e paga *mais* que na varredura original do `test_cost`
(que estimava +338 para "2 exames"), porque as correções de §7 melhoraram o
retorno das políticas que investigam. O ótimo está lá, intacto e generoso; o
agente é que não chega nele.

**É a estrutura do backup de Bellman sob casos intercalados.** Verificável no
código (`reward.py`, `case_reward`): para uma ação de exame,

```python
reward = -self.costs[action_id]     # -test_cost
...
outcome = self._decision_outcome(action_id, true_disease)
if outcome is None:                 # None para toda acao NAO conclusiva
    return reward                   # <- exame sempre devolve so o custo
```

Um exame **nunca** carrega sinal positivo no próprio passo — por construção,
e corretamente, já que seu valor é informacional. O benefício aparece quando o
caso volta com o laudo e é concluído certo, ~5 dias depois. E aqui está o
problema: sob `per_case_reward`, esse benefício é atribuído ao passo da
*conclusão*, que fica **~30 passos adiante** (5 dias × ~6 casos/dia).

Só que os ~30 passos no meio pertencem a **outros casos**. O backup de Bellman
encadeia `s_t → s_{t+1}`, e `s_{t+1}` é o estado de outro paciente. Para que
`Q(caso não testado, testar)` cresça, o valor precisa propagar para trás por
uma cadeia de ~30 transições que **não têm relação causal com o caso testado**.

Foi exatamente isso que a refatoração de §10 **não** tocou. Ela corrigiu o
*ruído* da atribuição (o crédito chegava embaralhado com o de dezenas de
casos); não corrigiu a *desconexão* entre a ação de investigar e o seu retorno.
Do ponto de vista do agente, pedir exame é uma ação que só custa.

Isso também explica o `epi_confirm` em 39,7%: entre "pagar por informação cujo
retorno o backup não alcança" e "empurrar o caso para frente", a segunda é
localmente melhor. É procrastinação racional dado o sinal que ele recebe.

Candidatos para a próxima rodada, em ordem de promessa:

1. **Fechar a cadeia dentro do próprio caso.** Fazer o desfecho da conclusão
   creditar também o passo do exame que o viabilizou — uma atribuição
   *retroativa por caso*, análoga ao que §10 fez por passo. É a correção que
   ataca o mecanismo descrito acima diretamente.
2. **Bônus informacional imediato** no passo do exame (recompensa intrínseca
   por redução de incerteza). Mais simples, mas é *reward shaping*: muda o
   ótimo se mal calibrado, e precisa de prova de invariância como a de §10.3.
3. **Episódios por caso.** Reformular o MDP para que um episódio seja *um
   caso*, eliminando a intercalação. Mudança grande, mas remove a causa em vez
   de compensá-la.
4. **`n_step` maior.** Hoje `n_step: 5`, contra uma cadeia de ~30. Aumentar
   para ~30 é uma linha de YAML e testa a hipótese barato — **começar por
   aqui**, antes de mexer em ambiente.

### 11.5 Reprodução

```bash
.venv/Scripts/python.exe -m agents.deepq.train --config experiments/configs/train/dqn_v12.yaml
.venv/Scripts/python.exe -m experiments.evaluate --config experiments/configs/benchmark_v12.yaml
```


---

## 12. `dqn_v13` — o melhor agente até agora

> ⚠️ **A explicação mecanística de §12.3 foi testada em §14 e não se
> sustentou.** Os *números* desta seção seguem válidos (o `dqn_v13` é o
> melhor agente medido); a *causa* que atribuí ao resultado, não. Leia §14
> antes de usar §12.3 como base para qualquer decisão.

**Veredito:** a hipótese 4 de §11.4 se confirmou, e com margem larga. Trocar
`n_step: 5` por `n_step: 30` — **uma linha de YAML**, ambiente idêntico — levou
o DQN de −1703,30 para **−81,90** e o fez ultrapassar o `confirmall`. Mas ele
continua fazendo **zero exames de laboratório**, e a razão é mecânica e
mensurável.

### 12.1 Números

| # | Agente | Recompensa Total | ± | Acurácia multiclasse | Exames |
|---|---|---:|---:|---:|---:|
| 1 | `testtwice` | +955,00 | 177,6 | 0,938 | 746,4 |
| 2 | `testonce` | +247,40 | 208,4 | 0,756 | 373,2 |
| 3 | **`dqn` (v13)** | **−81,90** | **120,1** | 0,699 | **0,0** |
| 4 | `confirmall` | −1365,20 | 1389,4 | 0,571 | 0,0 |
| 5 | `qlearning` | −2915,76 | 444,0 | 0,605 | 173,9 |
| 6 | `clinical` | −4036,52 | 165,3 | 0,571 | 0,0 |
| 7 | `random` | −5348,71 | 297,5 | 0,444 | 163,5 |

Trajetória do mesmo agente ao longo das três correções:

| | v11 (v3, agregado) | v12 (v4, por caso) | v13 (v4, `n_step` 30) |
|---|---:|---:|---:|
| Recompensa Total | −3716,97 | −1703,30 | **−81,90** |
| Acurácia multiclasse | 0,534 | 0,552 | **0,699** |
| Sensibilidade (dengue) | — | 0,720 | **0,927** |
| Posição | 5º/7 | 4º/7 | **3º/7** |

Dois detalhes que merecem nota: a **sensibilidade para dengue (0,927) empata
com a do `testtwice` (0,929)** — sem gastar um único exame; e o **desvio entre
seeds (120,1) é o menor de todos os agentes**, contra 1389,4 do `confirmall`.
A política é boa *e* estável.

### 12.2 O que ele descobriu: evidência barata em vez de exame caro

Distribuição de ações (3304 decisões, ε=0):

| Ação | v12 | v13 |
|---|---:|---:|
| `epi_confirm` | 39,7% | **66,1%** |
| concluir CHIK | 31,5% | 17,0% |
| concluir DENGUE | 28,8% | 16,9% |
| exames (dengue+chik) | 0,0% | **0,0%** |
| concluir OUTRO | 0,0% | **0,0%** |

O `epi_confirm` custa **0,5 contra 4,0 do exame** (`DEFAULT_COSTS` em
`reward.py:65`, com `test_cost` sobrescrevendo as duas primeiras posições) e
devolve evidência baseada na **densidade local de casos**. Como o ambiente
posiciona dengue em (100,100) e chikungunya em (300,300), com raio 90, a
posição espacial é fortemente informativa para discriminar as duas
arboviroses.

O agente não está procrastinando — como interpretei no v12. Ele encontrou um
**canal de evidência 8× mais barato** que resolve a maior parte do problema, e
o explora sistematicamente. É uma estratégia economicamente sensata.

### 12.3 Por que a cadeia curta foi aprendida e a longa não

O mecanismo é direto e explica os dois resultados de uma vez:

| Ação | Custo | Quando o retorno chega | Em passos | `n_step: 30` alcança? |
|---|---:|---|---:|---|
| `epi_confirm` | 0,5 | revisita **no dia seguinte** (`_schedule_revisit` → `pending_revisit`) | ~6 | **sim, com folga** |
| exame de lab | 4,0 | laudo em `lab_delay_days: 5` | ~30 | **fica na borda** |

`n_step: 30` abriu exatamente a cadeia curta e deixou a longa no limite. O
retorno de *n* passos usado no alvo de TD passou a **conter** a conclusão que o
`epi_confirm` viabiliza, mas mal encosta na que o exame viabiliza.

Isso valida a leitura estrutural de §11.4 (o backup de Bellman precisa
atravessar ~30 transições de *outros* casos) e ainda a torna quantitativa: o
que importa é a razão entre o comprimento da cadeia e o `n_step`.

### 12.4 O teto desta estratégia

A acurácia multiclasse trava em 0,699 (contra 0,938 do `testtwice`), e o
motivo está no desenho do ambiente: os casos "outro" são **espacialmente
uniformes** por construção ("doença febril de fundo não é agrupada por vetor",
comentário em `synthetic_v3.yaml`). Evidência epidemiológica é densidade
local — logo, **não discrimina a terceira classe**. Para ela só o laboratório
resolve.

Daí os **0% de conclusões "OUTRO"**: a única evidência que o agente aprendeu a
usar é cega justamente para a classe que ele não conclui. Os +1037 pontos que
o separam do `testtwice` são, essencialmente, os 25% de casos não-arbovirose.

### 12.5 Próximo teste (em execução)

`dqn_v14` = v13 com `n_step: 60`, ambiente e todo o resto idênticos, para
cobrir a cadeia do laboratório com a mesma folga que o v13 deu à do
`epi_confirm`. Se os exames aparecerem, a leitura de §12.3 fica confirmada e o
caminho é calibrar `n_step` ao maior atraso do ambiente. Se o treino
desestabilizar sem produzir exames (risco real: *n* grande aproxima Monte
Carlo, com mais variância e viés off-policy), então comprimento de cadeia não
é a única causa, e o próximo passo passa a ser a **atribuição retroativa por
caso** (§11.4, item 1) — creditar o passo do exame quando a conclusão que ele
viabilizou é paga.

### 12.6 Reprodução

```bash
.venv/Scripts/python.exe -m agents.deepq.train --config experiments/configs/train/dqn_v13.yaml
.venv/Scripts/python.exe -m experiments.evaluate --config experiments/configs/benchmark_v13.yaml
```


---

## 13. `dqn_v14` — `n_step` não é um botão monotônico

> ⚠️ **Conclusão retirada em §15.** A queda documentada aqui (−3766) é da
> mesma ordem do espalhamento causado apenas pelo seed (−3190), então não
> é atribuível ao `n_step`. Os números seguem válidos como registro.

**Veredito:** dobrar `n_step` de 30 para 60, tudo o mais idêntico ao v13,
**destruiu** o aprendizado. O risco antecipado em §12.5 se materializou.

### 13.1 Números

| Config | `n_step` | Recompensa Total | Posição |
|---|---:|---:|---|
| `dqn_v12` | 5 | −1703,30 | 4º/7 |
| **`dqn_v13`** | **30** | **−81,90** | **3º/7** |
| `dqn_v14` | 60 | **−3765,62** | 5º/7 |

O v14 ficou **abaixo do `qlearning`** (−2915,76) e apenas 271 pontos acima do
`clinical` puro (−4036,52) — ou seja, praticamente não aprendeu nada.

A curva de treino confirma: o melhor checkpoint das 20 épocas foi **−3732,4**,
contra **−176,6** do v13. O valor inicial, antes de qualquer treino, era
−3819,0. Em 100.000 passos o v14 avançou 87 pontos.

### 13.2 A política colapsou

| Ação | v13 (`n_step` 30) | v14 (`n_step` 60) |
|---|---:|---:|
| `epi_confirm` | 66,1% | 39,5% |
| **`nada`** | 0,0% | **40,7%** |
| concluir CHIK | 17,0% | 19,8% |
| concluir DENGUE | 16,9% | **0,0%** |
| exames | 0,0% | 0,0% |

O v14 não só deixou de usar o `epi_confirm` produtivamente como **perdeu a
capacidade de concluir dengue** e passou a não agir em 40,7% das decisões — o
mesmo refúgio de inação que o `penalty_unresolved` foi introduzido para
combater (§ do `synthetic_v3.yaml`).

### 13.3 Leitura

`n_step: 30` não é "ainda insuficiente para a cadeia do laboratório"; é um
**ponto ótimo estreito**. A explicação está no trade-off clássico do retorno de
*n* passos: *n* maior encurta a distância que o crédito precisa percorrer por
bootstrapping, mas o retorno se aproxima de Monte Carlo — mais variância — e,
em DQN sem correção de importância, mais **viés off-policy**, já que as *n*
transições vêm de uma política antiga. Em *n* = 60 os dois custos superaram o
ganho.

Isso **não refuta** o mecanismo de §12.3; refuta apenas a ideia de que ele se
resolve aumentando `n_step`. A cadeia do laboratório (~30 passos) continua fora
de alcance, e agora sabe-se que não é por esse caminho que se chega nela.

### 13.4 O que fica

Dos quatro candidatos de §11.4, o item 4 (`n_step`) está **esgotado**: rendeu o
maior ganho isolado da sessão (+1621 pontos, v12→v13) e tem ótimo em ~30.

O teste seguinte, em execução, ataca o mecanismo pelo lado do ambiente em vez
do algoritmo: `dqn_v15` roda o **v13 exatamente como está** (`n_step: 30`) num
ambiente `synthetic_v5.yaml` idêntico ao v4 salvo por `lab_delay_days: 5 → 2`.
Isso encurta a cadeia do laboratório de ~30 para ~12 passos — a mesma folga que
o `epi_confirm` (~6) tem hoje.

É um **experimento de diagnóstico, não uma proposta de design**: se os exames
aparecerem, a razão cadeia/`n_step` de §12.3 está confirmada como a causa, e a
conclusão prática é que o atraso de 5 dias do laudo está no limite do que este
algoritmo consegue creditar. Se continuarem em zero, a explicação de §12.3 está
errada e a causa é outra — e o próximo passo passa a ser a **atribuição
retroativa por caso** (§11.4, item 1), que é mudança de código, não de config.

### 13.5 Reprodução

```bash
.venv/Scripts/python.exe -m agents.deepq.train --config experiments/configs/train/dqn_v14.yaml
.venv/Scripts/python.exe -m experiments.evaluate --config experiments/configs/benchmark_v14.yaml
```


---

## 14. `dqn_v15` — a explicação de §12.3 não se sustentou

> ⚠️ §14.4 anunciava a verificação de seed como "em curso". Ela foi feita:
> resultado e consequências em **§15**.

**Veredito:** encurtar a cadeia do laboratório **não** fez o agente pedir
exames. A hipótese mecanística de §12.3 está **refutada**, e o que aparece no
lugar dela é um problema metodológico que afeta a leitura de §12 e §13.

### 14.1 O experimento

`synthetic_v5.yaml` = `synthetic_v4.yaml` com `lab_delay_days: 5 → 2`, treinado
com o `dqn_v13` **sem nenhuma outra alteração** (`n_step: 30`). Isso encurta a
cadeia do laudo de ~30 para ~12 passos — dentro da mesma folga que o
`epi_confirm` (~6 passos) tem hoje. Previsão de §12.3: os exames deveriam
aparecer.

### 14.2 O que aconteceu

| Agente | v4 (`lab_delay` 5) | v5 (`lab_delay` 2) |
|---|---:|---:|
| `testtwice` | +955,00 | +936,20 |
| `testonce` | +247,40 | +247,40 |
| `confirmall` | −1365,20 | −1365,20 |
| `clinical` | −4036,52 | −4036,52 |
| **`dqn`** | **−81,90** | **−3400,55** |

Ações (ε=0): **49,4% `nada`**, 26,4% `epi_confirm`, 13,2% concluir chik, 11,0%
concluir dengue — e **0,0% de exames**, exatamente como antes.

Dois pontos tornam o resultado difícil de explicar pelo ambiente:

1. **As políticas fixas praticamente não mudaram** (`confirmall` e `clinical`
   idênticas até o centavo; `testtwice` perde 18,8 pontos pelo horizonte 3 dias
   mais curto). O ambiente v5 é essencialmente tão difícil quanto o v4.
2. **Só o DQN despencou** — 3319 pontos, do 3º para o 5º lugar, colapsando no
   mesmo refúgio de inação do v14.

O confundidor conhecido é pequeno: `settle_days = max(reward_delay, lab_delay)`
cai de 5 para 2, encurtando o `horizon` de 64 para 61 dias (~5%). É o que
explica os 18,8 pontos do `testtwice`; não explica 3319 do DQN.

### 14.3 O problema metodológico que isso expõe

Quatro configurações, **todas com `seed: 42`**, uma única execução cada:

| Config | Mudança em relação à anterior | Resultado |
|---|---|---:|
| `dqn_v12` | ponto de partida (`n_step` 5) | −1703,30 |
| `dqn_v13` | `n_step` 5 → 30 | **−81,90** |
| `dqn_v14` | `n_step` 30 → 60 | −3765,62 |
| `dqn_v15` | `lab_delay` 5 → 2 (`n_step` 30) | −3400,55 |

Atribuí cada salto à mudança de configuração. Mas o v15 mostra que uma
alteração de ambiente que **mal move as políticas fixas** derruba o DQN em 3319
pontos — o que é forte evidência de que boa parte desse espalhamento é
**variância de treino**, não efeito do parâmetro.

Com uma execução por configuração não há como separar as duas coisas. A
conclusão de §12 ("`n_step: 30` é a causa do salto") e a de §13 ("`n_step` tem
ótimo estreito em ~30") foram tiradas de *n* = 1 e **não estão estabelecidas**.

O que **continua válido**, porque foi medido sobre 10 seeds de avaliação:
o checkpoint `dqn_v13/policy_best.pth` obtém −81,90 ± 120,1, com acurácia
multiclasse 0,699 e o menor desvio entre seeds de todos os agentes. Esse
*agente* é bom. O que não se sustenta é a explicação de *por que* ele surgiu.

### 14.4 Verificação em curso

`dqn_v13_seed43.yaml` — o `dqn_v13` **exatamente como está**, trocando apenas
`seed: 42 → 43`. É a pergunta mínima: o −81,90 reproduz?

- **Se reproduzir** (mesma ordem de grandeza), o efeito do `n_step` é real, §12
  se sustenta, e §14.2 vira uma pergunta específica sobre o `lab_delay`.
- **Se não reproduzir**, o espalhamento de §14.3 é variância de treino, e a
  conclusão correta da sessão passa a ser: *o DQN neste ambiente é instável ao
  ponto de uma execução isolada não ser evidência*. Nesse caso o próximo passo
  não é mais nenhum ajuste de hiperparâmetro, e sim **rodar cada configuração
  com ≥3 seeds** antes de qualquer nova conclusão — e atacar a instabilidade em
  si (buffer maior, `lr` menor, Double DQN) antes de voltar à questão dos
  exames.

### 14.5 Reprodução

```bash
.venv/Scripts/python.exe -m agents.deepq.train --config experiments/configs/train/dqn_v15.yaml
.venv/Scripts/python.exe -m experiments.evaluate --config experiments/configs/benchmark_v15.yaml
```


---

## 15. Verificação de seed — as conclusões de §12 e §13 não se sustentam

**Veredito:** o `dqn_v13` rodado com `seed: 43`, **sem nenhuma outra
alteração**, dá −3272,25 em vez de −81,90. O melhor resultado da sessão era
sorte de seed.

### 15.1 O número

| | `dqn_v13` seed 42 | `dqn_v13` seed 43 |
|---|---:|---:|
| Recompensa Total | **−81,90** ± 120,1 | **−3272,25** ± 370,0 |
| Acurácia multiclasse | 0,699 | 0,476 |
| Melhor época (treino) | −176,6 | −3141,8 |
| Posição | 3º/7 | 5º/7 |
| Exames | 0,0 | 0,0 |

Config idêntica, ambiente idêntico, 10 seeds de avaliação nos dois casos.
Diferença: **3190 pontos**.

### 15.2 Por que isso invalida §12 e §13

Os "efeitos" que atribuí a parâmetros, postos lado a lado com o espalhamento
causado **só pelo seed**:

| Comparação | Δ atribuído | |
|---|---:|---|
| `n_step` 5 → 30 (§12) | +1621 | menor que o ruído |
| `n_step` 30 → 60 (§13) | −3684 | da ordem do ruído |
| `lab_delay` 5 → 2 (§14) | −3319 | da ordem do ruído |
| **seed 42 → 43 (nada mais)** | **−3190** | **o ruído** |

Nenhum dos três efeitos é distinguível da variância de treino. As conclusões
"`n_step: 30` causou o salto" (§12) e "`n_step` tem ótimo estreito em ~30"
(§13) **estão retiradas**. O mesmo vale para a leitura de §14.2 sobre o
`lab_delay`: o colapso do v15 provavelmente não teve nada a ver com o laudo.

### 15.3 O que sobrou de sólido

**Sobre o ambiente** — não depende de treino, então continua de pé:

- A atribuição de recompensa por caso (§10), com teste de invariância provando
  que a soma do episódio não muda, e efeito medido no sinal (97,7% → 0,0% de
  passos sem recompensa; desvio 31,0 → 10,2).
- A correção do vazamento em `_advance_empty_days` (§10.2), com teste de
  regressão.
- A constatação de que esse vazamento **nunca afetou as métricas reportadas**
  (§10.2), verificada re-rodando os baselines.
- 161 testes passando.

**Sobre o agente** — um único achado reproduz em **todas as seis execuções**
(v12, v13 seed 42, v13 seed 43, v14, v15):

> **O DQN nunca pede exame de laboratório. Zero, em todas as configurações e
> em todos os seeds.**

Esse é o resultado robusto da sessão. Não os números de recompensa — que
oscilam 3190 pontos por conta do seed — mas o comportamento, que não oscila
nada.

**Sobre o checkpoint `dqn_v13/policy_best.pth`:** ele de fato obtém −81,90 ±
120,1 sobre 10 seeds de avaliação. É um artefato bom e utilizável. Só não é
reproduzível a partir da config, o que o torna inútil como evidência sobre
qualquer hiperparâmetro.

### 15.4 O que fazer a seguir

O erro metodológico foi comparar configurações com *n* = 1. Qualquer retomada
precisa começar por consertar isso:

1. **Protocolo multi-seed.** Nenhuma comparação entre configs sem ≥3 seeds de
   **treino** e reporte de média ± desvio. Custo: ~2h20 por execução, ~7h por
   configuração. É caro, e é o preço de tirar conclusão de DQN.
2. **Atacar a instabilidade antes dos exames.** Um espalhamento de 3190 pontos
   entre seeds é o problema dominante — maior que qualquer efeito que se queira
   medir. Candidatos padrão, na ordem: `buffer_size` maior (6000 é pequeno para
   episódios de ~400-800 passos), `lr` menor, Double DQN, `target_update_freq`
   mais frequente.
3. **Só então** voltar à questão dos exames — que, pela §15.3, é a única
   pergunta cujo fenômeno é estável o bastante para valer investigação. A
   hipótese viva continua sendo a **atribuição retroativa por caso** (§11.4,
   item 1): creditar o passo do exame quando a conclusão que ele viabilizou é
   paga. É mudança de código no `RewardEngine`, com a mesma exigência de
   invariância da soma do episódio que §10.3 estabeleceu.

### 15.5 Reprodução

```bash
.venv/Scripts/python.exe -m agents.deepq.train --config experiments/configs/train/dqn_v13_seed43.yaml
.venv/Scripts/python.exe -m experiments.evaluate --config experiments/configs/benchmark_v13_seed43.yaml
```
