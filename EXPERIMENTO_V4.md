# Experimento v4 — o que destravou o agente: crédito, não observação

Três braços de treino no ambiente kriging com SEIR de literatura, 3 seeds cada,
para responder **uma** pergunta: por que o agente subinvestia em exames, e o que
resolve isso sem destruir a dinâmica temporal da epidemia.

Resultado: **a atribuição de crédito no aprendiz explica o ganho inteiro.** Nem
a correção do modelo epidêmico, nem a informação temporal na observação
produziram o salto sozinhas.

---

## 1. A pergunta

Até a fase v3, todo agente treinado subinvestia: o PPO pedia 7 a 37 exames por
episódio contra os ~370 da melhor política fixa. A hipótese corrente era a
**intercalação de casos** — um exame devolve sempre apenas `−custo`, e o
benefício aparece dezenas de passos depois, num passo que pertence a outro
paciente.

A saída óbvia (um episódio = um caso) foi **descartada**: destruiria a dinâmica
temporal da epidemia, que é requisito da publicação. As decisões precisam
acontecer dentro do tempo epidêmico.

Duas coisas foram feitas em vez disso, e os três braços existem para separá-las:

1. corrigir o modelo epidêmico (que estava errado — §2);
2. mudar como o algoritmo atribui crédito, mantendo o ambiente intacto (§3).

---

## 2. O que os três braços têm em comum

Tudo abaixo é idêntico em A, B e C. O que muda está no §3.

**Ambiente** — [`experiments/configs/env/kriging_v8.yaml`](experiments/configs/env/kriging_v8.yaml)

- Distribuição espacial real: `P(célula | doença)` estimada por Ordinary
  Kriging sobre as notificações do Rio 2015-16, com rotação/espelho sorteados
  por episódio (`augment_surfaces`).
- **Dinâmica temporal SEIR com parâmetros de literatura** (`epi_model: seir`),
  em [`dengue_envs/core/epi_model.py`](dengue_envs/core/epi_model.py). Substitui
  o SIR anterior, que não dividia a transmissão pela população e usava período
  infeccioso de 250 dias: um R0 declarado de 1,5 valia 225, e a epidemia inteira
  cabia em 9 dias.

  | doença | latente | infeccioso | tempo de geração | R0 |
  |---|---:|---:|---:|---|
  | dengue | 11 d | 5 d | 16 d | 1,25–1,70 |
  | chikungunya | 8 d | 6 d | 14 d | 1,46–1,67 |

  Fontes: Chan & Johansson 2012 (incubação); Carrington & Simmons 2014
  (viremia); Aldstadt et al. 2012 (intervalo serial 15–17 d); Villela et al.
  2017 (R0 da dengue no Rio); Moreira et al. 2023 (chikungunya). O período
  latente absorve a incubação extrínseca no mosquito, para que o tempo de
  geração corresponda ao intervalo serial observado.

  Efeito medido: 184 dias de notificação (contra 9), pico de 4 casos/dia
  (contra 60), ~30 passos entre duas decisões do mesmo caso (contra 278).
- Recompensa canônica: exame 4,0; `epi_confirm` 4,0 (raio 20, limiar 1);
  `penalty_unresolved` −10 só para caso investigado e largado; "nada" é ação
  nula (custo 0); `lab_delay_days: 5`; `max_case_revisits: 2`.
- `per_case_reward: true` (entrega a recompensa no passo do caso; a soma do
  episódio é idêntica ao modo agregado), `map_size: 100`, `context_features`.

**Treino** — PPO, 300 mil passos (10 épocas × 30 mil), `lr` 3e-4, γ 0,99,
λ 0,95, `eps_clip` 0,2, `ent_coef` 0,02, `batch_size` 256, 8 ambientes de
treino, `reward_scale` 0,10. Rede: mesmo tronco convolucional em todos.

**Avaliação** — checkpoint **final** (nunca o `best`, que seleciona ruído:
medimos 1.000–1.460 pontos de viés), benchmark de 10 seeds fixas
(100…550), contra `clinical` e `testonce`.

---

## 3. O que muda em cada braço

| | crédito (learner) | observação | config |
|---|---|---|---|
| **A** | GAE padrão | 16 dims de contexto | `train/ppo_v4a_seir_s*.yaml` |
| **B** | por caso, desconto em dias | 16 + **4 temporais** = 20 | `train/ppo_v4b_credito_s*.yaml` |
| **C** | por caso, desconto em dias | 16 dims (**igual ao A**) | `train/ppo_v4c_credito_sem_tempo_s*.yaml` |

O par **A × C** isola o crédito (observação idêntica). O par **B × C** isola as
features temporais (crédito idêntico).

### Braço A — SEIR + GAE padrão

Nenhum código novo. É o PPO da fase v3 rodando no ambiente corrigido, e serve
para responder: *a correção do modelo epidêmico, sozinha, resolve?*

O GAE encadeia passos **consecutivos da trajetória**. Como o agente decide um
caso por passo e os casos se sobrepõem, o valor de pedir um exame precisa
atravessar ~30 decisões sobre outros pacientes até o laudo voltar.

### Braço B — crédito por caso + features temporais

**(a) Decomposição por caso no ambiente.** Cada passo publica em `info` de qual
caso foi a decisão, em que dia, quanto daquela recompensa lhe pertence, e se o
caso terminou — [`case_by_case.py`](dengue_envs/wrappers/case_by_case.py):

```python
u = self.unwrapped; dia = int(u.t)
case_r = u.apply_case_action(case_id, int(action))
feito = u.case_is_done(case_id)
r_case = case_r + (u.attribute_terminal(case_id) if feito else 0.0)
obs, day_r, terminated, truncated, info = self._next_case()
info = self._credit_info(info, case_id, dia, r_case, feito)
return obs, case_r + day_r, terminated, truncated, info
```

A recompensa devolvida ao ambiente (`case_r + day_r`) **não muda** — é a métrica
do benchmark. `r_case` viaja em paralelo, no `info`.

Peças que isso exigiu, em [`dengue_diagnostics.py`](dengue_envs/envs/dengue_diagnostics.py):
`case_is_done` (o caso saiu da fila, foi finalizado ou não voltará),
`attribute_terminal` (paga a parcela terminal do caso **uma vez só**, controlada
por um conjunto de casos já atribuídos) e `episode_uid` (identificador
aleatório por episódio, porque os `case_id` reiniciam e não podem ser fundidos
entre episódios).

**(b) GAE por caso, descontado em dias** —
[`agents/ppo/credit.py`](agents/ppo/credit.py). As transições do lote são
agrupadas por `(episode_uid, case_id)` e o GAE corre ao longo da linha do tempo
de cada paciente:

```
desconto_j = gamma ** (dia_{j+1} − dia_j)      # dias, não passos
delta_j    = r_j + desconto_j · V_{j+1} − V_j
A_j        = delta_j + desconto_j · λ · A_{j+1}
```

No último passo do caso dentro do lote: se o caso terminou, não há futuro; se o
lote cortou a linha do tempo dele, faz bootstrap com o próprio `V` e um dia de
desconto. Transições sem caso (preenchimento no fim do episódio) viram grupos
de uma posição.

`credit_scale` reaplica o `reward_scale` do treino, porque o `RewardScaleWrapper`
escala a recompensa mas não o `info` — sem isso os alvos do crítico ficariam 10×
maiores que o pretendido.

**A medição que motivou tudo isso** (kriging v8, no passo de cada exame,
correlação entre o retorno e o exame ter levado ao diagnóstico correto):

| definição de retorno | correlação |
|---|---:|
| global, desconto por passo (GAE padrão) | **−0,019** |
| do próprio caso, desconto por dia | **0,994** |

O sinal existe na trajetória; a definição padrão de retorno o apaga.

**(c) Features temporais na observação** (4 dims): dia/horizonte; casos
notificados hoje (satura em 10); tendência dos últimos 7 dias contra os 7
anteriores, em [0,1]; idade do caso (satura em 14 dias). Todas lidas do que a
vigilância **observa** — há teste de que nenhuma consulta a curva verdadeira do
gerador. Ambiente separado
([`kriging_v8_temporal.yaml`](experiments/configs/env/kriging_v8_temporal.yaml))
porque a observação muda de forma e o benchmark precisa reconstruir o mesmo
ambiente do treino.

### Braço C — crédito por caso, sem features temporais

Ablação: (a) e (b) do braço B, com a observação **idêntica à do braço A**
(`env_config: ../env/kriging_v8.yaml`, sem `temporal_features`).

---

## 4. Resultados

Checkpoint final, benchmark de 10 seeds, média sobre 3 seeds de treino:

| braço | recompensa | acurácia | exames/episódio | seeds (45/46/47) |
|---|---:|---:|---:|---|
| `testonce` (melhor política fixa) | +2782 | 96,0% | 369 | — |
| **C — crédito, sem tempo** | **+2678 ± 48** | 93,6% | 281 | 2681 / 2628 / 2725 |
| **B — crédito + tempo** | **+2649 ± 50** | 93,1% | 266 | 2706 / 2615 / 2627 |
| A — GAE padrão | +729 ± 119 | 74,3% | 30 | 844 / 607 / 737 |
| `clinical` (não agir) | −64 | 71,0% | 0 | — |

**O que os contrastes licenciam concluir:**

- **A × C (+729 → +2678, observação idêntica):** o ganho vem da atribuição de
  crédito. É a única diferença entre os dois braços.
- **B × C (+2649 contra +2678):** as features temporais **não contribuíram**. A
  diferença entre as médias (29) é menor que o desvio de cada braço (~49), e as
  faixas por seed se sobrepõem.
- **Corrigir o modelo epidêmico não bastou:** o braço A treinou na epidemia
  correta e manteve o padrão antigo de subinvestimento (30 exames).
- **O agente discrimina:** 281 exames contra os 369 do `testonce`, com 2,4
  pontos de acurácia a menos — escolhe *quais* casos investigar, em vez de
  testar todos.
- **Estabilidade:** desvio de ~49 pontos entre seeds nos braços com crédito,
  contra ~1950 de diferença para o braço A.

---

## 5. O que isto não estabelece

- **O acoplamento entre casos fica fora do crédito.** Um laudo positivo alimenta
  o mapa de confirmados, que ajuda `epi_confirm` em casos **futuros** na região.
  O retorno por caso não enxerga esse benefício. Hoje pesa pouco (o agente não
  usa `epi_confirm` nem o mapa), mas é uma limitação real do método.
- **Falta o último degrau:** +2678 contra +2782 do `testonce`.
- **O mapa segue decorativo.** Zerar o tensor espacial muda 0–2% das decisões; o
  agente decide pelas features do caso.
- **Sem sazonalidade.** Com R0 = 1,25 a epidemia leva ~9 meses.
- **Chikungunya não é mais forçadamente menor que a dengue** — as faixas de R0
  agora vêm da literatura e se sobrepõem. É mudança de premissa, a confirmar.
- **As features temporais podem sair.** Não pagaram seu custo neste desenho.
  Isso não prova que informação temporal seja inútil, só que *estas quatro*, com
  este crédito, não acrescentaram.

---

## 6. Garantias de teste

O método só vale se a recompensa comparada continuar sendo a do ambiente. Dois
invariantes travados por teste
([`test_case_credit.py`](dengue_envs/tests/test_case_credit.py),
[`test_case_credit_gae.py`](agents/ppo/tests/test_case_credit_gae.py)):

- **A decomposição reconstrói o total.** A soma das parcelas por caso é igual à
  recompensa do episódio, incluindo a cauda dos casos que ficam abertos
  (verificado em 4 seeds e sob SEIR). Ligar as features não altera a recompensa.
- **A vantagem de um caso não depende da intercalação.** Inserir 30 decisões
  ruidosas sobre outros pacientes entre dois passos do caso X não muda a
  vantagem de X.

Mais: atribuição terminal acontece uma única vez por caso; `episode_uid` separa
casos homônimos de episódios diferentes; as features temporais não leem a curva
verdadeira; e o `PerCasePPO` falha com mensagem explícita se o ambiente não
publicar os campos de crédito. Suíte completa: **259 testes**.

Os 19 testes do modelo epidêmico travam propriedades **verificáveis contra a
teoria** — R0 efetivo medido pela taxa de crescimento (dentro de 3% do
declarado), equação do tamanho final z = 1 − exp(−R0·z), duração em meses — e
não o formato das curvas. O SIR defeituoso produzia curvas de formato plausível;
só a epidemiologia estava errada. Um teste documenta esse bug, para que ninguém
volte a usar o modelo legado achando que R0 = 1,5 significa o que diz.

---

## 7. Reprodução

```bash
.venv/Scripts/python.exe -m agents.ppo.train --config experiments/configs/train/ppo_v4c_credito_sem_tempo_s45.yaml
.venv/Scripts/python.exe -m experiments.evaluate --config experiments/configs/benchmark_ppo_v4c_credito_sem_tempo_s45.yaml
```

Trocar `v4c_credito_sem_tempo` por `v4a_seir` ou `v4b_credito` para os demais
braços; seeds 45, 46, 47. Baselines: `benchmark_v8seir_kriging.yaml`.

Resultados brutos em `results/bm_ppo_v4{a,b,c}_*` e
`results/baseline_v8seir_kriging`.

**Custo:** ~3h por seed (300 mil passos), com dois treinos simultâneos na GPU.
O laço é limitado pelo **ambiente**, não pela rede — medido: 48 passos/s tanto
em CPU quanto em CUDA.

**Nota operacional:** três quedas de máquina (reinício do Windows, GPU perdida,
travamento) custaram 8 seeds inteiras nesta série. O treino passou a salvar
`policy_epoch.pth` a cada época — antes, uma queda na época 9 de 10 não deixava
nada aproveitável, porque `policy_final` só é escrito no fim e `policy_best`
seleciona ruído.
