# Da confirmação epidemiológica ao PPO no mapa real

Relatório da investigação iniciada nas três propostas de revisão do
`epi_confirm`. Tudo aqui é medido; onde há incerteza, ela está declarada.
Histórico completo, com os erros do caminho, em `REFATORACAO_AMBIENTE.md`.

---

## 1. O achado principal

> **O ambiente sintético resolvia o problema sozinho, pela geografia.**
> Medido: a posição do caso, sozinha, acerta a doença em **93,7%** das vezes no
> sintético e em **50,7% ± 2,4** na distribuição real do Rio.

*(Correção: a primeira medição deste número reportou 47,1%, que era o menor de
quatro grupos de seeds, não a média. A média com o desvio é 50,7 ± 2,4 — a
conclusão não muda.)*

50,7% entre duas classes é cara-ou-coroa. Ou seja: no sintético o exame era
**redundante** — o agente não o comprava porque não valia a pena comprar. Na
distribuição real ele passa a ser a única fonte de informação, e o agente
**começa a comprá-lo**, nos três seeds.

Isso reinterpreta boa parte da investigação. Passamos ~20 treinos tratando
"o agente nunca pede exame" como falha do agente. Era, em grande medida, uma
resposta correta a um ambiente que dava a resposta de graça.

---

## 2. Onde estão os agentes

Os baselines **se reordenam** entre as distribuições — nenhuma conclusão de
economia transfere de uma para a outra.

| agente | sintético | **kriging (real)** |
|---|---:|---:|
| `testonce` | +247,4 | **+2093** |
| `testtwice` | **+955,0** | +640 |
| `confirmall` | −1365,2 | +542 |
| `clinical` (não agir) | −267,2 | −15 |
| **`ppo`** (3 seeds) | **+645,6 ± 76,0** | **+723,4 ± 72,6** |
| `dqn` (controle) | −4164 ± 9 | — |

No sintético o PPO é **2º**; no kriging cai para **3º**, e a distância até o
topo vai de 204 para 1.370 pontos. A distribuição real premia muito mais quem
investiga — e é exatamente onde o agente ainda subinveste (7 a 37 exames,
contra os 373 do `testonce`).

---

## 3. Ponto de partida: as três propostas

1. Usar algoritmos de otimização para varrer recompensas e hiperparâmetros.
2. O `epi_confirm` deveria ser **aprendido** da distribuição observada.
3. O `epi_confirm` deveria **custar perto de um exame** (simula ida a campo).

A investigação começou por (2) e (3), que eram acopladas. A (1) foi adiada — e
a própria investigação justificou o adiamento (§8).

---

## 4. O `epi_confirm` vazava a verdade e nunca funcionou

**Vazava:** consultava os mapas do *gerador*, indexados pela doença **real**,
inclusive de casos nunca notificados — por 0,5, contra 4,0 de um exame ruidoso.

**E era inerte:** comparava a densidade de **uma única célula** de um grid
400×400 contra `threshold=1`. Como nenhuma célula tem mais de 1 caso, a
condição nunca disparava. Verificação: **5598 chamadas, todas devolvendo 0**.
O bit `epiconf` era entrada constante zero, e o DQN gastava 66% das ações
comprando isso.

**Corrigido:** lê o mapa de casos confirmados **por laudo do próprio agente**,
raio 20, com auto-exclusão do caso (sem ela, raio 0 "acerta" 99,6% — é o exame
já pago voltando com outro nome). Calibração: 19,3% de confirmação a 80,1% de
precisão, contra 57,1% do palpite clínico.

O incentivo da proposta (2) existe e está travado por teste: **sem exame, zero
confirmações; com exames acumulados, elas aparecem.** O exame virou
investimento.

---

## 5. A penalidade punia a inação

`penalty_unresolved` (−10) caía sobre **todo** caso não concluído, inclusive
os que o agente nunca tocou — e era somada em bloco no fim, sem atribuição.
Medido: **62–85% dos casos** terminavam "em aberto", dominando a perda.

Da ótica do agente: "nada" custava 0,1 imediatos e os −10 chegavam
despersonalizados no fim. **Ele otimizava corretamente um sinal que mentia.**

**Redefinido:** não agir é *deixar valer o diagnóstico do médico* — decisão
legítima, julgada pelo acerto do palpite clínico. Só paga quem **abriu
investigação** e não a fechou, e a cobrança passou a ser no passo em que o caso
é largado. "Nada" virou a **ação nula**: custo 0, passo devolve exatamente 0,0.

Efeito: `clinical` foi de −4036,5 para **−267,2**, passando a bater o
`confirmall`. Abster-se virou melhor que opinar mal, que é o correto.

---

## 6. Custo computacional: 5× mais rápido

**Paralelizar o ambiente era o alvo errado** — o env faz 1263 passos/s contra
~25 do laço de treino, e `subproc` fica 35% *mais lento* por serializar 938 KB
por passo.

**O gargalo era o tamanho da observação**, que a rede descarta de qualquer
forma (`AdaptiveAvgPool2d((6,6))`):

| resolução | KB/obs | gather+H2D | replay buffer |
|---|---:|---:|---:|
| 400×400 | 937,5 | 7,6 s/100upd | 5,36 GB |
| **100×100** | **58,6** | **0,7 s** | **0,34 GB** |

Época: **6min20 → 1min14**. Verificado que não custa fidelidade: no mapa real
só 1,7% dos casos colidem a 100×100. E confirmado por ablação — **zerar o mapa
inteiro muda 0–2% das decisões do agente**; ele decide pelo `case_features`.

**Custo honesto:** quebrou o encoder do `qlearning` de duas formas, uma
silenciosa (fatiamento fora do eixo devolve array vazio, e todas as contagens
locais viravam 0 sem erro). Corrigido, com dois testes de regressão.

---

## 7. DQN: instabilidade real, mas não era a causa

Diagnóstico: a recompensa oscilava até 6.300 pontos entre avaliações vizinhas.

| braço | oscilação | Recompensa |
|---|---:|---:|
| `lr` 2.5e-5, MSE | 1732 (inalterada) | −1592 ± 1340 |
| `lr` 2.5e-5 + **Huber** | **578 (−64%)** | **−4164 ± 9** |

O Huber estabilizou de forma inequívoca (±1340 → **±9**), e o resultado
**piorou**. (Double DQN, ao contrário do que supus, já estava ligado por
padrão o tempo todo.)

> A instabilidade era real e **é tratável**, mas **não era a causa**.
> Estabilizado, o DQN converge de forma confiável para um ótimo local
> degenerado — 66,7% de inação, zero exames.

---

## 8. Três ataques que falharam pela mesma razão

| tentativa | por que falhou |
|---|---|
| **Shaping por potencial** | `s'` já é outro paciente → Φ telescopa no mesmo passo. Medido: F **negativo** em 4 desenhos de Φ. Com γ=0,99 e ~248 investigações abertas, a deriva `(1−γ)Φ ≈ 2,5W` supera o incremento `≈0,99W` — para **qualquer** W. |
| **Crédito retroativo** | Exigiria reescrever recompensa já entregue. Não implementável num MDP online. |
| **Calibrar a economia** | `test_cost` 4,0 → 2,0 dobrou a margem de investigar (1.276 → 2.768). Resultado: **zero exames nos 3 seeds**. |

Os três esbarram na **intercalação de casos**: o valor de investigar o caso X
precisa atravessar ~30 transições que pertencem a outros pacientes.

*Nota:* esta era a leitura ao fim da fase DQN. O §1 a qualifica — no
sintético, o exame também era genuinamente pouco útil.

---

## 9. PPO: o algoritmo era o gargalo

Mesmo ambiente, **mesmo tronco de rede**, mesma máscara de ação:

| | Recompensa | acurácia | estabilidade |
|---|---:|---:|---|
| DQN | −4164 ± 9 | 0,55 | converge para o lugar errado |
| **PPO** | **+632 ± 43** | 0,73 | curva monotônica, `best` = `final` |

~4.800 pontos de diferença. Primeiro agente aprendido a bater "não fazer nada".

**Reproduziu com seeds novos e mais treino:** +645,6 ± 76,0 (seeds 45/46/47,
300 mil passos).

Detalhe de implementação que quase passou: a `ProbabilisticActorPolicy` do
Tianshou **não aplica `obs.mask`** (o `DiscreteQLearningPolicy` aplica). Sem
tratar isso, o PPO ignoraria o `force_decision_after_tests` e estaria
resolvendo um problema mais fácil — **sem erro nenhum aparecer**. A máscara é
aplicada nos logits dentro do ator, e há teste para isso.

---

## 10. Duas lições de método

**Execução única não é evidência.** Mesma config, seeds diferentes: −81,90 e
−3272,25. Três conclusões minhas sobre `n_step` foram retiradas.

**`policy_best` seleciona ruído** — mede **1.000 a 1.460 pontos** acima do
`final` no DQN. No PPO o viés é zero, porque a curva só sobe.

E o que mais rendeu: **medir antes de treinar.** A inviabilidade do shaping
saiu em 10 minutos de medição, em vez de 43 de treino.

---

## 11. O modelo epidêmico estava errado (e como apareceu)

O passo seguinte previsto era "um episódio = um caso" (§8). Antes de
implementá-lo, fomos **medir** a dinâmica temporal que a reformulação
sacrificaria — e o que apareceu foi um defeito no gerador, presente desde o
início do projeto.

O SIR em `data/generator.py` tinha dois erros que se mascaravam:

- **Transmissão sem `/N`.** A força de infecção usava `beta * s * i` em vez de
  `beta * s * i / n`. Com população 150, um R0 declarado de 1,5 valia **225**.
- **Período infeccioso de 250 dias** (`gamma = 0,004`), contra os ~5 dias da
  literatura.

Resultado: a epidemia inteira cabia em **9 dias**, com pico no dia 3, apesar de
`epilength: 60`. Toda a estrutura temporal do ambiente — sobreposição de casos,
espera pelo laudo, fase da epidemia — estava calibrada a esse pico artificial.

**SEIR com parâmetros de literatura** (`dengue_envs/core/epi_model.py`). O
período latente absorve a incubação extrínseca no mosquito, para que o tempo de
geração do modelo corresponda ao intervalo serial observado em campo:

| doença | latente | infeccioso | tempo de geração | R0 |
|---|---:|---:|---:|---|
| dengue | 11 d | 5 d | 16 d | 1,25–1,70 |
| chikungunya | 8 d | 6 d | 14 d | 1,46–1,67 |

Fontes verificadas uma a uma: Chan & Johansson 2012 (incubação); Carrington &
Simmons 2014 (viremia); Aldstadt et al. 2012 (intervalo serial 15–17 d);
Villela et al. 2017 (R0 da dengue no Rio: 1,70 em 2002, 1,25 em 2012); Moreira
et al. 2023 (chikungunya: R0 1,56, IC 1,46–1,67, tempo de geração 14 d
assumido).

**Efeito medido no kriging:**

| | antes (bug) | **agora (SEIR)** |
|---|---:|---:|
| dias com notificação | 9 | **184** |
| pico | 60 casos/dia | **4 casos/dia** |
| passos entre decisões do mesmo caso | 278 | **30** |
| desconto acumulado nesse intervalo (γ=0,99) | 0,061 | **0,74** |

O modelo antigo permanece sob `epi_model: legacy`, que continua sendo o
**padrão** — trocá-lo mudaria em silêncio todos os resultados já produzidos.
Os ambientes novos (`synthetic_v8.yaml`, `kriging_v8.yaml`) pedem o SEIR
explicitamente.

**Os baselines se mantiveram na mesma ordem** (10 seeds):

| agente | kriging v7 | **kriging v8** | sintético v7 | **sintético v8** |
|---|---:|---:|---:|---:|
| `testonce` | +2093 | **+2782** | +247 | +473 |
| `testtwice` | +640 | +1051 | **+955** | **+1249** |
| `confirmall` | +542 | +346 | −1365 | −1821 |
| `clinical` | −15 | −64 | −267 | −355 |

Nenhuma conclusão de economia se inverteu: `testonce` segue ótimo no kriging e
`testtwice` no sintético.

**Custo honesto:** o formato das curvas do SIR parecia plausível; só a
epidemiologia estava errada. Por isso os 19 testes novos travam propriedades
**verificáveis contra a teoria** — o R0 efetivo medido pela taxa de crescimento
(dentro de 3% do declarado), a equação do tamanho final z = 1 − exp(−R0·z), a
duração em meses — e não o formato das curvas. Um teste documenta o bug do
modelo legado, para que ninguém volte a usá-lo achando que R0 = 1,5 significa o
que diz.

---

## 12. Crédito por caso: o que finalmente destravou

Restrição de projeto (definida com o orientador): **a dinâmica temporal da
epidemia precisa ser preservada** — é ela que dá sentido à publicação, e as
decisões têm de acontecer nesse tempo. Isso descarta "um episódio = um caso",
que resolveria a intercalação destruindo justamente o que interessa.

**A medição que apontou a saída** (kriging v8, política que investiga):

| sinal | correlação com a decisão individual |
|---|---:|
| retorno global do episódio | **−0,019** |
| retorno por caso, descontado em dias | **0,994** |

A informação existe — o que a apagava era **onde** o algoritmo a procurava.

**Implementação.** O ambiente decompõe a recompensa por caso e publica no
`info` de cada passo: `case_id`, `day`, `r_case`, `case_done`, `episode_uid`. O
`PerCasePPO` (`agents/ppo/credit.py`) reagrupa a trajetória por caso e calcula
GAE ao longo da linha do tempo daquele paciente, com **γ elevado aos dias**
decorridos entre as decisões — não aos passos. É a mesma ideia de atribuição de
crédito usada em sistemas multiagente.

Duas invariâncias travadas por teste:

- **A recompensa do ambiente não muda.** A soma das parcelas por caso
  reconstrói o total do episódio, incluindo a cauda dos casos que ficam abertos
  (verificado em 4 seeds e sob SEIR). A métrica de comparação segue canônica.
- **A vantagem de um caso não depende da intercalação.** Inserir 30 decisões
  ruidosas sobre outros pacientes entre dois passos do caso X não altera a
  vantagem de X.

Junto, 4 **features temporais** opcionais na observação: dia/horizonte, casos
notificados hoje, tendência de 7 dias contra os 7 anteriores, idade do caso.
Todas lidas do que a vigilância **observa** — há teste de que nenhuma consulta
a curva verdadeira do gerador.

**Resultado** (kriging v8, 3 seeds, checkpoint final, benchmark de 10 seeds):

| braço | recompensa | acurácia | exames/episódio |
|---|---:|---:|---:|
| `testonce` (melhor política fixa) | +2782 | 96,0% | 369 |
| **B: crédito por caso + tempo** | **+2649 ± 50** | **93,1%** | **266** |
| A: SEIR + GAE padrão | +729 ± 119 | 74,3% | 30 |
| `clinical` | −64 | 71,0% | 0 |

Por seed — B: +2706, +2615, +2627. A: +844, +607, +737.

- **95% do melhor baseline**, com **28% menos exames**: o agente discrimina
  quais casos investigar, que é exatamente a competência que o ambiente foi
  desenhado para cobrar.
- De 30 para 266 exames. É o primeiro agente do projeto que investiga.
- **Corrigir a epidemia, sozinho, não resolveu.** O braço A treinou no mesmo
  ambiente corrigido e manteve o padrão antigo de subinvestimento.
- O desvio entre seeds (50) é pequeno diante da diferença entre braços (~1900).

**Ressalva de desenho, declarada:** o braço B mudou duas coisas ao mesmo tempo
(crédito e observação). A ablação — crédito por caso com a observação idêntica à
do braço A — está rodando em 3 seeds.

**Limitação conhecida:** o crédito por caso ignora a externalidade do exame —
cada laudo positivo melhora o `epi_confirm` de casos **futuros** na região, e
esse ganho não volta para o caso que pagou o exame.

---

## 13. Próximos passos

### Imediatos e baratos

**1. Adotar o kriging como ambiente de referência.** O sintético mascarava o
fenômeno central — nele o problema de alocação de exames praticamente não
existe (§1). Toda medição de projeto daqui em diante deveria ser feita na
distribuição real.

**2. Refazer a varredura econômica no kriging** (~10 min, políticas fixas, sem
treino). Toda a calibração de `test_cost` — a varredura, o "testa2 é ótimo", o
ambiente degenerar em 6,0 — vale só para o sintético. Lá o `testonce` já é o
melhor. **É pré-requisito para qualquer decisão sobre custos.**

**3. Mais treino do PPO no kriging** (~2h/seed). O melhor seed ainda subia na
última época: +825,8 não é teto, é onde o orçamento acabou.

*(Os itens 1 a 3 acima já foram executados: o kriging é o ambiente de
referência, a varredura econômica foi refeita e o PPO rodou 300 mil passos.)*

### Estruturais

**4. ~~Um episódio = um caso~~ — descartado.** Destruiria a dinâmica temporal,
que é requisito da publicação. A intercalação foi resolvida no **learner**, não
no ambiente (§12).

**5. ~~Fechar a lacuna dos exames no kriging~~ — fechada.** De 7–37 para 266
exames, a 95% do melhor baseline (§12). O que resta é o último degrau: +2649
contra +2782 do `testonce`.

**5b. Separar crédito de observação temporal.** Ablação em andamento, 3 seeds.

**5c. Sazonalidade.** O SEIR não tem forçamento sazonal; com R0 = 1,25 a
epidemia leva ~9 meses. Decisão de modelagem em aberto.

**5d. Chikungunya não é mais forçadamente menor que a dengue.** As faixas de R0
vêm da literatura e se sobrepõem (1,46–1,67 contra 1,25–1,70). Antes a chik era
menor por construção. É uma mudança de premissa a confirmar com o orientador.

### Retomando a proposta (1)

**6. Busca automática de recompensa e hiperparâmetros.** Volta à mesa agora que
há um algoritmo estável — buscar sobre um DQN que oscilava 6.300 pontos seria
otimizar sorteio de seed. Recomendação: Optuna com TPE e pruner ASHA, objetivo
= média sobre ≥3 seeds **menos** o desvio (prefere configurações confiáveis, não
sortudas). O §6 tornou isso viável ao baratear cada avaliação.

### O que não vale a pena

Continuar ajustando o DQN. `lr`, `buffer`, `n_step` e `target_update_freq` não
moveram nada mensurável; o Huber estabilizou e piorou. O ganho veio de mudar de
algoritmo, não de girar botões.

---

## 14. Estado do código

- **259 testes passando** (154 no início desta investigação).
- Modelo epidêmico em `dengue_envs/core/epi_model.py`, com as fontes citadas no
  cabeçalho. `epi_model: legacy` continua o padrão; os ambientes v8 pedem
  `seir`.
- Ambientes: `kriging_v8.yaml` (referência), `kriging_v8_temporal.yaml`
  (idêntico, com as features temporais), `synthetic_v8.yaml`. Os v7
  correspondentes seguem no lugar, para reprodução.
- Crédito por caso: decomposição no ambiente
  (`dengue_diagnostics.py`/`case_by_case.py`), GAE por caso em
  `agents/ppo/credit.py`. Ligado por `train.per_case_credit: true`, que exige
  `per_case_reward` no ambiente.
- PPO em `agents/ppo/`, registrado no `AGENT_REGISTRY` — entra no benchmark com
  as mesmas seeds dos demais.
- Existe um `agents/ppo/ppo.py` (script standalone do CleanRL, anterior à
  sessão) que não conflita, mas convive confusamente com o pacote novo.

```bash
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m agents.ppo.train --config experiments/configs/train/ppo_v4b_credito_s45.yaml
.venv/Scripts/python.exe -m experiments.evaluate --config experiments/configs/benchmark_ppo_v4b_credito_s45.yaml
```

**Resultados brutos:** `results/bm_ppo_v4a_seir_s4*` (GAE padrão),
`results/bm_ppo_v4b_credito_s4*` (crédito + tempo),
`results/bm_ppo_v4c_credito_sem_tempo_s4*` (ablação),
`results/baseline_v8seir_{kriging,synthetic}`.

**Nota operacional:** o treino não retoma de onde parou. Uma reinicialização do
Windows no meio da noite custou duas seeds inteiras, que precisaram ser
refeitas do zero.
