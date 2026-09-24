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

> **Correção posterior (§15):** a medição acima continua valendo, mas a
> leitura de que *a geografia* explicava a diferença entre kriging e sintético
> estava **confundida**. O gerador kriging ignorava `other_prevalence` e não
> tinha casos "outro" (não-arbovirose); o sintético tinha. Sem eles, um laudo
> negativo de dengue implica chik e um exame sempre basta — é isso, e não a
> geografia, que fazia o `testonce` ótimo no kriging. Varrendo a geografia de
> 50% a 94% de acerto (§14), nada muda; ligando os casos "outro" no kriging
> (v9), os baselines voltam à ordem do sintético.

---



## 2. Onde estão os agentes

Os baselines **se reordenam** entre as distribuições — nenhuma conclusão de
economia transfere de uma para a outra.


| agente                | sintético         | **kriging (real)** |
| --------------------- | ----------------- | ------------------ |
| `testonce`            | +247,4            | **+2093**          |
| `testtwice`           | **+955,0**        | +640               |
| `confirmall`          | −1365,2           | +542               |
| `clinical` (não agir) | −267,2            | −15                |
| `ppo` (3 seeds)       | **+645,6 ± 76,0** | **+723,4 ± 72,6**  |
| `dqn` (controle)      | −4164 ± 9         | —                  |


No sintético o PPO é **2º**; no kriging cai para **3º**, e a distância até o
topo vai de 204 para 1.370 pontos. A distribuição real premia muito mais quem
investiga — e é exatamente onde o agente ainda subinveste (7 a 37 exames,
contra os 373 do `testonce`).

*Nota (§15): a reordenação dos baselines vinha da ausência de casos "outro" no
kriging, não da distribuição espacial.*

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


| resolução   | KB/obs   | gather+H2D   | replay buffer |
| ----------- | -------- | ------------ | ------------- |
| 400×400     | 937,5    | 7,6 s/100upd | 5,36 GB       |
| **100×100** | **58,6** | **0,7 s**    | **0,34 GB**   |


Época: **6min20 → 1min14**. Verificado que não custa fidelidade: no mapa real
só 1,7% dos casos colidem a 100×100. E confirmado por ablação — **zerar o mapa
inteiro muda 0–2% das decisões do agente**; ele decide pelo `case_features`.

**Custo honesto:** quebrou o encoder do `qlearning` de duas formas, uma
silenciosa (fatiamento fora do eixo devolve array vazio, e todas as contagens
locais viravam 0 sem erro). Corrigido, com dois testes de regressão.

---



## 7. DQN: instabilidade real, mas não era a causa

Diagnóstico: a recompensa oscilava até 6.300 pontos entre avaliações vizinhas.


| braço                   | oscilação         | Recompensa    |
| ----------------------- | ----------------- | ------------- |
| `lr` 2.5e-5, MSE        | 1732 (inalterada) | −1592 ± 1340  |
| `lr` 2.5e-5 + **Huber** | **578 (−64%)**    | **−4164 ± 9** |


O Huber estabilizou de forma inequívoca (±1340 → **±9**), e o resultado
**piorou**. (Double DQN, ao contrário do que supus, já estava ligado por
padrão o tempo todo.)

> A instabilidade era real e **é tratável**, mas **não era a causa**.
> Estabilizado, o DQN converge de forma confiável para um ótimo local
> degenerado — 66,7% de inação, zero exames.

---



## 8. Três ataques que falharam pela mesma razão


| tentativa                 | por que falhou                                                                                                                                                                                                          |
| ------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Shaping por potencial** | `s'` já é outro paciente → Φ telescopa no mesmo passo. Medido: F **negativo** em 4 desenhos de Φ. Com γ=0,99 e ~248 investigações abertas, a deriva `(1−γ)Φ ≈ 2,5W` supera o incremento `≈0,99W` — para **qualquer** W. |
| **Crédito retroativo**    | Exigiria reescrever recompensa já entregue. Não implementável num MDP online.                                                                                                                                           |
| **Calibrar a economia**   | `test_cost` 4,0 → 2,0 dobrou a margem de investigar (1.276 → 2.768). Resultado: **zero exames nos 3 seeds**.                                                                                                            |


Os três esbarram na **intercalação de casos**: o valor de investigar o caso X
precisa atravessar ~30 transições que pertencem a outros pacientes.

*Nota:* esta era a leitura ao fim da fase DQN. O §1 a qualifica — no
sintético, o exame também era genuinamente pouco útil.

---



## 9. PPO: o algoritmo era o gargalo

Mesmo ambiente, **mesmo tronco de rede**, mesma máscara de ação:


|         | Recompensa    | acurácia | estabilidade                       |
| ------- | ------------- | -------- | ---------------------------------- |
| DQN     | −4164 ± 9     | 0,55     | converge para o lugar errado       |
| **PPO** | **+632 ± 43** | 0,73     | curva monotônica, `best` = `final` |


~4.800 pontos de diferença. Primeiro agente aprendido a bater "não fazer nada".

**Reproduziu com seeds novos e mais treino:** +645,6 ± 76,0 (seeds 45/46/47,
300 mil passos).

Detalhe de implementação que quase passou: a `ProbabilisticActorPolicy` do
Tianshou **não aplica** `obs.mask` (o `DiscreteQLearningPolicy` aplica). Sem
tratar isso, o PPO ignoraria o `force_decision_after_tests` e estaria
resolvendo um problema mais fácil — **sem erro nenhum aparecer**. A máscara é
aplicada nos logits dentro do ator, e há teste para isso.

---



## 10. Duas lições de método

**Execução única não é evidência.** Mesma config, seeds diferentes: −81,90 e
−3272,25. Três conclusões minhas sobre `n_step` foram retiradas.

`policy_best` **seleciona ruído** — mede **1.000 a 1.460 pontos** acima do
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

- **Transmissão sem** `/N`**.** A força de infecção usava `beta * s * i` em vez de
`beta * s * i / n`. Com população 150, um R0 declarado de 1,5 valia **225**.
- **Período infeccioso de 250 dias** (`gamma = 0,004`), contra os ~5 dias da
literatura.

Resultado: a epidemia inteira cabia em **9 dias**, com pico no dia 3, apesar de
`epilength: 60`. Toda a estrutura temporal do ambiente — sobreposição de casos,
espera pelo laudo, fase da epidemia — estava calibrada a esse pico artificial.

**SEIR com parâmetros de literatura** (`dengue_envs/core/epi_model.py`). O
período latente absorve a incubação extrínseca no mosquito, para que o tempo de
geração do modelo corresponda ao intervalo serial observado em campo:


| doença      | latente | infeccioso | tempo de geração | R0        |
| ----------- | ------- | ---------- | ---------------- | --------- |
| dengue      | 11 d    | 5 d        | 16 d             | 1,25–1,70 |
| chikungunya | 8 d     | 6 d        | 14 d             | 1,46–1,67 |


Fontes verificadas uma a uma: Chan & Johansson 2012 (incubação); Carrington &
Simmons 2014 (viremia); Aldstadt et al. 2012 (intervalo serial 15–17 d);
Villela et al. 2017 (R0 da dengue no Rio: 1,70 em 2002, 1,25 em 2012); Moreira
et al. 2023 (chikungunya: R0 1,56, IC 1,46–1,67, tempo de geração 14 d
assumido).

**Efeito medido no kriging:**


|                                             | antes (bug)  | **agora (SEIR)** |
| ------------------------------------------- | ------------ | ---------------- |
| dias com notificação                        | 9            | **184**          |
| pico                                        | 60 casos/dia | **4 casos/dia**  |
| passos entre decisões do mesmo caso         | 278          | **30**           |
| desconto acumulado nesse intervalo (γ=0,99) | 0,061        | **0,74**         |


O modelo antigo permanece sob `epi_model: legacy`, que continua sendo o
**padrão** — trocá-lo mudaria em silêncio todos os resultados já produzidos.
Os ambientes novos (`synthetic_v8.yaml`, `kriging_v8.yaml`) pedem o SEIR
explicitamente.

**Os baselines se mantiveram na mesma ordem** (10 seeds):


| agente       | kriging v7 | **kriging v8** | sintético v7 | **sintético v8** |
| ------------ | ---------- | -------------- | ------------ | ---------------- |
| `testonce`   | +2093      | **+2782**      | +247         | +473             |
| `testtwice`  | +640       | +1051          | **+955**     | **+1249**        |
| `confirmall` | +542       | +346           | −1365        | −1821            |
| `clinical`   | −15        | −64            | −267         | −355             |


Nenhuma conclusão de economia se inverteu: `testonce` segue ótimo no kriging e
`testtwice` no sintético. *(Hoje sabemos por quê: a diferença entre as duas
colunas era a falta de casos "outro" no kriging — §15.)*

**Custo honesto:** o formato das curvas do SIR parecia plausível; só a
epidemiologia estava errada. Por isso os 19 testes novos travam propriedades
**verificáveis contra a teoria** — o R0 efetivo medido pela taxa de crescimento
(dentro de 3% do declarado), a equação do tamanho final z = 1 − exp(−R0·z), a
duração em meses — e não o formato das curvas. Um teste documenta o bug do
modelo legado, para que ninguém volte a usá-lo achando que R0 = 1,5 significa o
que diz.

---



## 12. Crédito por caso: o que finalmente destravou

Restrição de projeto: **a dinâmica temporal da epidemia precisa ser preservada** — é ela que dá sentido à publicação, e as
decisões têm de acontecer nesse tempo. Isso descarta "um episódio = um caso",
que resolveria a intercalação destruindo justamente o que interessa.

**A medição que apontou a saída** (kriging v8, política que investiga):


| sinal                                | correlação com a decisão individual |
| ------------------------------------ | ----------------------------------- |
| retorno global do episódio           | **−0,019**                          |
| retorno por caso, descontado em dias | **0,994**                           |


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


| braço                             | recompensa     | acurácia  | exames/episódio |
| --------------------------------- | -------------- | --------- | --------------- |
| `testonce` (melhor política fixa) | +2782          | 96,0%     | 369             |
| **B: crédito por caso + tempo**   | **+2649 ± 50** | **93,1%** | **266**         |
| A: SEIR + GAE padrão              | +729 ± 119     | 74,3%     | 30              |
| `clinical`                        | −64            | 71,0%     | 0               |


Por seed — B: +2706, +2615, +2627. A: +844, +607, +737.

- **95% do melhor baseline**, com **28% menos exames**: o agente discrimina
quais casos investigar, que é exatamente a competência que o ambiente foi
desenhado para cobrar.
- De 30 para 266 exames. É o primeiro agente do projeto que investiga.
- **Corrigir a epidemia, sozinho, não resolveu.** O braço A treinou no mesmo
ambiente corrigido e manteve o padrão antigo de subinvestimento.
- O desvio entre seeds (50) é pequeno diante da diferença entre braços (~1900).

**A ablação separou as duas mudanças.** O braço B mudou crédito e observação ao
mesmo tempo; o braço C repete o crédito com a observação **idêntica à do braço
A**:


| braço | crédito    | observação       | recompensa     |
| ----- | ---------- | ---------------- | -------------- |
| **C** | por caso   | 16 dims (= A)    | **+2678 ± 48** |
| B     | por caso   | 16 + 4 temporais | +2649 ± 50     |
| A     | GAE padrão | 16 dims          | +729 ± 119     |


**B e C são indistinguíveis** (29 pontos de diferença, contra ~49 de desvio em
cada um). **A e C diferem por ~1950 com observação idêntica.** Logo: o ganho é
da **atribuição de crédito**, e as 4 features temporais não pagaram seu custo —
podem sair. Detalhamento dos três braços em `EXPERIMENTO_V4.md`.

**Limitação conhecida:** o crédito por caso ignora a externalidade do exame —
cada laudo positivo melhora o `epi_confirm` de casos **futuros** na região, e
esse ganho não volta para o caso que pagou o exame.

*Os números desta seção são do kriging v8, sem casos "outro" (§15). Com a 4ª
seed e intervalos de confiança, ver §13; o resultado no ambiente corrigido
está no §16.*

---



## 13. Estatística: bootstrap hierárquico e pareado

Até aqui os resultados eram média ± desvio entre 3 seeds. Isso esconde duas
coisas: **há duas fontes de variância** (a seed que treinou e o surto em que se
avaliou), e **os episódios são pareados** — todos os braços usam as mesmas 10
seeds de avaliação, que fixam o mesmo surto e o mesmo médico (verificado nos
arquivos). `experiments/bootstrap.py` reamostra os dois níveis, com os
episódios **compartilhados** entre braços em cada réplica, e reporta média, IQM
e P(X > Y) episódio a episódio (Agarwal et al. 2021; Saravanan et al. 2020).

**Kriging v8, com a 4ª seed de A e C** (IC 95%, 10 mil réplicas):


| comparação               | recompensa                         | exames              | acurácia                  |
| ------------------------ | ---------------------------------- | ------------------- | ------------------------- |
| **C × A**                | **+2009** [+1197, +2882], P = 1,00 | +259                | +19,9 pp                  |
| B × C                    | −43 [−196, +85], P = 0,47          | −19                 | −0,7 pp                   |
| C × `testonce`           | −90 [−250, +67], P = 0,38          | −84 [−141, −38]     | −2,2 pp [−3,9, −0,7]      |


- **P(C > A) = 1,00:** em todos os 10 episódios, para todos os 16 pares de
seeds, o C vence. Replica nas 12 seeds inéditas (+1653 [+1072, +2313]).
- **B e C empatam** — as features temporais não contribuem, agora com IC.
- **C empata com a melhor política fixa** na recompensa, com 23% menos exames e
2,2 pontos a menos de acurácia (significativo).
- **O desvio de ±119 escondia o comportamento do A:** ele desaba em episódios
específicos (seeds 150, 250, 550, até −2200); o C nunca. Com a 4ª seed, o A é
+684, não +729.

**Custo honesto:** com 3–4 seeds o nível superior do bootstrap tem pouca
resolução (3 seeds admitem só 10 reamostras distintas); os intervalos são
honestos sobre o surto e otimistas sobre a seed. A cobertura no nível dos
episódios foi verificada: 93,5% em 1000 simulações para IC nominal de 95%,
igual a um bootstrap ingênuo de referência.

---



## 14. Robustez: anos e dificuldade espacial

O orientador pediu comparações além do kriging de 2015-16: "clamping, desvios
de uniforme", bootstrapping, robustez a cenários e o
[synthetic-pop](https://github.com/Mosqlimate-project/synthetic-pop). Todas as
medições abaixo são **sem retreino** (agentes do v4 no kriging v8).

**Por ano.** A chikungunya quase não circulou no Rio em 2015 (70 notificações,
contra 14 mil em 2016), então "2015" troca só a dengue
(`build_kriging_surfaces --years-dengue 2015 --years-chik 2016`; reconstruir a
referência pelo mesmo código dá superfície idêntica à em uso).


|                         | referência (15+16) | dengue 2015 + chik 2016 | Rio 2016          |
| ----------------------- | ------------------ | ----------------------- | ----------------- |
| C                       | +2692              | +2710                   | +2721             |
| C × A                   | +2009, P = 1,00    | +2077, P = 0,98         | +2022, P = 1,00   |
| C × `testonce`          | −90 [−250, +67]    | −72 [−236, +73]         | −61 [−226, +96]   |


Idêntico — e dava para prever antes de rodar: a posição sozinha acerta a doença
em **53–54% nas três superfícies** (acurácia de Bayes com prior igual,
`position_bayes_accuracy`), e a dengue de 2015 tem correlação 0,90 com a de
2016. É consistência entre anos, não generalização: a referência já contém
2015. Não vale treinar nesse eixo.

**Dificuldade espacial ("desvios de uniforme").** A superfície do kriging é
quase plana — razão de 8× entre a célula mais e a menos provável, nenhuma
zerada. `transform_surfaces` ganhou três botões (YAML: `surface_temperature`,
`surface_clamp`, `surface_mix_uniform`); a temperatura sozinha (p ∝ p^τ) cobre
de "a posição não diz nada" até o nível do sintético:


| τ   | posição acerta | C     | A    | C × A            | C × `testonce`      |
| --- | -------------- | ----- | ---- | ---------------- | ------------------- |
| 0   | 0,50           | +2718 | +670 | +2047, P = 0,97  | −65 [−242, +114]    |
| 1   | 0,54 (treino)  | +2692 | +691 | +2001, P = 1,00  | −90 [−250, +68]     |
| 4   | 0,65           | +2651 | +503 | +2149, P = 0,95  | −131 [−468, +115]   |
| 8   | 0,74           | +2396 | +277 | +2119, P = 0,89  | −387 [−1123, +162]  |
| 32  | 0,94           | +2686 | +541 | +2145, P = 0,95  | −96 [−740, +227]    |


- A vantagem do crédito vale **em toda a faixa** (C × A ≈ +2000, P ≥ 0,89).
- O C **não explora** a geografia mais informativa — curva plana, esperado de
quem treinou em τ = 1. Só treinando nos pontos da curva se sabe se aprenderia.
- Fora da distribuição o C fica mais frágil por seed: em τ = 8, as seeds 45 e
47 caem (+1405, +2303), enquanto 46 e 48 seguem acima de +2900.
- E a pergunta que a curva levantou: se a geografia a 94% não muda nada, por
que o sintético (também 94%) muda tanto? Resposta no §15.

---



## 15. O kriging não tinha casos "outro"

Os baselines não olham a posição, e mesmo assim o `testonce` fazia +2782 no
kriging contra +473 no sintético. Comparando os mundos na mesma seed: **483
casos no sintético, 360 no kriging — com exatamente a mesma dengue e chik.** A
diferença eram 123 casos "outro" (não-arbovirose).

O `KrigingWorld` **ignorava `other_prevalence` em silêncio**: o
`kriging_v8.yaml` declarava 25%, o mundo gerava zero. A docstring do próprio
gerador sintético descreve a consequência: sem casos "outro", um laudo negativo
de dengue implica chikungunya, o segundo exame é redundante e descartar nunca é
correto. Todos os resultados no kriging até o v8 foram medidos nesse ambiente
em que **um exame sempre basta**.

**Corrigido no `kriging_v9.yaml`** (`kriging_other_cases: true`): mesmo modelo
do sintético — espaço uniforme, quantidade proporcional aos arbovirais do dia.
A correção é opt-in para o v8 continuar reproduzível; sem ela, declarar
`other_prevalence` no kriging agora emite aviso. Três testes travam: a
prevalência declarada aparece, o v8 não muda e avisa, e os arbovirais por dia
são idênticos com e sem os casos "outro".

**Os baselines no v9 voltam à ordem do sintético** (10 seeds):


| agente       | kriging v8 | **kriging v9** | sintético v8 |
| ------------ | ---------- | -------------- | ------------ |
| `testtwice`  | +1051      | **+1225**      | **+1249**    |
| `testonce`   | **+2782**  | +322           | +473         |
| `clinical`   | −64        | −345           | −355         |
| `confirmall` | +346       | −1734          | −1821        |


A geografia não mudou; só os casos "outro". Isso também explica a falha do C
transferido para o sintético (−340): ele nunca tinha visto um caso "outro".

**De passagem:** os YAMLs de `experiments/configs/env/` nunca estiveram no git —
a regra `env/` do `.gitignore` (template de virtualenv) os pegava. Corrigido
com `!experiments/configs/env/`; as 21 configs agora são versionadas.

---



## 16. Resultado no v9: o crédito supera a melhor política fixa

A e C retreinados no v9, treino idêntico ao v4 (300 mil passos), **uma seed
cada** (45) por falta de tempo. Checkpoint final, benchmark de 10 seeds.


| agente                          | recompensa [IC 95%]         | acurácia | exames/episódio |
| ------------------------------- | --------------------------- | -------- | --------------- |
| **C (crédito por caso)**        | **+2387** [+2186, +2564]    | 0,938    | **675**         |
| `testtwice` (melhor fixa no v9) | +1225 [+1131, +1315]        | 0,938    | 978             |
| `testonce`                      | +322                        | 0,756    | 489             |
| `clinical`                      | −345                        | 0,576    | 0               |
| **A (GAE padrão)**              | **−1972** [−2897, −1028]    | 0,563    | 6               |


- **C × `testtwice`: +1163 [+980, +1330], P = 1,00** — vence nos 10
episódios, com a **mesma acurácia** e **31% menos exames**. O ambiente
corrigido cobra decidir *quais* casos precisam do segundo exame, e o C o faz.
No v8 ele só empatava com a melhor política fixa; aqui a supera.
- **O A colapsa:** ~6 exames por episódio, abaixo de não agir (A × `clinical`:
−1627, P = 0,10). C × A: +4359, P = 1,00.
- **Limitação:** com uma seed, o IC cobre só a variação entre surtos. No v8 o
desvio entre seeds foi ~50 no C e ~130 no A — ordens de grandeza abaixo das
diferenças acima, mas não é substituto de mais seeds.

Duas afirmações ainda **não medidas**: que o C escolhe os casos certos para o
segundo exame (falta a análise por caso), e por que o A colapsa (hipótese: a
cadeia exame → laudo → segundo exame → conclusão ficou mais longa, e o GAE
padrão a apaga).

---



## 17. Próximos passos

### Feito nesta etapa

- ~~Adotar o kriging como referência, refazer a varredura econômica, treinar
mais o PPO~~ (§12). ~~Um episódio = um caso~~ — descartado: a intercalação foi
resolvida no **learner** (§12). ~~Separar crédito de observação temporal~~ —
o crédito explica o ganho (§12, §13).
- ~~Intervalos de confiança~~ (§13), ~~consistência entre anos~~ e
~~varredura de dificuldade espacial~~ (§14), ~~casos "outro" no kriging~~ e
~~retreino no v9 com 1 seed~~ (§15, §16).
- Artigo (`Artigo/main.tex`) reescrito com a metodologia atual, os resultados
do v9 e uma seção de pendências marcada com `	odo{}`.

### Imediatos e baratos (minutos, sem treino)

**1. Baseline fixo sequencial** — testar dengue e só testar chik se o laudo for
negativo. É a política fixa natural entre `testonce` e `testtwice`, e o
primeiro contra-argumento de um revisor ao resultado do §16.

**2. Quais casos o C testa duas vezes.** Transforma "o C escolhe os casos
certos" (§16) de inferência em medição, e vira a figura de interpretabilidade
do artigo.

**3. Refazer sobre o v9 o que foi medido no v8:** bootstrap com seeds inéditas,
anos, varredura de temperatura, robustez de `results/demo` (médico, custo,
atraso) e a varredura econômica de `test_cost`. As ferramentas já existem;
basta apontar para o `kriging_v9.yaml`.

### Com treino

**4. Mais seeds no v9** (≥ 4 por braço; ~3h30 por treino, 2–3 em paralelo).
Sem isso o §16 fica sem variação entre seeds.

**5. Treinar A e C em pontos da curva de temperatura** (ex.: τ = 8 e 32):
o crédito vence o GAE quando treinado em cada cenário? E aprende a usar a
geografia quando ela informa? Depois: transferência entre cenários e treino em
mistura.

**6. Cidades via synthetic-pop.** Dá população do Censo 2022 com endereços
(CNEFE), mas não simula doença: falta um modelo de atribuição de doença sobre a
população. Também daria a densidade populacional que os casos "outro" deveriam
seguir (hoje uniformes, inclusive sobre o mar).

### Modelagem em aberto

**7. Sazonalidade.** O SEIR não tem forçamento sazonal; com R0 = 1,25 a
epidemia leva ~9 meses.

**8. R0 da chikungunya.** As faixas de literatura se sobrepõem às da dengue
(1,46–1,67 contra 1,25–1,70); antes a chik era menor por construção. Confirmar
com o orientador.

**9. Externalidade do exame** não capturada pelo crédito por caso (§12).

**10. Busca automática de recompensa e hiperparâmetros** (proposta 1):
Optuna com TPE e pruner ASHA, objetivo = média sobre ≥ 3 seeds **menos** o
desvio.

### O que não vale a pena

Continuar ajustando o DQN — `lr`, `buffer`, `n_step` e `target_update_freq`
não moveram nada; o ganho veio de mudar de algoritmo. E treinar nas superfícies
por ano (§14): mesma dificuldade, mesmo resultado.

---



## 18. Estado do código

- **288 testes passando** (154 no início desta investigação).
- Modelo epidêmico em `dengue_envs/core/epi_model.py`, com as fontes citadas no
cabeçalho. `epi_model: legacy` continua o padrão; os ambientes v8/v9 pedem
`seir`.
- Ambientes (agora versionados): **`kriging_v9.yaml` (referência atual, com
casos "outro")**, `kriging_v8.yaml` (resultados do v4, sem casos "outro"),
`kriging_v8_temporal.yaml`, `synthetic_v8.yaml`. Os v7 seguem no lugar.
- Gerador kriging (`dengue_envs/data/kriging_generator.py`):
`kriging_other_cases` (casos "outro", opt-in), `transform_surfaces` +
`position_bayes_accuracy` (dificuldade espacial), anos por doença em
`build_kriging_surfaces --years-dengue/--years-chik`. Superfícies em
`results/kriging/` (referência, `rio_d2015_c2016_*`, `rio_2016_*`).
- Crédito por caso: decomposição no ambiente
(`dengue_diagnostics.py`/`case_by_case.py`), GAE por caso em
`agents/ppo/credit.py`. Ligado por `train.per_case_credit: true`, que exige
`per_case_reward` no ambiente.
- Avaliação: `experiments/bootstrap.py` (IC hierárquico e pareado;
`python -m experiments.bootstrap [conjunto ...]`, conjuntos `benchmark`,
`benchmark_v9`, `seeds_ineditas`, `dengue_2015`, `rio_2016`,
`temperatura_<τ>`) e `experiments/robustez.py` (caches em `results/demo/`).
- Existe um `agents/ppo/ppo.py` (script standalone do CleanRL, anterior à
sessão) que não conflita, mas convive confusamente com o pacote novo.

```bash
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m agents.ppo.train --config experiments/configs/train/ppo_v5c_credito_s45.yaml
.venv/Scripts/python.exe -m experiments.evaluate --config experiments/configs/benchmark_ppo_v5c_credito_s45.yaml
.venv/Scripts/python.exe -m experiments.bootstrap benchmark_v9
```

**Resultados brutos:** v9 — `results/bm_ppo_v5{a_gae,c_credito}_s45`,
`results/baseline_v9_kriging`; v8 — `results/bm_ppo_v4a_seir_s4*`,
`results/bm_ppo_v4b_credito_s4*`, `results/bm_ppo_v4c_credito_sem_tempo_s4*`,
`results/baseline_v8seir_{kriging,synthetic}`; bootstrap em
`results/bootstrap/`; robustez em `results/demo/`.

**Nota operacional:** o treino não retoma de onde parou. Uma reinicialização do
Windows no meio da noite custou duas seeds inteiras, que precisaram ser
refeitas do zero. Treinos longos devem ser lançados junto com uma tarefa que
espere o fim e encadeie o benchmark.
