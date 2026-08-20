# Achados para a reunião — respostas às perguntas do orientador

Documento de diagnóstico produzido a partir dos experimentos. Cada seção
responde a um ponto levantado na conversa. Todos os números são reprodutíveis
(10 seeds fixas, mesmo ambiente para todos os agentes).

---

## 1. Por que o agente vai melhor no "problema mais difícil" (kriging)?

**Resposta: ele não vai. A diferença é ruído de um fator não controlado — a
qualidade do médico.**

Evidências:

1. **A qualidade clínica sorteada é idêntica nos dois ambientes.** Ela é
   amostrada *antes* da criação do mundo, então para a mesma seed o valor é o
   mesmo em `synthetic` e `kriging` (média 0,7177 nos dois).
2. **A composição dos casos é idêntica:** 280 casos, 50% dengue / 50% chik.
3. **Mesmo assim a acurácia do baseline clínico difere** (0,690 vs 0,731) — e
   esse baseline *nem usa informação espacial*. Ou seja, a diferença não pode
   vir da distribuição espacial: é realização aleatória das moedas de confusão
   clínica, com apenas 10 episódios.

O ponto decisivo:

| | correlação com a recompensa |
|---|---:|
| Qualidade do médico × recompensa do **DQN** | **+0,974** |
| Qualidade do médico × recompensa do **clínico** | **+0,977** |

**97% da variação dos resultados é explicada pelo médico sorteado, não pelo
agente.** Com `clinical_specificity ~ U(0,5 , 0,95)`, cada episódio é um
problema de dificuldade radicalmente diferente, e 10 episódios não bastam para
separar sinal de ruído.

**Correção (já implementada — é exatamente a sugestão da Beta):** fixar o nível
clínico médio permite comparar ambientes em dificuldade equiparada. Ver seção 5.

---

## 2. Dá para chegar a 90% de acurácia?

**Resposta: não com os parâmetros atuais. O teto do ambiente é ~88%, e isso é
uma limitação de informação, não do algoritmo de RL.**

Duas restrições estruturais:

**(a) Cada caso recebe UMA única ação.** Verificado empiricamente: nenhum caso é
apresentado ao agente mais de uma vez (o caso aparece só no dia em que é
notificado). Logo o agente **testa OU decide** — nunca os dois.

**(b) O laudo é imperfeito:** `sensibilidade = especificidade = 0,9` e
`inconclusivo = 0,1`.

Disso sai o teto analítico:

```
acc_max ≈ (1 − p_inconclusivo) · sens_lab + p_inconclusivo · acc_clínica
        =        0,9           ·   0,9    +      0,1       · acc_clínica
        ≈ 0,81 + 0,1 · acc_clínica  ≈  0,88
```

Confirmado empiricamente com o novo baseline `testall` (testa **todos** os casos
— custo máximo, política irreal):

| Ambiente | Acurácia do `testall` |
|---|---:|
| synthetic | **0,873** |
| kriging | **0,884** |

**Nenhuma política pode superar isso de forma relevante.** Para chegar a 90% é
preciso mudar o *ambiente*, não o algoritmo:

- reduzir `inconclusive_prob` (0,1 → 0,02) — realista para RT-PCR;
- aumentar sensibilidade/especificidade do laudo (0,9 → 0,95–0,99) — realista
  para RT-PCR/NS1;
- **ou** permitir que o caso volte ao agente após o laudo (testar no dia *t*,
  decidir em *t + atraso*). Esta é a mudança mais profunda e também resolve a
  observação "se ele tem o resultado do teste deveria usá-lo" (seção 4).

---

## 3. ⚠️ A recompensa premiava testar tudo — **corrigido** (ver 3.1)

Ao adicionar o baseline trivial `testall`, o ranking ficou:

| Agente | Recompensa | Acurácia | Testes/episódio |
|---|---:|---:|---:|
| **testall** (trivial) | **−35,5** | **0,873** | 280 |
| dqn (treinado) | −395,9 | 0,753 | 103 |
| clinical | −702,8 | 0,690 | 0 |
| random | −2288,3 | 0,633 | 91 |

**Uma política trivial de testagem universal supera o DQN treinado em 11×.**

A causa está nos pesos: deixar um caso errado e **não testado** custa −10,
enquanto um teste custa apenas 1,0. Valor esperado por caso:

| Qualidade clínica | não fazer nada | testar | confirmar | melhor |
|---:|---:|---:|---:|:--|
| 0,50 | −4,60 | −0,14 | −9,50 | testar |
| 0,60 | −3,50 | −0,13 | −5,40 | testar |
| 0,72 | −2,18 | −0,12 | −0,48 | testar |
| 0,80 | −1,30 | −0,11 | **+2,80** | confirmar |
| 0,90 | −0,20 | −0,10 | **+6,90** | confirmar |

Consequências:

1. **"Não fazer nada" nunca é ótimo** — mas é exatamente o que o baseline
   clínico faz. Ele é um piso fraco demais.
2. **A política ótima depende da qualidade do médico**, que o agente *não
   observa* (só poderia inferir comparando laudos com o palpite clínico ao longo
   do episódio). É, na prática, um POMDP.
3. **Isso contradiz a premissa do artigo.** O texto vende "manter acurácia
   gastando menos testes", mas a função de recompensa, como estava, dizia "teste
   tudo".

### 3.1 Correção aplicada (recompensa v2)

A distorção tinha uma causa precisa: a penalidade de −10 só valia para casos
**nunca testados**. Pedir exame virava um "passe livre" — isentava da punição
mesmo quando o laudo voltava inconclusivo e não mudava nada. O teste era
premiado pelo *ato de ter sido pedido*, não pela informação que agregava.

Correção (em `dengue_envs/core/reward.py`):

| Parâmetro | v1 (antes) | v2 (agora) |
|---|---:|---:|
| `penalty_misdiagnosed` — erro, **testado ou não** | — (não existia) | **−3,0** |
| `penalty_untested_misdiagnosed` — extra por não testar | −10,0 | **0,0** |

Resultado verificado empiricamente — o ótimo passou a **depender do estado**:

| Qualidade do médico | nada | testall | confirmall | melhor |
|---|---:|---:|---:|:--|
| 0,50 – 0,64 (ruim) | −384 | **−188** | −2326 | **testar** |
| 0,72 – 0,90 (bom) | +56 | −140 | **+1414** | **confirmar** |
| **média (10 seeds)** | −95 | −142 | **+129** | — |

- `testall` **deixou de ser ótimo** (−142; antes era −35 e campeão).
- "Não fazer nada" também não é ótimo em lugar nenhum.
- A melhor política **fixa** rende +129; um **oráculo** que conhecesse a
  qualidade do médico renderia **+599**. Essa diferença (+470) é exatamente a
  margem que o agente de RL tem para aprender — e ele só consegue capturá-la
  inferindo a competência do médico a partir do histórico de laudos. É um
  problema de decisão sob observação parcial genuíno, e agora bem-posto.

Treino em andamento com essa recompensa: `experiments/configs/train/dqn_v2.yaml`
(saída em `results/dqn_v2/`; o checkpoint v1 fica preservado como referência).

### 3.2 Bug de memória encontrado ao lançar o treino

O Tianshou aloca silenciosamente um buffer para o *collector de teste* quando
nenhum é passado:

```python
buffer = VectorReplayBuffer(DEFAULT_BUFFER_MAXSIZE * len(env), len(env))  # 10 000 * n_envs
```

Com 2 envs de teste e o mapa 4×400×400, isso são **20 000 transições ≈ 12,8 GB**
— só para avaliar, mais do que o próprio buffer de treino. Era a maior fatia de
RAM do processo (18,8 GB no total) e não aparecia em nenhuma configuração.

Corrigido em `agents/deepq/train.py`: o buffer de teste passou a ser
dimensionado pelo tamanho real de um episódio (`test_buffer_per_env`, padrão
1000). Consumo do treino caiu de **18,8 GB → 7,2 GB**, o que permitiu triplicar
o buffer de treino (2 000 → 6 000) sem risco de estouro. Efeito colateral: o
treino ficou **~2× mais rápido** (17 it/s), e 100 mil passos passaram a levar
~100 min em vez de várias horas.

### 3.3 Resultado do primeiro treino v2 — e um segundo problema metodológico

Benchmark honesto (10 seeds independentes, recompensa v2):

| Agente | Recompensa |
|---|---:|
| **dqn (v2)** | **+68,6** |
| confirmall | **+128,8** |
| clinical | −95,2 |
| testall | −142,0 |
| random | −1687,1 |

O DQN saiu do vermelho (+68,6, primeiro lugar entre os agentes do benchmark),
mas **ainda não supera a política fixa `confirmall` (+128,8)** — e está longe do
oráculo adaptativo (+599). Ou seja: ele ainda não aprendeu a inferir a
competência do médico.

A curva de treino explica parte disso — ela **não converge**:

```
-140, -251, -4983, -105, -396, -62(melhor), -183, -1088, ... -3700, -739, -3320
```

Causa: a avaliação usava **2 episódios** (o valor estava amarrado a
`num_test_envs`). Como a recompensa de um episódio varia de +1992 a −2326
conforme o médico sorteado, o `test_reward` media **qual médico calhou**, não a
qualidade da política — e o `save_best_fn` acabava premiando um sorteio de
sorte. É o mesmo confundidor da seção 1, agora contaminando a *seleção de
modelo*.

Correções aplicadas:

1. `test_episodes` desacoplado de `num_test_envs` (padrão **20**);
2. treinos com **nível clínico fixo** (`clinical_quality`), que elimina a maior
   fonte de variância — em andamento para `low`, `medium` e `high`
   (`experiments/configs/train/dqn_v2_{low,medium,high}.yaml`).

Também foi adicionado o baseline **`confirmall`**, que é a política fixa mais
forte sob a v2 e portanto a barra real a ser superada.

### 3.4 Por que o DQN não converge — diagnóstico de fundo

Mesmo com **nível clínico fixo** e **20 episódios** de avaliação, as curvas
continuaram oscilando sem convergir (no nível `high`, de **+1019** para
**−7245** entre épocas consecutivas). Isso descarta o sorteio do médico como
única causa e aponta para algo estrutural.

Duas medições explicam o fenômeno:

**(a) A recompensa por passo é muito dispersa.** Num episódio típico: média
+1,3, desvio **16,1**, variando de **−90 a +210** por passo. Isso ocorre porque
os desfechos atrasados de vários casos maturam no mesmo dia — um único passo
carrega o resultado de dezenas de decisões. (O placar final, ao contrário do que
se poderia supor, responde por apenas ~6% da magnitude.)

**(b) A variância é irredutível para um agente sem memória.** Mesmo fixando o
nível `medium`, a competência do médico continua sendo sorteada da Beta
(observado: 0,59 a 0,90). Para a **mesma política fixa** (`confirmall`), a
recompensa do episódio variou de **−524 a +1550** (desvio 610), com correlação
+0,59 com a competência sorteada.

Ou seja: **para a mesma observação, o retorno varia em milhares de pontos por
causa de uma variável latente que o agente não enxerga.** O alvo de TD fica
irredutivelmente ruidoso e o Q não tem como convergir. Não é um problema de
hiperparâmetro — é o problema estando mal-observado.

**Consequência para o artigo (e para a arquitetura):** a política ótima exige
*acumular evidência sobre a competência do médico ao longo do episódio* —
comparando os laudos que voltam com o palpite clínico. A arquitetura atual (CNN
sem memória, decidindo caso a caso a partir de um mapa esparso) não tem como
computar essa estatística.

Caminho recomendado, do mais barato ao mais estrutural:

1. **Expor a estatística na observação** (mais barato e provavelmente
   suficiente): incluir a taxa corrente de concordância entre laudo e palpite
   clínico — uma estimativa direta e suficiente da competência do médico. Hoje a
   informação existe apenas implícita, espalhada num mapa 400×400 esparso, o que
   é quase impossível de extrair por convolução.
2. **Dar memória ao agente** (DRQN/recorrência), para que ele mesmo construa a
   estatística.
3. **Reduzir a escala/dispersão da recompensa**, para diminuir a magnitude dos
   erros de TD.

### 3.5 Resultado dos 3 níveis clínicos (15 seeds novas, fora do treino)

| Nível | Agente | Recompensa | Acurácia | Testes |
|---|---|---:|---:|---:|
| **low** | testall | **−158,7** | **0,858** | 280 |
| | clinical (nada) | −205,6 | 0,591 | 0 |
| | **dqn** | −317,6 | 0,591 | **0** |
| | confirmall | −809,6 | 0,591 | 0 |
| | *[oráculo]* | *−56,5* | — | — |
| **medium** | confirmall | **+602,5** | 0,740 | 0 |
| | **dqn** | −24,0 | 0,738 | **0** |
| | clinical (nada) | −39,5 | 0,740 | 0 |
| | testall | −141,1 | **0,874** | 280 |
| | *[oráculo]* | *+623,1* | — | — |
| **high** | confirmall | **+2023,7** | 0,889 | 0 |
| | **dqn** | +1148,6 | **0,892** | 21 |
| | clinical (nada) | +127,7 | 0,889 | 0 |
| | testall | −124,0 | 0,889 | 280 |
| | *[oráculo]* | *+2023,7* | — | — |

**Resultado negativo, e claro: o DQN perde para a melhor política trivial nos
três níveis.** Mais revelador: ele **quase não testa** (0, 0 e 21 testes). No
nível `low`, onde testar é comprovadamente a resposta certa (`testall` é o melhor
com −158,7 e a maior acurácia, 0,858), o agente testa **zero** vezes e fica em
−317,6 — pior até do que não fazer nada (−205,6). Ele não aprendeu a política
apropriada a cada regime.

Uma implicação metodológica importante para o desenho experimental:

> **Com o nível clínico FIXO, o problema de RL fica quase trivial.** O oráculo
> adaptativo empata com a melhor política fixa em `high` (2023,7 vs 2023,7) e
> quase empata em `medium` (623,1 vs 602,5). Só em `low` há folga real
> (−56,5 vs −158,7).

Ou seja: os 3 níveis são ótimos para **caracterizar o ambiente e os baselines**
de forma controlada — mas a contribuição de RL só é interessante no cenário
**misto** (competência do médico variando bastante), onde inferir o médico vale
muito (+129 fixo contra +599 do oráculo). Os dois cenários se complementam:
níveis fixos para medir, cenário misto para justificar o agente.

### 3.6 Correção proposta: tornar a competência do médico observável (v3)

Se o problema é que a variável decisiva é latente, a correção mais direta é
**torná-la observável**. Toda vez que um laudo volta, comparar o resultado com o
palpite clínico informa sobre a competência do médico — a informação já existe,
mas apenas implícita e espalhada num mapa 400×400 esparso, o que é inviável de
extrair por convolução.

Implementado (`context_features`, opcional e **desligado por padrão**, para não
invalidar checkpoints antigos). A observação ganha um vetor de 2 posições:

| Feature | Significado |
|---|---|
| taxa de concordância | fração dos laudos informativos que confirmaram o palpite clínico (0,5 enquanto não há laudo) |
| força da evidência | quantos laudos já voltaram, saturando em 30 |

Laudos **inconclusivos** são excluídos: não alteram o diagnóstico e não informam
nada sobre o médico.

Validação: a estimativa observável acompanha a competência real (latente) com
**correlação +0,991** (8 episódios, testando todos os casos).

| Médico real | Estimado |
|---:|---:|
| 0,502 | 0,470 |
| 0,592 | 0,578 |
| 0,722 | 0,676 |
| 0,876 | 0,780 |
| 0,904 | 0,821 |

Efeito colateral desejável: para estimar o médico o agente **precisa testar**
alguns casos no início do episódio — surge naturalmente um trade-off
explorar/explorar, que antes não existia.

Treino em andamento: `experiments/configs/train/dqn_v3_context.yaml`
(cenário misto, onde a adaptação vale mais). Cobertura: 5 testes novos em
`dengue_envs/tests/test_context_features.py` — 137 no total.

---

## 6. RT-PCR + retorno do caso após o laudo — o teto de 90% foi rompido

Duas mudanças estruturais, feitas juntas porque atacam a mesma limitação.

### 6.1 Parâmetros do exame (RT-PCR)

Os valores do laboratório estavam fixos no código (não eram configuráveis) e
eram pessimistas para um teste molecular:

| Parâmetro | Antes | Agora (RT-PCR) |
|---|---:|---:|
| sensibilidade | 0,90 | **0,95** |
| especificidade | 0,90 | **0,98** |
| inconclusivos | 0,10 | **0,02** |

Agora configuráveis por YAML: `lab_sensitivity`, `lab_specificity`,
`lab_inconclusive_prob`.

> ⚠️ **Pendência para o artigo:** os valores são plausíveis para RT-PCR, mas
> ainda precisam ser ancorados em referência da literatura — a faixa varia com
> kit, dia de coleta e sorotipo. Como são configuráveis, trocá-los por valores
> citáveis não exige mexer em código.

### 6.2 O caso volta ao agente depois do laudo

Antes, cada caso era apresentado **uma única vez**: pedir exame significava
abrir mão de decidir sobre aquele caso (o resultado era aplicado
automaticamente e o agente nunca mais o via). Era essa a origem da observação
*"se ele tem o resultado do teste, deveria usá-lo"*.

Agora, quando o laudo chega, o caso **retorna** ao agente para decisão, com
limite de `max_case_revisits` (padrão 2) para evitar laços de testes repetidos.
Casos já confirmados/descartados não retornam.

### 6.3 Resultado: a meta de 90% foi superada

| Cenário | Acurácia (testando todos) |
|---|---:|
| Antigo (0,9/0,9, 10% inconcl., sem retorno) | 0,873 |
| **Novo (RT-PCR + retorno)** | **0,957** |

Panorama das políticas (10 seeds):

| Política | Recompensa | Acurácia | Testes |
|---|---:|---:|---:|
| nada (clínico puro) | −95,2 | 0,690 | 0 |
| confirmall | +128,8 | 0,690 | 0 |
| 1 teste → confirmar | +2419,2 | 0,960 | 280 |
| **seletiva: 2º teste só se inconclusivo** | **+2441,6** | **0,963** | 285 |
| 2 testes → confirmar (sempre) | +2237,8 | 0,970 | 560 |

Observações importantes:

1. **A meta de ~90% do orientador foi superada** (0,96), e a maior parte do ganho
   veio da simples correção dos parâmetros do exame (0,873 → 0,957), não do
   algoritmo.
2. **A política seletiva é a melhor** e é genuinamente inteligente: repete o
   exame **apenas** quando o primeiro veio inconclusivo (+22,4 sobre a fixa, com
   só 5 testes a mais). Há, portanto, comportamento não trivial a aprender.
3. **Mas a margem é pequena** (+22 sobre +2419, ~1%). Com o exame custando 1,0 e
   uma decisão correta valendo +10, testar todo mundo uma vez passou a ser
   quase sempre certo — ou seja, **não há escassez**, e sem escassez não existe
   problema de alocação.

### 6.4 Consequência para a premissa do trabalho

O artigo estuda **alocação de testes sob escassez**. Com os parâmetros atuais,
testar todos uma vez é praticamente ótimo — o dilema desaparece. Para restaurá-lo:

- **subir o custo do teste** (1,0 → ~8,0), ou
- **impor um teto diário de exames** (capacidade laboratorial finita) — mais
  realista epidemiologicamente e o que transforma o problema em priorização
  genuína: *quais* casos merecem o exame de hoje.

A segunda opção parece a mais alinhada com a motivação do artigo, mas é decisão
de pesquisa.

### 6.5 Exame encarecido — escassez restaurada (ambiente v2)

O custo do exame estava fixo no código (1,0) e agora é configurável
(`test_cost`). Calibrei por **varredura empírica**, medindo em quantos episódios
"testar" vence "confirmar direto":

| Custo do exame | Melhor política fixa | Episódios em que TESTAR vence |
|---:|---:|:--|
| 1,0 | +2419 | 10/10 (médicos até 0,90) |
| 3,0 | +1859 | 9/10 |
| 5,0 | +1299 | 9/10 (até 0,88) |
| **7,0** | **+739** | **5/10** (até 0,76) ← equilíbrio |
| 9,0 | +179 | 4/10 (até 0,65) |

**Escolhido `test_cost: 7.0`**: é onde a decisão se divide meio a meio entre os
episódios, ou seja, onde acertar a escolha rende o máximo. Com custo 1 ou 5, a
resposta certa era quase sempre a mesma ("teste") e não havia problema de
alocação.

Panorama final do ambiente v2 (10 seeds):

| Política | Recompensa | Acurácia | Testes |
|---|---:|---:|---:|
| nada (clínico puro) | −95,2 | 0,690 | 0 |
| confirmall | +128,8 | 0,690 | 0 |
| **1 teste → confirmar** (melhor fixa) | **+739,2** | **0,960** | 280 |
| **[oráculo por episódio]** | **+995,0** | — | — |

**Margem do oráculo sobre a melhor política fixa: +255,8** (≈35%). É o que um
agente adaptativo tem a capturar — e, ao contrário do cenário anterior, agora a
margem é substancial *e* a acurácia já supera a meta de 90%.

Ambiente: `experiments/configs/env/synthetic_v2.yaml`.
Treinos: `dqn_v4.yaml` (com features de contexto) e `qlearning_v2.yaml`.

### 6.6 O `testall` ficou degenerado — trocado pelo `testonce`

Com os casos retornando, o `testall` deixou de ser um teto e virou desperdício:
como **nunca confirma**, o mesmo caso volta e é testado de novo até o limite de
revisitas, pagando o exame três vezes (**−5647,6** no v2).

A referência correta passou a ser o novo baseline **`testonce`**: pede o exame na
primeira avaliação e **confirma quando o laudo chega** (+739,2, acurácia 0,960).
É a política fixa mais forte e a barra que um agente aprendido precisa superar.

### 6.7 Primeiro treino no v2 — resultado negativo

Benchmark (10 seeds):

| Agente | Recompensa |
|---|---:|
| confirmall | **+128,8** |
| clinical (nada) | −95,2 |
| qlearning | −1850,2 |
| dqn (parcial, ~5 de 20 épocas) | −2057,4 |
| random | −3031,8 |
| testall (degenerado) | −5647,6 |

**Os dois agentes treinados ficaram muito abaixo até de "não fazer nada".** Ambos
gastam exames demais — a um custo de 7 por exame, testar sem necessidade é
ruinoso.

Ressalvas importantes, para não superinterpretar:

- **O DQN não é avaliação justa:** o treino foi interrompido em ~5 de 20 épocas
  (processo encerrado junto com a sessão), ainda em regime de alta exploração
  (ε alto). Relançado para completar.
- **O Q-Learning completou** os 400 episódios, mas **não convergiu**: a
  recompensa foi de −5023 para ~−2000 e o número de estados **ainda crescia** no
  último episódio (3 237). É a maldição da dimensionalidade que o próprio artigo
  argumenta — só que agora com evidência quantitativa.
- A tarefa ficou **mais difícil** de propósito: com o exame caro, errar a
  alocação custa caro. Isso é o desejado (existe escassez), mas exige treino
  mais longo.

### 6.8 Escala da recompensa — atacando a causa da divergência

Todas as rodadas do DQN até aqui oscilaram sem convergir. O suspeito apontado na
§3.4 era a **magnitude** da recompensa, e a medição confirma:

| | sem escala | com escala 0,05 |
|---|---:|---:|
| desvio por passo | **20,9** | 1,05 |
| faixa por passo | **[−287, +121]** | [−14, +6] |
| retorno do episódio | ~±1000–2000 | ~±50–100 |

Pedir a uma rede que preveja alvos na casa dos milhares com erro quadrático
produz gradientes enormes — é a receita clássica de divergência do DQN, e
explica saltos como +1019 → −7245 entre épocas consecutivas.

Implementado `RewardScaleWrapper` (`reward_scale: 0.05`), aplicado **apenas no
treino**. Multiplicar a recompensa por uma constante positiva **não altera a
política ótima**; só coloca os alvos de TD numa escala tratável (desvio por
passo ≈ 1, que é o usual em DQN). É a mesma motivação do *reward clipping* do
DQN original, sem distorcer as proporções entre ações.

A avaliação e o benchmark rodam **sem** a escala, para que os números continuem
comparáveis com os resultados anteriores (verificado: razão exata de 0,050 entre
as duas configurações).

**Efeito:** as oscilações violentas sumiram. A curva passou a subir até a época 4
(melhor ponto) e depois degradar suavemente — um modo de falha diferente
(convergência para política subótima), não mais divergência numérica.

### 6.9 Resultado final no ambiente v2

Benchmark (10 seeds, escala original):

| Agente | Recompensa | Acurácia | F1 | Testes | **Custo/acerto** |
|---|---:|---:|---:|---:|---:|
| **testonce** (melhor fixa) | **+739,2** | **0,960** | 0,964 | 280,0 | 1,037 |
| **dqn** | **+330,3** | 0,784 | 0,810 | **98,6** | **0,449** |
| confirmall | +128,8 | 0,690 | 0,692 | 0 | — |
| clinical | −95,2 | 0,690 | 0,692 | 0 | — |
| qlearning | −1850,2 | 0,580 | 0,732 | 66,8 | 0,337 |
| random | −3031,8 | 0,593 | 0,671 | 132,5 | 0,668 |

**O que deu certo:**

1. A escala de recompensa levou o DQN de **−2057 → +330**. Ele passou a superar
   `confirmall` e o clínico puro.
2. **O agente usa o retorno do caso exatamente como projetado.** Distribuição de
   ações medida:

   | | 1ª visita (n=1400) | Retorno pós-laudo (n=474) |
   |---|---:|---:|
   | confirmar | 64,4% | **100%** |
   | testar chik | 33,9% | — |
   | nada | 1,7% | — |

   Ou seja: decide se vale o exame; quando o laudo volta, **sempre** decide de
   posse do resultado. É exatamente o fluxo pedido na reunião.
3. **Melhor custo por acerto entre os agentes que testam** (0,449 contra 1,037 do
   `testonce`): faz 65% menos exames e ainda assim eleva a acurácia de 0,690
   (clínico) para 0,784.

**O que ainda não deu certo (e é o ponto honesto):**

1. **O DQN continua abaixo do `testonce`** (+330 vs +739). Sob os pesos atuais,
   testar todos uma vez ainda rende mais do que sua seleção.
2. **A seleção não é adaptativa.** Correlação entre competência do médico e taxa
   de exames: **+0,310** — isto é, ele testa *ligeiramente mais* quando o médico
   é *melhor*, o oposto do esperado. Na prática a taxa é quase constante
   (30%–41%). Ele aprendeu **quanto** testar, mas não **quando**.
3. **O Q-Learning é o pior agente treinado** (−1850, acurácia 0,580 — abaixo do
   clínico, 0,690). Ele chega a piorar o diagnóstico. Não convergiu: o número de
   estados ainda crescia (3 237) no último episódio.

**Conclusão:** o mecanismo está certo e o agente já extrai valor dele, mas a
decisão *de quem testar* continua não informada — é exatamente a lacuna que as
features de contexto deveriam preencher e ainda não preenchem.

---

## 7. Causa raiz das falhas do DQN: horizonte de desconto

Depois de três intervenções que não mudaram o comportamento do agente
(penalidade por caso não resolvido, *reward shaping* e máscara de ação), a
medição abaixo explica **todas elas de uma vez**:

| | |
|---|---:|
| Passos por episódio | **373** |
| Horizonte efetivo com γ = 0,99 (`1/(1-γ)`) | **100** passos |
| Desconto de uma recompensa que chega no fim (`γ³⁷³`) | **0,024** |

O placar final domina a recompensa total (a penalidade por caso não resolvido
sozinha vale −10 × 373 = −3730), mas chega ao agente multiplicado por **2%**.

> **O sinal que deveria ensinar "conclua seus casos" está fora do horizonte de
> desconto.** Não é um problema de incentivo nem de exploração: é de escala
> temporal.

Isso explica retroativamente por que cada correção falhou:

- `penalty_unresolved = -10`: aplicado no passo terminal → invisível.
- `shaping_conclude_bonus`: ajuda, mas o desfecho que ele antecipa também estava
  descontado.
- Máscara de ação: nunca chegou a ativar, porque o agente parou de testar.

Comportamento resultante do agente (v6): **94,7% "nada"**, 0% de exames, e apenas
**5,2% dos casos concluídos** — pior que o clínico puro.

| γ | Horizonte | Desconto no fim do episódio |
|---:|---:|---:|
| 0,99 | 100 | 0,024 |
| **0,999** | **1000** | **0,689** |
| 0,9999 | 10000 | 0,963 |

**Correção em teste (v7):** γ = 0,999, que põe o horizonte (1000) acima do
comprimento do episódio (373). É uma mudança de uma linha, mas é a que ataca a
causa medida — as anteriores tratavam sintomas.

**Lição metodológica para o artigo:** em ambientes com recompensa concentrada no
fim do episódio e episódios longos, o fator de desconto precisa ser escolhido em
função do comprimento do episódio, não adotado por convenção. O valor 0,99 é
padrão na literatura de Atari, onde os episódios têm centenas de passos mas a
recompensa é distribuída ao longo deles — condição que este ambiente não
satisfaz.

**Reprodutibilidade preservada:** o cenário antigo é recuperável por YAML
(`lab_sensitivity: 0.9`, `lab_specificity: 0.9`, `lab_inconclusive_prob: 0.1`,
`max_case_revisits: 0`) e reproduz exatamente a acurácia de 0,873.

---

## 4. Fluxo de decisão (o que precisa ser escrito no artigo)

O fluxo real, verificado no código:

- Cada caso é apresentado ao agente **exatamente uma vez**, no dia em que é
  notificado. O agente escolhe **uma** das 6 ações.
- `testar dengue (0)` / `testar chik (1)` — custo 1,0. O laudo chega após
  `lab_delay_days` e **atualiza sozinho** o diagnóstico do caso. O agente
  **não volta a decidir sobre esse caso**.
- `epi confirm (2)` — custo 0,5. Agrega evidência epidemiológica; **não é
  decisão** e não sofre as penalidades de decisão.
- `nada (3)` — custo 0,1. Mantém o palpite clínico.
- `confirmar (4)` — aceita o diagnóstico atual. Decisão: +10 se certo, −20 se
  errado (com atraso de `reward_delay_days`).
- `descartar (5)` — afirma "não é arbovirose". +10 se de fato não era; **−30**
  se era doença real (falso negativo de vigilância).
- Placar final: +1 por diagnóstico correto; −10 por caso errado **que nunca foi
  testado**.

Isso esclarece a dúvida "se ele tem o resultado do teste, deveria usá-lo": o
resultado **é** usado — automaticamente, ao chegar do laboratório —, mas o
agente não toma uma segunda decisão sobre aquele caso. Já o agente aleatório
decide antes de qualquer laudo, o que explica seu desempenho catastrófico.

**O que o DQN treinado realmente faz** (medido em 5 seeds):

| Ação | Frequência |
|---|---:|
| confirmar | 47,6% |
| testar chik | 23,8% |
| epi confirm | 16,9% |
| testar dengue | 11,8% |
| nada | 0,0% |
| descartar | 0,0% |

Ele aprendeu a **nunca** ficar parado e a **nunca** descartar (coerente com o
−30), mas confirma demais em episódios de médico ruim — é o que derruba sua
recompensa.

---

## 5. Sensibilidade/especificidade por Beta, com 3 níveis (implementado)

Implementado conforme sugerido, em `dengue_envs/core/clinical.py`:

- Uma Beta para a **sensibilidade** e outra para a **especificidade** clínicas,
  parametrizadas pela **média** (`a = μκ`, `b = (1−μ)κ`), sorteadas por médico.
- Três níveis médios prontos:

| Nível | sens. média | espec. média |
|---|---:|---:|
| `low` | 0,60 | 0,60 |
| `medium` | 0,75 | 0,75 |
| `high` | 0,90 | 0,90 |

- A concentração κ (padrão 20) controla a dispersão entre médicos
  (κ=20 ⇒ desvio ≈ ±0,09). Também aceita médias explícitas, p.ex. os valores do
  artigo: `{sensitivity: 0.85, specificity: 0.60}`.

Uso (YAML):

```yaml
env:
  clinical_quality: medium          # ou low / high
  # clinical_quality: {level: high, concentration: 50}
  # clinical_quality: {sensitivity: 0.85, specificity: 0.60}
```

Configs prontos: `experiments/configs/env/clinical_{low,medium,high}.yaml`.
O caminho antigo (`clinical_specificity`) continua funcionando — 130 testes
passam.

Ganho conceitual: a confusão clínica agora é **assimétrica e interpretável** —
sensibilidade é P(diz dengue | é dengue) e especificidade é P(diz chik | é chik),
em vez de uma taxa única nos dois sentidos.

---

## 8. Espaço de ações redesenhado: concluir uma classe, não só confirmar/negar

### 8.1 O problema identificado

Pergunta levantada: o ideal seria o agente aprender a **olhar os dados e
concluir sem testar**, não apenas aprender "quanto" testar. Investiguei se o
ambiente sequer permitia isso.

Medi quanta informação existe **sem nenhum exame**:

| Fonte | Acurácia |
|---|---:|
| Palpite clínico sozinho | 0,530 |
| **Só a posição (x,y)** | **0,683** |
| Posição + palpite clínico | 0,704 |
| *(referência: 1 exame)* | 0,756 |

A posição espacial sozinha supera o médico em 15 pontos — sinal forte e
usável, consistente com dengue e chik terem focos espaciais distintos.

Mas havia uma barreira estrutural: a ação `confirm` só podia **aceitar** o
`agent_diagnosis` corrente — não existia forma de o agente dizer "o médico
disse dengue, mas os dados indicam chikungunya" sem gastar um exame para
"editar" o diagnóstico. Medido: uma política que infere pelos dados e nunca
testa chegava a 0,588 de acurácia, contra 0,704 que o mesmo classificador
atingiria se pudesse atribuir a classe diretamente — **11 pontos inacessíveis
por falta de ação**, não por falta de sinal.

### 8.2 A correção: três ações conclusivas, uma por classe

`confirm`/`discard` (ações 4/5) foram substituídas por três ações que
alegam a classe diretamente — a alegação vem da **própria ação**, não do
`agent_diagnosis` corrente:

| Ação | Significado |
|---|---|
| 4 — `conclude_dengue` | "este caso é dengue" |
| 5 — `conclude_chik` | "este caso é chikungunya" |
| 6 — `conclude_other` | "este caso não é arbovirose" |

Regra de recompensa unificada (`claimed = action_id − 4`):
- `claimed == verdade` → `reward_correct_decision` (+10);
- `claimed == OTHER` numa doença real → `penalty_missed_case` (−30, falso
  negativo de vigilância, o pior erro — preserva a semântica antiga do
  `discard`);
- qualquer outro erro (arbovirose trocada, ou arbovirose alegada num caso
  que era OTHER) → `penalty_incorrect_decision` (−20).

Espaço de ações: `Discrete(6)` → `Discrete(7)`. A rede se adapta sozinha
(`action_shape` é lido de `env.action_space.n`), então não houve mudança na
arquitetura da `DengueNet`.

### 8.3 Validação: a inferência sem teste agora compensa

Refiz o experimento anterior, agora usando as ações conclusivas para o
classificador atribuir a classe **diretamente**, sem gastar nenhum exame:

| Política | Recompensa | Acurácia | Exames |
|---|---:|---:|---:|
| confirmar sempre (sem dados, sem exame) | −3823,6 | 0,375 | 0 |
| **infere dos dados e conclui direto** | **+131,4** | **0,711** | **0** |
| 2 exames em todos → conclui | +955,0 | 0,938 | 746 |

Um salto de quase **4000 pontos** só por poder agir sobre a inferência —
confirmando que o espaço de ações novo viabiliza exatamente o comportamento
pedido. Falta o DQN aprendê-lo (ver §8.5).

### 8.4 Consequências para os agentes fixos e testes

`confirmall`, `testonce` e `testtwice` foram reescritos: em vez de uma ação
`CONFIRM` fixa, agora calculam `4 + agent_diagnosis` para concluir com o
diagnóstico corrente — preserva o comportamento pretendido de cada baseline
sob a nova semântica. `testall` segue degenerado (documentado desde a v2).

Cobertura: `dengue_envs/tests/test_core_reward.py` e `test_other_disease.py`
foram reescritos para a nova semântica de 3 ações conclusivas (o teste antigo
`test_incorrect_confirm_is_penalized`, por exemplo, dependia de mutar
`agent_diagnosis` manualmente — o que não afeta mais o resultado da decisão,
já que a alegação vem da ação). Suíte: **153 testes passando**.

### 8.5 Currículo em andamento

Hipótese: o ambiente v3 (3 classes, exame caro, decisão forçada após 2
exames) tem uma cadeia de crédito difícil demais para aprender do zero (ver
§7). Lancei um currículo de 2 fases:

1. **Fase 1** (`dqn_v9_phase1.yaml`) — treina do zero no ambiente v2
   (binário, sem `OTHER`), onde o DQN já demonstrou aprender algo (+330 antes
   desta mudança de ação).
2. **Fase 2** (`dqn_v9_phase2.yaml`) — carrega os pesos da fase 1
   (`init_from`) e continua o treino no ambiente v3, com epsilon menor e
   decaindo mais rápido (fine-tuning, não redescoberta do zero).

O warm-start é arquiteturalmente direto: as duas fases usam
`context_features: true` (vetor de contexto de 14 posições) e
`Discrete(7)`, e a máscara de ação do v3 é aplicada pelo Tianshou sobre os
Q-values — a rede em si não lê `obs["mask"]`, então o `state_dict` é
idêntico nas duas fases. Validado com smoke test antes do treino completo.

### 8.6 Resultado do currículo — negativo, e revela uma causa nova

O currículo **não ajudou**. As duas fases colapsaram para "não fazer nada",
pior que o `dqn_v8` sem currículo:

| Fase | Ambiente | Recompensa | Ação dominante |
|---|---|---:|---:|
| Fase 1 | v2 (binário) | **−2312,3** | nada (77,6%) |
| Fase 2 (warm-start) | v3 | **−3711,7** | nada (86,1%) |
| *referência: `dqn_v4`* | *v2, config antiga* | *+330,3* | *testava e concluía* |
| *referência: `dqn_v8`* | *v3, sem currículo* | *−2910,3* | *testava (73,3 exames)* |

O mais informativo: a Fase 1 **já falha sozinha**, no ambiente v2 — o mesmo
onde o `dqn_v4` original tinha sucesso. Como o espaço de ações novo foi
validado separadamente (§8.3, o classificador sklearn ganha ~4000 pontos com
ele), o suspeito não é a mudança de ação — é outra coisa que mudou junto.

**Hipótese em teste:** o γ = 0,999 (herdado do `dqn_v8`/§7) pode ter sido
generalizado sem necessidade. Medido para o comprimento de episódio do v2
(~280 passos):

| γ | Horizonte efetivo | Desconto no fim do episódio (γ²⁸⁰) |
|---:|---:|---:|
| 0,99 (`dqn_v4`, funcionou) | 100 | 0,060 |
| 0,999 (usado aqui, falhou) | 1000 | 0,756 |

Com γ = 0,999 quase não há desconto ao longo do episódio inteiro — os alvos
de TD passam a depender de um horizonte muito mais longo, e é plausível que
100 mil passos de treino não bastem para estabilizar isso (a rede prefere a
ação "seguro" — nada, custo −0,1 — enquanto as estimativas de longo prazo
ainda são ruidosas). A lição de §7 (γ alto ajuda quando o episódio é longo)
pode ter sido superestendida: nunca testamos γ = 0,999 isolado do resto do que
mudou entre v6→v8, e agora há evidência de que γ = 0,999 quebra até o
ambiente mais simples.

Reexecutando a Fase 1 com γ = 0,99 (`dqn_v10_phase1.yaml`), mantendo tudo o
mais igual, para isolar essa variável antes de repetir a Fase 2.

### 8.7 γ = 0,99 recupera a política ativa — mas só parcialmente

Confirmado: γ era um fator real. Com γ = 0,99, a distribuição de ações mudou
por completo:

| | γ = 0,999 (colapsado) | γ = 0,99 (corrigido) |
|---|---:|---:|
| nada | 77,6% | **2,2%** |
| testar chik | 0,4% | **45,6%** |
| epi confirm | 1,1% | 9,6% |
| concluir dengue | 18,7% | 16,9% |
| concluir chik | 0,0% | 25,6% |

A política passou a usar **todo** o espaço de ações de forma diferenciada,
em vez de se refugiar na inação.

Só que a recuperação foi **parcial**, não total:

| Agente | Recompensa |
|---|---:|
| testonce | +722,2 |
| confirmall | +103,8 |
| **dqn (γ=0,99, novo espaço de ações)** | **−1070,4** |
| qlearning | −1622,6 |
| clinical | −2895,2 |
| *referência: `dqn_v4` (6 ações, config antiga)* | *+330,3* |

O DQN corrigido já supera Q-Learning e o clínico puro, mas continua abaixo do
`confirmall` e muito abaixo do `dqn_v4` original no mesmo ambiente v2. γ
explica o colapso total, mas não fecha sozinho a distância até o resultado
anterior — resta uma lacuna atribuível a outra causa (candidatos: o vetor de
contexto passou de 2 para 14 posições, ou o espaço de ações maior — 7 em vez
de 6 — exige mais treino para convergir, mesmo sendo estritamente mais
expressivo).

Prosseguindo para a Fase 2 (`dqn_v10_phase2.yaml`) com esta base corrigida —
a política já é sã o suficiente para valer a pena continuar o currículo, e o
warm-start deve ajudar a superar a lacuna restante no ambiente v3.

### 8.8 Resultado final do currículo — melhora real, mas não decisiva

| Versão | Recompensa | Acurácia | Exames |
|---|---:|---:|---:|
| `dqn_v8` (sem currículo, γ=0,999) | −2910,3 | 0,601 | 75,9 |
| `dqn_v9` (currículo, γ=0,999) | −3711,7 | 0,565 | 0,0 |
| **`dqn_v10` (currículo, γ=0,99)** | **−2843,9** | **0,625** | **268,0** |

O currículo com γ corrigido supera as duas tentativas anteriores em todas as
métricas — maior acurácia, mais exames feitos (evidência de engajamento real)
e a melhor recompensa dos três — mas a diferença sobre o `dqn_v8` é pequena
(+66,4, dentro do ruído entre seeds) e **nenhuma versão do DQN supera as
políticas fixas** (`testtwice` +955,0, `testonce` +247,4, `confirmall`
−1365,2 — todas acima do DQN).

A distribuição de ações no retorno pós-laudo (após o 1º exame) revela um
padrão saudável — testar dengue → testar chik → concluir — mas com uma
lacuna: **`conclude_other` nunca é usado no retorno (0%)**. Como
`other_prevalence = 0,25`, isso implica um teto de acurácia de ~0,75 mesmo
que o resto da política fosse perfeita (nunca acertando os 25% de casos
"outro"); a acurácia observada (0,625) fica abaixo até desse teto, indicando
que também há confusão residual entre dengue e chik.

**Conclusão do experimento de currículo:** ele evitou o colapso, mas não foi
suficiente para o DQN superar as políticas fixas simples no ambiente v3. O
gargalo não parece mais ser o horizonte de desconto (γ) nem a falta de
informação sobre o caso (case_features) — ambos já corrigidos — e sim a
dificuldade intrínseca da cadeia de decisão de 3 passos (testar → testar →
concluir corretamente entre 3 classes) dentro do orçamento de treino usado
(20 épocas / 100 mil passos por fase). Hipóteses para a próxima rodada:
mais épocas de treino, um `shaping_conclude_bonus` maior especificamente
para `conclude_other`, ou aceitar o resultado atual como evidência de que o
problema, no espaço de estados atual, favorece políticas fixas bem
desenhadas sobre RL — o que também é um resultado válido para o artigo.

---

## 9. Três correções estruturais e a conclusão sobre o DQN

### 9.1 Correções feitas (todas de bugs reais, medidos)

Partindo do déficit do v10, medi **onde** exatamente o agente perdia:

| | DQN v10 | testtwice |
|---|---:|---:|
| Casos concluídos | **64,3%** | 100% |
| Acurácia | 0,626 | 0,942 |
| Exames por caso | 63% com zero, 19% com três | 100% com dois |

Os 35,7% de casos órfãos custavam **−1332 por episódio (47% do déficit)**, e a
origem se dividia em duas falhas estruturais:

**(a) `epi_confirm` era um beco sem saída** (37,4% dos órfãos). A ação
adicionava evidência epidemiológica ao caso mas **não agendava revisita** — o
agente pagava por informação que nunca poderia usar. É exatamente a mesma
falha já corrigida para os exames (§6.2), que passou despercebida para esta
ação. Corrigido: `epi_confirm` agora agenda revisita.

**(b) A máscara de decisão nunca disparava** (52,3% dos órfãos). Ela exigia os
**dois** exames feitos; como o agente nunca testava dengue, a condição nunca
se satisfazia e o caso simplesmente esgotava as revisitas e sumia da fila.
Corrigido: a máscara também dispara na **última apresentação** do caso.

Resultado das duas: **órfãos de 35,7% → 3,8%**, e os remanescentes são todos
`nada` (abandono explícito, legítimo).

**(c) A rede não enxergava o próprio caso.** Medindo a sensibilidade dos
Q-values do modelo treinado:

| Perturbação | \|ΔQ\| | Muda a decisão? |
|---|---:|---|
| Diagnóstico do **próprio caso** (contexto) | 0,0 – 0,63 | **não** |
| Ruído aleatório no mapa | 207 – 731 | **sim** |

As magnitudes de entrada estavam equilibradas (mapa 1,86 / contexto 2,05), então
não era escala — era **capacidade**: a cabeça recebia 2304 dims de mapa contra
32 de contexto (96% × 1,3%). Consequência medida: **48,6% das conclusões
contrariavam a evidência do próprio exame, acertando apenas 20,7%** (contra
78,6% quando coerentes). Corrigido: mapa projetado para 128 dims e contexto
expandido para 128 — o contexto passou de 1,3% para **40%** da entrada.

> **Nota de honestidade:** a incoerência foi *introduzida por esta sessão*. O
> `confirm` antigo era coerente com o diagnóstico por construção; ao trocá-lo
> pelas três ações por classe (§8.2), ganhou-se a expressividade de discordar
> do médico com base nos dados — que era o objetivo — mas o agente passou a
> precisar *aprender* a coerência, e com 1,3% da capacidade não tinha como.

### 9.2 O resultado: nenhuma correção destravou o aprendizado

| Versão | Recompensa | Acurácia | Exames |
|---|---:|---:|---:|
| v8 (sem currículo, γ=0,999) | −2910,3 | 0,601 | 75,9 |
| v9 (currículo, γ=0,999) | −3711,7 | 0,565 | 0,0 |
| v10 (currículo, γ=0,99) | −2751,9 | 0,521 | 202,8 |
| **v11 (rede reequilibrada)** | **−3717,0** | 0,534 | 1,0 |
| *melhor política fixa (`testtwice`)* | *+955,0* | *0,938* | *746* |

Cada configuração converge para uma **política degenerada diferente** — ora
não testar nada, ora testar em excesso sem concluir, ora concluir ignorando a
evidência. Nenhuma se aproxima da política fixa trivial.

### 9.3 A causa raiz que resta: crédito por caso

A economia por caso é clara e favorece amplamente investigar:

| Escolha (por caso) | Valor esperado |
|---|---:|
| testar 2× e concluir | **+0,89** |
| concluir direto com o palpite clínico | −3,6 |
| `nada` (deixa o caso órfão) | −10,8 |

O agente deveria descobrir isso facilmente. Não descobre — e a razão está em
como a recompensa chega até ele:

- O `CaseByCaseWrapper` acumula as ações do dia e chama `env.step()` **uma vez
  por dia**. Dentro do dia, cada decisão de caso retorna **exatamente zero**; a
  consequência do dia inteiro cai num único passo.
- Medido: **90% da variância da recompensa de um passo vem de decisões tomadas
  ~5 dias antes, sobre outros casos** (desvio 17,07 contra 1,89 do custo da
  ação atual).

Ou seja, o sinal que o agente recebe ao decidir sobre o caso X é dominado por
desfechos de dezenas de **outros** casos. A função Q não tem como aprender o
valor de uma ação individual.

**A correção necessária** é atribuir a recompensa de cada caso ao passo daquele
caso. Isso exige separar duas coisas hoje acopladas no ambiente — *aplicar a
ação de um caso* e *avançar o dia* — e usar `reward_delay_days: 0` (mantendo o
atraso do **laudo**, que é o que cria a estrutura sequencial interessante).

É uma mudança no contrato central do ambiente e invalidaria os resultados
anteriores, por isso não foi feita sem decisão explícita.

---

## Recomendações (ordem de prioridade)

1. ~~Recalibrar a função de recompensa~~ — **feito** (seção 3.1). O ótimo deixou
   de ser trivial e passou a depender do estado. Treino v2 em andamento.
2. **Rodar os 3 níveis clínicos** com mais seeds (≥30). Isso remove o
   confundidor que hoje domina 97% da variância e torna a comparação
   synthetic × kriging legítima.
3. **Decidir sobre o teto de 90%**: sem mudar parâmetros do laudo (ou permitir
   decisão pós-laudo), a meta é inalcançável. Recomendo revisitar o caso de
   permitir que o caso retorne ao agente depois do resultado.
4. **Atualizar o artigo**, que está defasado: descreve recompensa +5/−15 (o
   código usa +10/−20/−30 + placar final), buffer de 50 000 (usa 5 000), 100
   cenários (a tabela reporta 10) e não inclui baseline clínico nem kriging.
5. **Melhorar a construção do mapa por kriging** — ponto levantado e ainda em
   aberto; precisa ser detalhado.
