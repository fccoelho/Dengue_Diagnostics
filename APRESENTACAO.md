# Reunião — estado do trabalho

Respostas aos pontos levantados na última conversa, com os números que
sustentam cada uma. Todos reprodutíveis (seeds fixas, mesmo ambiente para todos
os agentes). Detalhamento técnico em `ACHADOS.md`.

---

## Resumo em uma página

| Pergunta | Resposta curta |
|---|---|
| Por que o agente vai melhor no kriging? | **Não vai.** 97% da variância vem do médico sorteado, não do ambiente. |
| Dá para chegar a 90% de acurácia? | **Não com os parâmetros atuais.** O teto de informação do ambiente é ~88%. |
| Sensib./especif. por Beta, com 3 níveis | **Implementado**, com os 3 níveis e configs prontos. |
| Fluxo de decisão de cada estratégia | **Documentado** (seção 4) — inclui uma restrição estrutural que não estava explícita. |
| Melhorar o mapa por kriging | **Em aberto** — preciso de orientação. |

Um achado não previsto, e o mais consequente: **a função de recompensa estava
premiando testar tudo**, o que contradizia a premissa do trabalho. Foi
encontrado, corrigido e validado (seção 3).

---

## 1. O agente não vai melhor no kriging — é ruído

Verificações:

1. A competência clínica sorteada é **idêntica** nos dois ambientes para a mesma
   seed (é amostrada antes da criação do mundo). Média 0,7177 nos dois.
2. A composição dos casos é idêntica: 280 casos, 50% dengue / 50% chik.
3. Ainda assim, o baseline clínico — que **não usa informação espacial** — muda
   de 0,690 para 0,731 de acurácia. Logo, a diferença não pode vir da
   distribuição espacial.

O número que fecha o argumento:

| Correlação com a recompensa do episódio | |
|---|---:|
| Competência do médico × recompensa do **DQN** | **+0,974** |
| Competência do médico × recompensa do **clínico** | **+0,977** |

Com `clinical_specificity ~ U(0,5 , 0,95)`, cada episódio é um problema de
dificuldade completamente diferente. Com 10 episódios, o ruído domina o sinal.

**Correção:** fixar o nível clínico médio (Beta, seção 5) permite comparar
ambientes em dificuldade equiparada. E usar ≥30 seeds.

---

## 2. 90% de acurácia não é alcançável — é limite do ambiente, não do algoritmo

Duas restrições estruturais:

**(a) Cada caso recebe UMA única ação.** Verificado empiricamente: nenhum caso é
apresentado ao agente mais de uma vez. O agente **testa OU decide**, nunca os
dois.

**(b) O laudo é imperfeito:** sensibilidade = especificidade = 0,9;
inconclusivo = 0,1.

Disso sai o teto:

```
acc_max ≈ (1 − p_inconclusivo)·sens_lab + p_inconclusivo·acc_clínica
        =        0,9          ·  0,9    +      0,1      · acc_clínica
        ≈ 0,88
```

Confirmado com um baseline novo, `testall`, que testa **todos** os casos (custo
máximo, política irreal — serve só como teto):

| Ambiente | Acurácia do `testall` |
|---|---:|
| synthetic | **0,873** |
| kriging | **0,884** |

**Nenhuma política pode superar isso de forma relevante.** Para chegar a 90%,
muda-se o ambiente:

- reduzir `inconclusive_prob` (0,1 → 0,02) — realista para RT-PCR;
- aumentar sensibilidade/especificidade do laudo (0,9 → 0,95–0,99);
- **ou** permitir que o caso volte ao agente depois do laudo — que também
  responde à observação "se ele tem o resultado do teste, deveria usá-lo".

---

## 3. A recompensa premiava testar tudo (corrigido)

Com o baseline `testall` no benchmark, o ranking ficou:

| Agente | Recompensa | Acurácia | Testes |
|---|---:|---:|---:|
| **testall** (trivial) | **−35,5** | 0,873 | 280 |
| dqn (treinado) | −395,9 | 0,753 | 103 |
| clinical | −702,8 | 0,690 | 0 |
| random | −2288,3 | 0,633 | 91 |

**Uma política trivial superava o DQN treinado em 11×.**

**Causa:** a penalidade de −10 só valia para casos **nunca testados**. Pedir
exame virava um "passe livre" — isentava da punição mesmo quando o laudo voltava
inconclusivo e não mudava nada. O teste era premiado pelo *ato de ser pedido*,
não pela informação que agregava.

**Correção:**

| Parâmetro | Antes | Agora |
|---|---:|---:|
| penalidade por erro, **testado ou não** | não existia | **−3,0** |
| penalidade extra por não ter testado | −10,0 | **0,0** |

**Resultado:** o ótimo deixou de ser trivial e passou a depender do estado:

| Qualidade do médico | nada | testall | confirmall | melhor |
|---|---:|---:|---:|:--|
| 0,50 – 0,64 (ruim) | −384 | **−188** | −2326 | **testar** |
| 0,72 – 0,90 (bom) | +56 | −140 | **+1414** | **confirmar** |

A melhor política **fixa** rende +129; um **oráculo** que conhecesse a competência
do médico renderia **+599**. Essa diferença é o que justifica um agente aprendido.

---

## 4. Fluxo de decisão (para escrever no artigo)

Cada caso é apresentado ao agente **exatamente uma vez**, no dia em que é
notificado, e recebe **uma** das 6 ações:

| Ação | Custo | É decisão? | Efeito |
|---|---:|:--:|---|
| testar dengue / chik (0,1) | 1,0 | não | laudo chega após o atraso e **atualiza sozinho** o diagnóstico; o agente não decide de novo sobre o caso |
| epi confirm (2) | 0,5 | não | agrega evidência epidemiológica; não sofre penalidade de decisão |
| nada (3) | 0,1 | não | mantém o palpite clínico |
| confirmar (4) | 0,0 | **sim** | +10 se certo, −20 se errado (com atraso) |
| descartar (5) | 0,0 | **sim** | +10 se de fato não era arbovirose; **−30** se era caso real |

Placar final: +1 por acerto; −3 por erro.

Isso esclarece a dúvida "se ele tem o resultado do teste, deveria usá-lo": o
resultado **é** usado — automaticamente, quando volta do laboratório —, mas o
agente não toma uma segunda decisão sobre aquele caso. Já o agente aleatório
decide antes de qualquer laudo, o que explica seu desempenho catastrófico.

**O que o DQN treinado realmente faz** (medido):

| Ação | Frequência |
|---|---:|
| confirmar | 47,6% |
| testar chik | 23,8% |
| epi confirm | 16,9% |
| testar dengue | 11,8% |
| nada | 0,0% |
| descartar | 0,0% |

Aprendeu a nunca ficar parado e a nunca descartar (coerente com o −30), mas
confirma demais quando o médico é ruim.

---

## 5. Sensibilidade e especificidade por Beta, com 3 níveis (implementado)

Uma Beta para a **sensibilidade** e outra para a **especificidade** clínicas,
parametrizadas pela média (`a = μκ`, `b = (1−μ)κ`), amostradas **por médico**:

| Nível | sens. média | espec. média |
|---|---:|---:|
| `low` | 0,60 | 0,60 |
| `medium` | 0,75 | 0,75 |
| `high` | 0,90 | 0,90 |

A concentração κ (padrão 20) controla a dispersão entre médicos (≈ ±0,09).
Aceita também médias explícitas — inclusive os 85%/60% citados no artigo.

```yaml
env:
  clinical_quality: medium      # ou low / high
  # clinical_quality: {sensitivity: 0.85, specificity: 0.60}
```

Ganho conceitual: a confusão clínica agora é **assimétrica e interpretável** —
sensibilidade é P(diz dengue | é dengue), especificidade é P(diz chik | é chik) —
em vez de uma taxa única nos dois sentidos.

Configs prontos: `experiments/configs/env/clinical_{low,medium,high}.yaml`.

---

## 6. Resultados por nível clínico (15 seeds fora do treino)

| Nível | Melhor política | DQN treinado nesse nível | Testes do DQN |
|---|---|---:|---:|
| **low** | testall (−158,7 / acc **0,858**) | **−317,6** (acc 0,591) | **0** |
| **medium** | confirmall (**+602,5**) | **−24,0** | **0** |
| **high** | confirmall (**+2023,7**) | **+1148,6** (acc 0,892) | 21 |

**Resultado negativo e claro: o DQN perde para a melhor política trivial nos três
níveis.** Mais revelador: ele **quase não testa**. No nível `low`, onde testar é
comprovadamente certo (`testall` tem a melhor recompensa **e** a maior acurácia),
o agente testa zero vezes e fica pior do que não fazer nada.

**Implicação metodológica importante:**

> Com o nível clínico **fixo**, o problema de RL fica quase trivial. O oráculo
> adaptativo **empata** com a melhor política fixa em `high` (2023,7 vs 2023,7) e
> quase empata em `medium` (623,1 vs 602,5).

Os dois cenários se complementam: **níveis fixos** para caracterizar ambiente e
baselines sem confundidor; **cenário misto** (médico variando muito) para
justificar o agente, pois é lá que inferir o médico vale (+129 fixo vs +599
oráculo).

---

## 7. Por que o DQN não converge

Mesmo com nível fixo e 20 episódios de avaliação, as curvas oscilam sem
convergir (no `high`: **+1019 → −7245** entre épocas consecutivas). Duas medições
explicam:

**(a) Recompensa por passo muito dispersa:** média +1,3, desvio **16,1**,
variando de **−90 a +210**. Os desfechos atrasados de dezenas de casos maturam no
mesmo dia. (O placar final responde por apenas ~6% da magnitude.)

**(b) Variância irredutível para um agente sem memória:** fixando o nível
`medium`, a competência do médico ainda varia (0,59–0,90 pela Beta). Para a
**mesma política fixa**, a recompensa do episódio variou de **−524 a +1550**.

> Para a mesma observação, o retorno varia milhares de pontos por causa de uma
> variável latente que o agente não enxerga. O alvo de TD fica irredutivelmente
> ruidoso. **Não é hiperparâmetro — é o problema estando mal-observado.**

### Correção proposta e implementada (v3)

Tornar a competência do médico **observável**: a cada laudo que volta, comparar o
resultado com o palpite clínico. A informação já existia, mas apenas implícita e
espalhada num mapa 400×400 esparso — inviável de extrair por convolução.

A observação ganhou 2 features: **taxa de concordância** (laudo × palpite
clínico) e **força da evidência**. Validação: a estimativa observável acompanha a
competência real com **correlação +0,991**.

Efeito colateral desejável: para estimar o médico, o agente **precisa testar**
alguns casos no início — surge um trade-off explorar/explorar que antes não
existia. Isso dá sentido epidemiológico ao teste: ele não serve só para
diagnosticar aquele caso, serve para **calibrar a confiança na triagem clínica**.

**Status:** treino interrompido em ~11 de 20 épocas. O checkpoint parcial rende
−60,5 (acurácia 0,750) e — sinal relevante — **voltou a testar (100 casos**, contra
0 dos modelos por nível). Ou seja, a feature **mudou o comportamento na direção
esperada**, mas ainda não superou o `confirmall` (+128,8). A curva seguia
instável, o que aponta a dispersão da recompensa (item **a** acima) como próximo
suspeito.

---

## 8. Estado do artigo — precisa de atualização

O texto atual está defasado em relação ao código:

| No artigo | No código |
|---|---|
| recompensa +5 / −15 | +10 / −20 / −30 + placar final |
| buffer de 50 000 | 5 000–6 000 |
| "100 cenários" | a tabela reporta 10 |
| sensib. 85% / especif. 60% fixas | Beta com 3 níveis |
| sem baseline clínico | `clinical`, `testall`, `confirmall` |
| sem kriging | kriging implementado e avaliado |
| 16 GB de RAM | 32 GB |

Também: Introdução e Metodologia em inglês, Resultados em português.

A tabela de resultados atual (acurácia ~0,71 para todos os agentes) foi obtida
com a recompensa antiga e **precisa ser refeita**.

---

## 9. Trabalho de engenharia (viabilizou os experimentos)

| Item | Antes | Depois |
|---|---:|---:|
| Parâmetros da rede | 69,4 M | **1,25 M** |
| Checkpoint | ~555 MB | **5 MB** |
| Velocidade do ambiente | 20 passos/s | **134 passos/s** |
| RAM do treino | 18,8 GB | **7,2 GB** |
| Velocidade do treino | — | **2× mais rápido** |

Quatro estouros de memória foram diagnosticados e corrigidos, entre eles: dois
grids 400×400 float64 sendo gravados no `info` de **cada** transição (≈51 GB no
buffer) e um buffer de avaliação de 12,8 GB que o Tianshou alocava em silêncio.

Suíte de testes: **137 passando**.

---

## 10. Pontos para decidir na reunião

1. **Escala da recompensa.** A dispersão por passo (−90 a +210) é o próximo
   suspeito da instabilidade. Reduzir a escala dos desfechos de decisão?
2. **O caso deve poder voltar ao agente após o laudo?** Isso resolveria a
   pergunta sobre "usar o resultado do teste" e é o caminho mais direto para
   furar o teto de 88%.
3. **Parâmetros do laudo.** Manter 0,9/0,9 com 10% de inconclusivos, ou adotar
   valores de RT-PCR (0,95–0,99)? Define se 90% é atingível.
4. **Kriging:** o que exatamente melhorar na construção do mapa?
5. **Desenho experimental:** níveis fixos para caracterizar + cenário misto para
   justificar o agente — faz sentido?
