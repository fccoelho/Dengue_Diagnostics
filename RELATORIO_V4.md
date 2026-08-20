# Relatório de progresso — agente de alocação de exames

Resumo do que foi implementado e medido desde a última conversa, com os
resultados do experimento atual (v4) e uma proposta de mudança no ambiente.

---

### 1 Exame com parâmetros de RT-PCR

| Parâmetro | Antes | Agora |
|---|---:|---:|
| sensibilidade | 0,90 | 0,95 |
| especificidade | 0,90 | 0,98 |
| laudos indeterminados | 0,10 | 0,02 |


### 2 Retorno do caso após o laudo

Quando o resultado do exame chega, o caso **volta ao agente** para decisão. Antes,
pedir exame significava abrir mão de decidir sobre aquele caso: o resultado era
aplicado automaticamente e o agente nunca mais o via. Era exatamente a origem da
observação *"se ele tem o resultado do teste, deveria usá-lo"*.

### 3 Exame encarecido (custo 1,0 → 7,0)

Com o exame barato, testar todo mundo era quase sempre ótimo — **não havia
escassez, e sem escassez não existe problema de alocação**, o que contradizia a
premissa do trabalho.

O custo foi calibrado por varredura empírica, medindo em quantos episódios
"testar" vence "confirmar direto":

| Custo | Melhor política fixa | Episódios em que testar vence |
|---:|---:|:--|
| 1,0 | +2419 | 10/10 |
| 5,0 | +1299 | 9/10 |
| **7,0** | **+739** | **5/10** ← equilíbrio |
| 9,0 | +179 | 4/10 |

Escolhido **7,0**: é onde a decisão se divide meio a meio, isto é, onde acertar a
escolha rende o máximo.

---

### 4 Resultados do experimento v4

Ambiente: 400×400, 60 dias, ~280 casos, atraso de 5 dias, médico ~U(0,50–0,95),
RT-PCR, retorno do caso, exame a 7,0. **10 cenários pareados** por semente.

| Agente | Recompensa | Acurácia | F1 | Exames | **Custo/acerto** |
|---|---:|---:|---:|---:|---:|
| *testar-uma-vez* (melhor política fixa) | **+739,2** | **0,960** | 0,964 | 280,0 | 1,037 |
| **DQN** | **+330,3** | 0,784 | 0,810 | **98,6** | **0,449** |
| *confirmar-tudo* | +128,8 | 0,690 | 0,692 | 0 | — |
| clínico puro (piso) | −95,2 | 0,690 | 0,692 | 0 | — |
| Q-Learning tabular | −1850,2 | 0,580 | 0,732 | 66,8 | 0,337 |
| aleatório | −3031,8 | 0,593 | 0,671 | 132,5 | 0,668 |
| *[oráculo por episódio]* | *+995,0* | — | — | — | — |

### 5 O que deu certo

**O agente usa o mecanismo de retorno exatamente como projetado:**

| Ação | 1ª avaliação | Retorno pós-laudo |
|---|---:|---:|
| confirmar | 64,4% | **100,0%** |
| solicitar exame | 33,9% | — |
| aguardar | 1,7% | — |
| descartar | 0,0% | 0,0% |

Ele decide se o exame se justifica e, quando o laudo volta, **sempre conclui de
posse do resultado**.

**Melhor custo por diagnóstico correto entre as políticas que testam:** 0,449
contra 1,037 da melhor política fixa — 65% menos exames, elevando a acurácia de
0,690 (clínico) para 0,784.

Um ponto técnico que destravou o treino: a recompensa por passo tinha desvio ≈21
e chegava a −287, com retornos na casa dos milhares. Isso fazia o DQN divergir
(oscilações de +1019 para −7245 entre épocas). Reescalando a recompensa **apenas
no treino** por um fator constante — o que não altera a política ótima — o agente
saiu de −2057 para **+330**.

### 6 O que ainda não deu certo

1. **O DQN não supera a melhor política fixa** (+330 vs +739).
2. **A seleção não é adaptativa.** A correlação entre competência do médico e
   taxa de testagem é **+0,310** — ele testa *ligeiramente mais* quando a triagem
   é *melhor*, o oposto do desejável. Na prática a taxa é quase constante
   (30%–41%). **Ele aprendeu *quanto* testar, mas não *quem* testar.**
3. **O Q-Learning tabular é o pior agente treinado** (−1850, acurácia 0,580 —
   abaixo do próprio clínico, 0,690: chega a degradar a triagem). Não convergiu:
   o número de estados visitados ainda crescia no último episódio (3.237). É
   evidência quantitativa da maldição da dimensionalidade.

---

## 7 Proposta: incluir casos que não são arboviroses

Esta é a mudança que considero mais importante para a próxima rodada, e ela
resolve **dois problemas de uma vez**.

### 7.1 O problema

Hoje o gerador produz **apenas dengue e chikungunya** — a verdade do ambiente
nunca contém "outro". Isso tem duas consequências indesejadas:

**(a) Um único exame resolve tudo, e o segundo é redundante.** Com só duas
classes, um laudo negativo para dengue *implica logicamente* chikungunya:

```
teste dengue NEGATIVO  →  só resta chik  →  diagnóstico resolvido
```

Por isso a política trivial "um exame e confirma" já atinge 0,960 de acurácia, e
solicitar o segundo exame acrescenta quase nada (0,970) por o dobro do custo. A
investigação não tem profundidade: não existe o caso em que o agente precisa
decidir se vale aprofundar.

**(b) A ação "descartar" é inútil.** Como a verdade nunca é "outro", descartar é
**sempre** erro e incorre na penalidade mais pesada (−30). Todos os agentes
aprenderam a nunca usá-la — 0% de uso em todas as medições. Ou seja, temos uma
ação no espaço de decisão que é decorativa.

### 7.2 A proposta

Fazer o gerador produzir também **síndromes febris que não são arbovirose**, com
alguma prevalência (por exemplo, 20–30% dos casos notificados). Isso é
epidemiologicamente realista: na triagem real, boa parte dos casos suspeitos não
se confirma.

### 7.3 O que isso muda

**A investigação passa a ser genuinamente sequencial:**

```
teste dengue NEGATIVO  →  pode ser chik OU não-arbovirose  →  AMBÍGUO
                       →  vale um segundo exame? ou descartar?
```

O laudo negativo deixa de resolver o caso. O agente passa a enfrentar a decisão
que hoje não existe: **até onde aprofundar a investigação**. O segundo exame
deixa de ser redundante e vira informativo.

**A ação "descartar" vira uma decisão legítima:** com dois laudos negativos, o
caso provavelmente não é arbovirose, e descartá-lo passa a ser a resposta certa —
inclusive economizando recursos.

**A acurácia deixa de ser binária disfarçada.** Hoje a métrica se chama
"multiclasse" mas a verdade só tem duas classes; qualquer "outro" atribuído conta
como erro. Com a mudança, ela passa a medir de fato o que promete.

**O paralelo com a prática fica mais forte:** o valor da vigilância não está só
em separar dengue de chik, mas em identificar o que **não** é arbovirose e não
deve consumir recurso nem entrar na estatística de surto.

---

## 8. Outras pendências

- **Ancorar em referência os parâmetros de RT-PCR** (item 2.2).
- **Capacidade laboratorial finita:** hoje a escassez é modelada só pelo custo
  unitário. Um teto diário de exames — refletindo a capacidade real da rede —
  transformaria o problema em priorização explícita: não apenas *se* vale
  investigar, mas *quais* casos merecem a capacidade daquele dia.
- **Seleção informada:** é o gargalo central do agente. Duas direções — dar
  memória explícita (redes recorrentes) ou verificar por ablação se as variáveis
  de contexto já fornecidas estão de fato sendo usadas pela rede.

---