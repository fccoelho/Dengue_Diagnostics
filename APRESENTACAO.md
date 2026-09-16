# Diagnóstico do agente de vigilância — o que descobrimos

Resumo executivo. Detalhamento em `RESUMO_SESSAO.md` e `REFATORACAO_AMBIENTE.md`.

---

## 1. O achado central

> **O ambiente sintético resolvia o problema sozinho, pela geografia.**

Medimos quanto a **posição** de um caso, sozinha, prediz a doença:


| distribuição                   | posição acerta a doença |
| ------------------------------ | ----------------------- |
| sintético (2 focos gaussianos) | **93,7%**               |
| kriging (Rio 2015-16, real)    | **50,7% ± 2,4**         |


50,7% entre duas classes é cara-ou-coroa.

**Consequência:** no sintético o exame era *redundante* — o agente não o comprava
porque não valia a pena. Passamos ~20 treinos tratando isso como falha do
agente. Era, em boa medida, resposta correta a um ambiente que dava a resposta
de graça.

Na distribuição real, o agente **passa a comprar exame** — nos três seeds.

---



## 2. Quatro defeitos do ambiente, corrigidos


| defeito                                                       | evidência                             | efeito da correção                             |
| ------------------------------------------------------------- | ------------------------------------- | ---------------------------------------------- |
| Recompensa atribuída por **dia**, não por caso                | 97,7% dos passos com recompensa zero  | passos com sinal: 2,3% → 100%                  |
| `epi_confirm` lia a **verdade** do gerador e nunca funcionava | **5598 chamadas, todas devolvendo 0** | passa a ler o mapa de laudos do próprio agente |
| Penalidade punia a **inação**                                 | 62–85% dos casos "em aberto"          | `clinical` de −4036 para −267                  |
| "Nada" custava 0,1                                            | ~1120 passos/episódio                 | virou ação nula (custo 0)                      |


O `epi_confirm` é o mais ilustrativo: comparava a densidade de **uma célula** de
um grid 400×400 contra um limiar que nenhuma célula atingia. A ação existia,
custava, e não informava nada — e o DQN gastava 66% das ações comprando isso.

**205 → 214 testes automatizados** ao longo do trabalho.

---



## 3. O algoritmo era o gargalo

Mesmo ambiente, mesma rede, mesma máscara de ação:


|         | Recompensa    | acurácia | comportamento                                 |
| ------- | ------------- | -------- | --------------------------------------------- |
| **DQN** | **−4164 ± 9** | 0,55     | converge para inação (66,7%), zero exames     |
| **PPO** | **+632 ± 43** | 0,73     | curva monotônica, primeiro a bater "não agir" |


~4.800 pontos de diferença. Antes disso, gastamos ~14h ajustando o DQN
(`lr`, buffer, `n_step`, Double DQN, Huber) — **nenhum ajuste moveu algo
mensurável**. A perda Huber estabilizou o treino (desvio entre seeds de ±1340
para ±9) e o resultado **piorou**: converge de forma confiável para um ótimo
local ruim.

Reproduzido com seeds novos e 3× mais treino: **+645,6 ± 76,0**.

---



## 4. Resultados da fase PPO (antes da correção do modelo epidêmico)

Os números abaixo são do ambiente v7, com o SIR defeituoso descrito no §6. Os
resultados atuais estão no §7.

**Distribuição sintética**


| agente                | Recompensa | exames |
| --------------------- | ---------- | ------ |
| `testtwice`           | +955       | 746    |
| `ppo`                 | **+646**   | 0      |
| `testonce`            | +247       | 373    |
| `clinical` (não agir) | −267       | 0      |


**Distribuição real (kriging)** — os baselines se reordenam


| agente      | Recompensa | exames |
| ----------- | ---------- | ------ |
| `testonce`  | **+2093**  | 373    |
| `ppo`       | **+723**   | 7–37   |
| `testtwice` | +640       | 746    |
| `clinical`  | −15        | 0      |


**Nenhuma conclusão de economia transfere entre as distribuições.** No sintético
testar duas vezes é ótimo em toda a faixa de custo; no kriging, testar **uma**
vez é ótimo em toda a faixa — inclusive quando o exame é quase gratuito.

---



## 5. O subinvestimento em exames — e a causa

**O agente subinvestia em exames.** No kriging fazia 7–37 contra os 373 do
`testonce` (+2093). A pergunta deixou de ser "por que ele não investiga" e
passou a ser **"por que investiga tão pouco"**.

**Diagnóstico do mecanismo:** um exame devolve *sempre* apenas `−custo`; o
benefício aparece ~30 passos depois, num passo que pertence a **outro
paciente**. Três ataques falharam pela mesma razão:

- *Reward shaping* por potencial → matematicamente impossível aqui (medido: F
negativo para qualquer parametrização, porque `s'` já é outro caso).
- Crédito retroativo → exigiria reescrever recompensa já entregue.
- Baratear o exame pela metade → **zero mudança** no comportamento.

**O mapa é decorativo.** Zerar o tensor espacial inteiro muda **0–2%** das
decisões. Testamos treinar nas duas distribuições misturadas, para forçar o
mapa a ter função (é a única forma de saber em que regime se está). Resultado:
piorou nos dois regimes (+300 e +186, contra +646 e +723 dos especialistas) e o
agente **não alternou** de política. O canal espacial ganhou função e o agente
não a usou.

---



## 6. O modelo epidêmico estava errado

Ao desenhar a reformulação "um episódio = um caso", fomos medir a dinâmica
temporal — e encontramos um defeito no gerador de epidemias, herdado desde o
início do projeto.

O SIR não dividia a transmissão pela população e usava um período infeccioso de
250 dias. **Um R0 declarado de 1,5 valia 225 na prática.** A epidemia inteira
cabia em **9 dias**, com pico no dia 3, apesar de `epilength: 60`.

Toda a estrutura temporal do ambiente estava calibrada a esse pico artificial:
sobreposição de casos, tempo até o laudo, fase da epidemia.

**Substituído por um SEIR com parâmetros de literatura** (`dengue_envs/core/epi_model.py`):

| doença | tempo de geração | R0 | fonte |
|---|---|---|---|
| dengue | 16 dias | 1,25–1,70 | Aldstadt 2012; Villela 2017 (Rio) |
| chikungunya | 14 dias | 1,46–1,67 | Moreira 2023 |

O período latente absorve a incubação no mosquito, para que o tempo de geração
do modelo corresponda ao intervalo serial observado. Travado por teste: o R0
medido pela taxa de crescimento bate o declarado dentro de 3%.

| | antes (bug) | **agora (SEIR)** |
|---|---:|---:|
| duração da epidemia | 9 dias | **184 dias** |
| pico | 60 casos/dia | **4 casos/dia** |
| passos entre duas decisões do mesmo caso | 278 | **30** |

O modelo antigo continua no código, sob `epi_model: legacy`, que segue sendo o
padrão — mudar isso alteraria em silêncio todos os resultados já produzidos.

**Os achados econômicos sobreviveram à correção.** `testonce` continua ótimo no
kriging (+2782) e `testtwice` no sintético (+1249); nenhuma ordem se inverteu.

---

## 7. O que destravou o agente: crédito por caso

Em vez de reformular o ambiente (que destruiria a dinâmica temporal, necessária
para a publicação), mudamos **como o algoritmo atribui crédito**.

A observação que motivou: a recompensa global do episódio não tem correlação
com a decisão individual (**−0,019**); a mesma decisão, avaliada ao longo da
linha do tempo **do próprio caso** e descontada pelos **dias** decorridos,
correlaciona **0,994**.

O ambiente decompõe a recompensa por caso e o PPO estima a vantagem ao longo da
trajetória daquele paciente, com γ elevado aos dias entre as decisões — análogo
à atribuição de crédito em sistemas multiagente. **A recompensa do ambiente, que
é a métrica de comparação, não muda** (coberto por teste de invariância), e o
agente continua decidindo dia a dia dentro da epidemia.

**Resultado — kriging com SEIR, 3 seeds, checkpoint final, benchmark de 10 seeds:**

| | recompensa | acurácia | exames/episódio |
|---|---:|---:|---:|
| `testonce` (melhor política fixa) | +2782 | 96,0% | 369 |
| **PPO com crédito por caso** | **+2649 ± 50** | **93,1%** | **266** |
| PPO com GAE padrão | +729 ± 119 | 74,3% | 30 |
| `clinical` (não agir) | −64 | 71,0% | 0 |

- **95% do melhor baseline**, com **28% menos exames** que ele — o agente
  escolhe *quais* casos investigar, em vez de testar todos.
- De 30 para 266 exames: é o primeiro agente do projeto que investiga.
- **Corrigir a epidemia não bastou** — o braço com GAE padrão treinou no mesmo
  ambiente corrigido e continuou quase sem testar.
- Desvio de 50 pontos entre seeds, contra ~1900 de diferença entre os braços.

**Ressalva de desenho:** o braço vencedor mudou duas coisas ao mesmo tempo — o
crédito por caso e 4 features temporais na observação (fase da epidemia,
tendência de 7 dias, idade do caso). A ablação que separa as duas está rodando.

---



## 8. Nota de método

Duas lições que custaram tempo real e mudaram como trabalhamos:

**Execução única não é evidência.** A mesma configuração com seed diferente deu
**−81,90 e −3272,25**. Três conclusões nossas foram retiradas por causa disso.
Todo resultado passou a exigir ≥3 seeds, com média ± desvio.

**Selecionar o melhor checkpoint seleciona ruído.** Medido em 12 rodadas: o
"melhor" fica **1.000 a 1.460 pontos** acima do checkpoint final. Passamos a
reportar o final.

E o que mais rendeu: **medir antes de treinar.** A inviabilidade do *reward
shaping* saiu em 10 minutos de medição, em vez de 43 de treino. O mesmo
princípio cortou o tempo de treino de 2h16 para 43 min, ao revelar que a
avaliação consumia 4,7× mais passos que o treino. E foi assim que o defeito do
SIR apareceu: medindo a dinâmica temporal antes de reformular o ambiente por
causa dela.

**Erros do modelo podem parecer plausíveis.** O SIR gerava curvas com formato
razoável — só a epidemiologia estava errada. Os testes agora travam
propriedades verificáveis contra a teoria (R0 efetivo, equação do tamanho
final), não o formato das curvas.

---

## 9. Próximos passos

1. **Ablação em andamento:** crédito por caso *sem* as features temporais, 3
   seeds, para separar as duas mudanças.
2. **Sazonalidade.** O modelo não tem variação sazonal; com R0 = 1,25 a
   epidemia leva ~9 meses. Vale decidir se entra.
3. **Chikungunya não é mais forçadamente menor que a dengue** — as faixas de R0
   agora vêm da literatura e se sobrepõem. É uma decisão de modelagem a
   confirmar.
4. **Busca automática de recompensa e hiperparâmetros** (a proposta 1 do
   orientador), agora que há um agente estável sobre o qual buscar.

**259 testes automatizados**, dos quais 45 escritos para esta fase.