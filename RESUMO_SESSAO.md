# Da confirmação epidemiológica ao diagnóstico do agente

Relatório da investigação iniciada a partir das três propostas de revisão do
`epi_confirm`. Tudo aqui é medido; onde há incerteza, ela está declarada.
Histórico completo, incluindo os erros do caminho, em `REFATORACAO_AMBIENTE.md`.

---

## 1. Ponto de partida

Três propostas em discussão:

1. Usar algoritmos de otimização para varrer recompensas e hiperparâmetros.
2. O `epi_confirm` deveria ser algo que o agente **aprende** a partir da
   distribuição observada da doença, não um dado pronto.
3. O `epi_confirm` deveria **custar perto de um exame**, porque simula ida a
   campo, que é cara.

A investigação começou pela (2) e (3), que eram acopladas. A (1) foi adiada por
um motivo que a própria investigação confirmou — ver §8.

---

## 2. O que o `epi_confirm` realmente fazia

### 2.1 Vazava a verdade

`_epi_confirm` consultava `world.get_maps_up_to_t()`: histogramas construídos
do dataframe do **gerador**, indexados pela coluna `disease` **real** — casos
inclusive nunca notificados. O agente perguntava "a densidade real da doença D
aqui é alta?" e recebia a resposta da verdade-terreno, por **0,5**, contra
**4,0** de um exame que ainda por cima é ruidoso.

A intuição da proposta (2) estava certa, e o problema era maior que "deveria
ser aprendido": era privilégio de informação.

O tensor de observação **não** vazava — seus canais vinham de `obs_cases`.

### 2.2 E, apesar disso, nunca informou nada

Medindo antes de corrigir:

```
grid (400, 400), dia final t=65
dengue: total=140 | células ocupadas=140 | max/célula=1 | células com >1 = 0
chik  : total=140 | células ocupadas=140 | max/célula=1 | células com >1 = 0
```

A função retornava `1 if dmap[x,y] > threshold` com `threshold=1`. Como
**nenhuma célula jamais tem mais de 1 caso**, a condição nunca era satisfeita.

Verificação direta: **5598 chamadas em 5 episódios, todas devolvendo 0.**

A causa é geométrica — ~140 casos de cada arbovirose num grid de 160 mil
células. Dois casos na mesma célula é evento raríssimo.

**Consequências:**

- O bit `epiconf` em `case_features` era uma **entrada constante zero**.
- O DQN da rodada anterior gastava **66,1% das suas ações** comprando isso.
  Não era estratégia; era pagar 0,5 por nada, com o efeito colateral de adiar
  a decisão.

---

## 3. A correção

| Antes | Depois |
|---|---|
| mapas do gerador (verdade) | mapa de confirmados **do agente** |
| casos reais, notificados ou não | **só laudos positivos** do próprio agente |
| célula única → sempre 0 | vizinhança de raio 20 |
| incluía o próprio caso | **exclui** o próprio caso |
| custo 0,5 | **custo 4,0** (= `test_cost`) |
| 1 bit para o agente | 1 bit + densidade local + 2 canais de mapa |

A auto-exclusão não é detalhe: sem ela, raio 0 "acerta" 99,6% — que é o exame
já pago voltando com outro nome.

**Calibração medida** (política que testa tudo):

| raio | limiar | confirma | acerto quando confirma |
|---:|---:|---:|---:|
| 10 | 1 | 4,5% | 83,0% |
| **20** | **1** | **19,3%** | **80,1%** |
| 30 | 2 | 21,6% | 79,7% |

Referência: sem evidência epidemiológica, o palpite clínico acerta **57,1%**.
Raio 20 dá **+23 pontos** sobre isso.

**O incentivo pretendido pela proposta (2) existe e está travado por teste.**
O mapa conta apenas casos confirmados por laudo — como faz a vigilância real, e
o que evita o agente realimentar os próprios palpites como evidência.
Resultado: sem nenhum exame, zero confirmações saem; com exames acumulados,
elas passam a sair. **O exame virou investimento** — cada laudo positivo
melhora toda `epi_confirm` futura naquela região.

**Verificação nos baselines:** mudaram **só** as políticas que compram a ação
(`qlearning` −2916 → −3177, `random` −5349 → −5654, ambas por pagar 8× mais).
`testtwice`, `testonce`, `confirmall` e `clinical` ficaram idênticas ao
centavo — a mudança é cirúrgica.

---

## 4. Efeito colateral: o treino ficou 5× mais rápido

Ao preparar o treino, a medição do gargalo trouxe um resultado inesperado.

**Paralelizar o ambiente era o alvo errado.** O env faz **1263 passos/s**
contra **~25** do laço de treino completo; e `subproc` fica **35% mais lento**
por serializar 938 KB de observação por passo.

**O gargalo era o tamanho da observação** — que a rede descarta de qualquer
forma, porque o encoder termina em `AdaptiveAvgPool2d((6,6))`:

| resolução | KB/obs | gather+H2D | replay buffer |
|---|---:|---:|---:|
| 400×400 | 937,5 | 7,6 s/100upd | 5,36 GB |
| **100×100** | **58,6** | **0,7 s** | **0,34 GB** |

Época: **6min20 → 1min14**. Rodada de 20 épocas: **~2h16 → ~43 min**.

Verificado que não custa fidelidade: no **mapa real** (kriging do Rio) só
**1,7%** dos casos colidem a 100×100 — *menos* que no sintético. O gargalo
espacial é o `pooled_size` (12 km/célula com o bbox do Rio), não o tensor.
Esse parâmetro estava fixo e inalcançável pela configuração; foi exposto.

**Custo honesto:** essa mudança quebrou o encoder do `qlearning`, que indexava
o tensor com coordenadas do mundo — de duas formas, uma delas silenciosa
(fatiamento fora do eixo devolve array vazio, e todas as contagens locais
viravam 0 sem erro). Corrigido lendo os atributos do `obs_cases` em vez do
mapa; dois testes de regressão adicionados.

---

## 5. Primeira medição no ambiente novo

Seis treinos (3 seeds × 2 tamanhos de buffer), protocolo multi-seed:

| braço | Recompensa (média ± desvio) |
|---|---:|
| buffer 6.000 | **−3975 ± 37** |
| buffer 50.000 | −3700 ± 200 |

Resultado ruim — vizinho do `clinical`, que na época valia −4036. Mas a
**decomposição** apontou a causa com precisão:

| rodada | concluídos | **em aberto** | custo exames | **penal. aberto** |
|---|---:|---:|---:|---:|
| b6k s42 | 377 | **743** | −1436 | **−7430** |
| b6k s43 | 168 | **952** | 0 | **−9520** |
| b50k s44 | 202 | **918** | −1316 | **−9180** |

**62–85% dos casos terminavam "em aberto"**, e essa penalidade era o termo
dominante — muito acima de qualquer custo de investigação.

---

## 6. A penalidade punia a inação

`penalty_unresolved` (−10) caía sobre **todo** caso não concluído, inclusive
aqueles em que o agente simplesmente não agiu. E era somada **em bloco no fim**
do episódio, sem ser atribuída a decisão nenhuma.

Da ótica do agente: escolher "nada" custava 0,1 imediatos, e os −10 chegavam
despersonalizados no fim. **Ele otimizava corretamente um sinal que mentia.**

**Redefinição adotada:** não agir é *deixar valer o diagnóstico do médico* —
decisão legítima da vigilância, julgada pelo acerto do palpite clínico. Só paga
por ficar em aberto quem **abriu investigação** (exame ou ida a campo) e não a
fechou. A penalidade passou a ser cobrada **no passo** em que o caso é largado.

E "nada" virou a **ação nula** no sentido padrão de RL: custo 0, o passo devolve
exatamente 0,0.

**Efeito nos baselines:**

| agente | antes | depois |
|---|---:|---:|
| `clinical` | −4036,5 | **−267,2** |
| `confirmall` | −1365,2 | −1365,2 |
| `testtwice` | +955,0 | +955,0 |

Abster-se passou a bater `confirmall`, que é o comportamento correto: **não
opinar é melhor que opinar mal**. As políticas que concluem tudo não mudaram
um centavo.

---

## 7. Onde o agente está hoje

**Benchmark, 10 seeds de avaliação:**

| # | Agente | Recompensa | Acurácia | Exames |
|---|---|---:|---:|---:|
| 1 | `testtwice` | **+955,0** | 0,94 | 746,4 |
| 2 | `testonce` | +247,4 | 0,76 | 373,2 |
| 3 | **`clinical` (não fazer nada)** | **−267,2** | 0,57 | 0,0 |
| 4 | `confirmall` | −1365,2 | 0,57 | 0,0 |
| 5 | `qlearning` * | −3908,2 | 0,71 | 342,1 |
| 6 | `random` | −6066,0 | 0,44 | 194,5 |

\* q-table de regime anterior; é piso, não teto.

**Melhor DQN da sessão: −358** (braço `lr`, seed 43) — quase encostando no
`clinical`, mas num braço cuja variância entre seeds é ±1340.

O `testtwice` define o teto prático: **investigar compensa, e muito**. O
problema não é a economia do ambiente.

---

## 8. O experimento final: instabilidade não era a causa

Diagnóstico prévio: a recompensa de teste oscilava até **6.300 pontos entre
avaliações vizinhas** depois de 100 mil passos. Enquanto isso durasse, nenhuma
mudança de recompensa seria detectável — que é exatamente por que a proposta
(1), de busca automática, foi adiada: **buscar sobre um algoritmo que oscila
6.300 pontos é otimizar sorteio de seed.**

Dois braços, tudo o mais idêntico:

| braço | oscilação | vs referência (1608) | Recompensa |
|---|---:|---|---:|
| `lr` 2.5e-5, MSE | 1732 | inalterada (+8%) | **−1592 ± 1340** |
| `lr` 2.5e-5 + **Huber** | **578** | **−64%** | **−4164 ± 9** |

**O Huber estabilizou de forma inequívoca.** O desvio entre seeds desabou de
**±1340 para ±9** — duas execuções independentes chegaram praticamente ao mesmo
ponto. (O Tianshou usa MSE por padrão, em que o gradiente cresce linearmente
com o erro; o Huber o satura. Double DQN, ao contrário do que se supunha, já
estava ligado o tempo todo.)

**E o ponto é pior.** −4164 contra −1592.

A política que ele encontra, reprodutivelmente:

| rodada | nada | exames | conc D | conc C | conc OUT |
|---|---:|---:|---:|---:|---:|
| lrhuber s42 | 66,7% | **0,0%** | 14,9% | 0,0% | 18,4% |
| lrhuber s43 | 66,7% | **0,0%** | 0,0% | 28,4% | 4,9% |

Ambas travam em 2/3 de inação, **zero exames**, e cada uma ignora uma das
classes conclusivas.

> **A conclusão:** a instabilidade era real e **é tratável**, mas **não era a
> causa** do mau desempenho. Estabilizado, o DQN converge de forma confiável
> para um ótimo local degenerado que nunca compra informação. O problema não é
> o otimizador não convergir — é *para onde* ele converge.

Nota: o melhor resultado individual da sessão (−358) veio do braço **instável**.
Instabilidade ocasionalmente tropeça em algo bom; estabilidade sem exploração
não tropeça em nada.

*Ressalva: o braço Huber tem 2 seeds, não 3 — houve um desligamento da máquina
durante o terceiro. Com ±9 entre os dois, a conclusão sobre estabilização é
sólida; a magnitude do −4164 mereceria mais um seed (43 min).*

---

## 9. Duas lições de método

**Execução única não é evidência.** A mesma configuração com seed diferente deu
**−81,90 e −3272,25** — 3190 pontos de distância. Três conclusões que eu havia
tirado sobre `n_step` foram retiradas por isso. Todo resultado passou a exigir
≥3 seeds, com média ± desvio.

**`policy_best` seleciona ruído.** É o máximo sobre avaliações ruidosas. Medido
em 12 rodadas: fica **1.000 a 1.460 pontos acima** do `policy_final`. Números
reportados por `best` são otimistas por construção; o relatório passou a usar
`final`.

---

## 10. O que os dados apontam

A questão deixou de ser hiperparâmetro. A §8 aponta duas frentes:

1. **Exploração dirigida.** Com `eps_train_final: 0.05` e uma política colapsada
   em "nada", a chance de descobrir por acaso a cadeia *testar → esperar laudo →
   concluir* (~30 passos, intercalada com outros casos) é ínfima.

2. **PPO.** Otimiza a política diretamente, com bônus de entropia, em vez de
   depender de ε-greedy sobre uma função Q instável. Não tem o modo de falha
   que dominou esta investigação.

Duas hipóteses de mecanismo ainda não testadas:

- **Atribuição retroativa por caso.** Um exame devolve *sempre* apenas
  `-test_cost`; o benefício aparece ~30 passos depois, num passo que pertence a
  outro caso. Creditar o passo do exame quando a conclusão que ele viabilizou é
  paga ataca isso diretamente.
- **`pooled_size`.** A rede vê células de 12 km. Já configurável, nunca variado.

**Sobre a proposta (1)** — busca automática de recompensa e hiperparâmetros:
ela volta à mesa assim que o item 1 ou 2 acima der um algoritmo estável.
Recomendação: Optuna com TPE e pruner ASHA, objetivo = média sobre ≥3 seeds
*menos* o desvio (para preferir configurações confiáveis, não sortudas), e
*shaping* baseado em potencial (Ng, Harada & Russell, 1999), que provadamente
não altera a política ótima. A §4 já tornou isso viável ao cortar a avaliação de
2h16 para 43 min.

---

## 11. Estado do código

- **195 testes passando** (eram 154 no início desta investigação).
- Ambiente atual: `experiments/configs/env/synthetic_v7.yaml`.
- Todos os checkpoints anteriores à §3 são incompatíveis (a observação mudou).

```bash
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m experiments.evaluate --config experiments/configs/benchmark_v9.yaml
.venv/Scripts/python.exe -m agents.deepq.train --config experiments/configs/train/dqn_v10_lrhuber_s42.yaml
```
