# Avanços para o artigo — o que fizemos desde a última reunião

*Atualizado em 01/10/2026. Artigo: `Artigo/main.pdf` (27 páginas). Notebook para os orientadores: `notebooks/apresentacao_orientadores.html`.*

## Resumo em uma página

O professor pediu para **solidificar a narrativa, treinar em cada cenário e na mistura, ver a mobilidade entre cenários e usar bootstrap para confirmar as ideias**. Tudo isso foi feito. Depois resolvemos as três críticas que um revisor faria: tudo é simulado, só havia 3 sementes e a matriz tinha sido feita antes de uma correção no simulador. Por fim, **confirmamos tudo num conjunto novo de 30 surtos (sementes 4001–4030) que nunca tinha sido olhado** (seção 3.7).

**A história do artigo agora:**

1. **Nos dados reais, a posição quase não revela a doença.** Ela acerta só 52–57% em todos os anos epidêmicos do Rio e do Recife, com IC de ±0,01 pelo bootstrap espacial. O exame é a principal fonte de informação.
2. **O aprendizado por reforço padrão não aprende a usar o exame.** O valor do exame se dilui entre decisões intercaladas sobre outros pacientes (correlação −0,02).
3. **O crédito por caso resolve.** A vantagem é calculada na linha do tempo de cada paciente, descontada em dias (correlação 0,99). Ele vence o padrão nas 16 combinações de treino × teste.
4. **Nas cidades reais, o agente empata com a melhor regra de especialista, com 7% menos exames.** Isso vale com 5 sementes, nas duas cidades, variando custo, sensibilidade do exame e acurácia do médico, e se repete nos surtos novos. Levado do Rio para anos inéditos do Recife, fica a até 8% da regra.
5. **Lendo as decisões caso a caso, o agente é a regra do especialista mais um refinamento:** não testar quando o médico diz "não é arbovirose". Escrito como regra fixa, esse refinamento iguala a regra do especialista com menos exames e é melhor quando o exame é caro.
6. **Onde a posição informa (cenário idealizado), o agente supera todas as regras (+627).** Mas essa política não transfere para cidades reais.
7. **Uma falha de transferência do Recife foi rastreada até um defeito do simulador.** Corrigido o defeito, Rio e Recife transferem nos dois sentidos.

**Resultados negativos que reportamos honestamente:**
- treinar numa mistura de geografias não funciona;
- treinar com exame muito caro (custo 8) colapsa;
- a regra derivada não é significativamente melhor no custo padrão;
- nos surtos novos, o agente do Rio fica um pouco abaixo da regra no Recife (−150 a −177, ~7–8%).

---

## 1. O que mudou no simulador

| Mudança | Por quê | Efeito |
|---|---|---|
| Casos "outro" (25% de suspeitos sem arbovirose) também no gerador de krigagem | antes só existiam no sintético; um exame sempre bastava | o problema ganhou profundidade: segundo exame e "concluir outro" passam a fazer sentido |
| **Recife como segunda cidade** | testar transferência entre cidades | microdados abertos 2013–2025 geocodificados por rua e bairro: 72% na rua, 6% em rua de bairro vizinho, 22% no bairro, <1% sem local |
| Superfícies por ano (Recife 2015–2025) | ter anos de treino e anos de teste | treino 2017–20 e 2022–25; teste 2016 e 2021 |
| **Bootstrap espacial** (20 réplicas por cidade/ano) | medir a incerteza das superfícies e dar variedade ao treino | reamostra notificações e refaz a krigagem; no Recife também a geocodificação |
| **Correção: casos "outro" só na área habitada** | era um defeito: fora do município do Recife só caíam casos "outro" | máscara de área habitada pela mesma regra nas duas cidades; recupera o polígono oficial do Recife (875 × 879 células) |
| Escala física comum (200 m por célula), testada | raio da confirmação era 1 km no Recife e 3 km no Rio | quebrou o Rio → Recife: com casos por surto fixos, o Recife ficou 5× mais denso; fica como trabalho futuro |
| Regra fixa nova `sequencial_confia_outro` | escrita a partir das decisões do agente | ver seção 4 |
| **Simulador 2× mais rápido** | o treino passava 90% do tempo no ambiente (pandas) | cache da tabela de casos, mapas por bincount, observação vetorizada; resultados idênticos bit a bit |

## 2. Protocolo experimental (o que dá credibilidade estatística)

- **5 sementes de treino por braço e geografia** (45–49). Antes eram 3.
- **30 surtos pareados inéditos por teste** (sementes 3001–3030): todos os agentes e regras veem exatamente os mesmos surtos.
- **Bootstrap hierárquico pareado:** reamostra sementes de treino e, dentro delas, surtos, com 10 mil réplicas. Reportamos média, IC de 95% e P(X > Y).
- **Matriz completa treino × teste:**
  - treinos em Rio, Recife, idealizado e mistura (2× passos), braços A (GAE padrão) e C (crédito por caso);
  - testes em Rio 2015–16, Recife 2016 e 2021 (fora do treino) e idealizado.
- **Varreduras de sensibilidade** no Rio, sem retreino:
  - custo do exame 2, 3, 6 e 8;
  - sensibilidade do RT-PCR 0,80, 0,90 e 0,99;
  - acurácia do médico fixa em 0,55, 0,70, 0,85 e 0,95;
  - além de agentes retreinados com custo 2 e 8.
- **Parâmetros do RT-PCR ancorados na literatura:**
  - Santiago et al. 2013 (dengue: 98% de detecção nas amostras agudas);
  - Waggoner et al. 2016 (multiplex; concordância menor para chik com carga viral baixa).

Volume total: 46 treinos no v7 (31 novos + 15 reaproveitados) e 307 avaliações × 30 surtos ≈ 9.200 episódios de 300 dias.

## 3. Resultados principais (v7: ambiente corrigido, 5 sementes)

### 3.1 Ambiente completo, Rio de Janeiro (Tabela 5 e Figura 7 do artigo)

| Política | Recompensa [IC 95%] | Acurácia | Exames |
|---|---|---|---|
| **C: crédito por caso** | **+2168** [+2042, +2290] | 0,935 | 651 |
| Regra derivada (confia no "outro")\* | +2205 [+2089, +2316] | 0,939 | 661 |
| Sequencial guiada pelo médico (melhor regra a priori) | +2149 [+2043, +2254] | 0,941 | 699 |
| Sequencial, dengue primeiro | +1799 | 0,937 | 772 |
| Testa duas vezes | +1144 | 0,937 | 931 |
| Testa uma vez | +407 | 0,763 | 466 |
| Só o médico | −368 | 0,551 | 0 |
| A: GAE padrão | −1743 [−2299, −1193] | 0,580 | 63 |

- **C × regra:** +19 [−89, +121], empate, com 49 exames a menos [−62, −39] (−7%).
- **C × A:** +3911 [+3395, +4438], P = 1,00.

\* Escrita depois de olhar o agente.

### 3.2 Transferência entre geografias (Figura 9: `Artigo/img/fig_transfer.png`)

Agente C menos a regra do especialista:

| Treinado em ↓ / testado em → | Rio | Recife 2016 | Recife 2021 | Idealizado |
|---|---|---|---|---|
| Rio | +19 [−89, +121] | −91 [−250, +44] | −48 [−209, +83] | +11 |
| Recife (corrigido) | −28 [−115, +54] | +23 [−77, +116] | +20 [−81, +117] | +27 |
| Idealizado | −4309 | −2756 | −2975 | **+627** [+499, +752] |
| Mistura (2× treino) | **−197** | **−274** | **−265** | −408 |

- **C vence A em todas as 16 células** (+1215 a +4275; P de 0,95 a 1,00).
- **Rio ↔ Recife transfere nos dois sentidos.** O agente do Recife corrigido é o mais estável (desvio entre sementes de 21 no Rio; era 722 antes da correção).
- **Idealizado:** aprender vence a regra onde a posição informa, mas não transfere.
- **Mistura: resultado negativo.** Fica abaixo da regra nas três cidades reais e é instável. No v6, com 3 sementes, parecia empatar; com 5 sementes, não se sustenta.

### 3.3 O defeito do simulador (Figura 10)

- **Sintoma:** o agente treinado no Recife falhava até nos próprios anos guardados (−168, −220) e desabava no Rio (−1643, desvio entre sementes de 722).
- **Ablação** (zerar entradas sem retreinar):
  - sem a posição do caso, ele colapsa no próprio Recife (+2135 → −4631);
  - sem o mapa, transfere melhor para o Rio (+464 → +1514);
  - o agente do Rio ignora o espaço.
- **Causa:** os casos "outro" caíam no grid inteiro, mas o município ocupa só 44% dele. Um caso fora da cidade era "outro" com certeza, o que dá ~14% dos casos identificáveis só pela posição.
- **Primeiro remédio (sortear escala e posição da cidade): piorou** (Rio −3491), consistente com o diagnóstico, porque aumentava a área vazia.
- **Correção:** casos "outro" só na área habitada. Com ela, a transferência funciona nos dois sentidos (tabela 3.2).
- **Lição para o artigo:** a avaliação agregada não distinguia um atalho de aprendizado ruim; as ablações e o teste entre cidades localizaram o problema.

### 3.4 O que o agente aprendeu, caso a caso (Figura 8: `Artigo/img/fig_cases.png`)

Registramos ~35 mil decisões por agente e teste:
- **Para suspeitas de dengue ou chik, faz o mesmo que a regra:** primeiro exame da doença suspeita (99%) e segundo exame depois de todo negativo (~100%).
- **A economia vem de não testar 94% dos casos que o médico rotula como "não é arbovirose".** São 4,3% dos casos, e 84% deles de fato não são arbovirose.
- **Escrito como regra fixa (a regra derivada):**
  - **custo padrão:** +56 [−26, +138] sobre a regra do especialista, com 38 exames a menos; o empate não é significativo;
  - **custo 6:** +133 [+47, +219];
  - **custo 8:** +209 [+120, +300].
- **O agente do cenário idealizado faz outra coisa:**
  - testa só 46% dos casos;
  - pula metade dos segundos exames;
  - deixa sem exame os casos longe da fronteira entre os focos.

### 3.5 O segundo exame é sempre ótimo neste ambiente

- **Valor de concluir na hora** depois de um primeiro negativo: 10·p − erro·(1−p).
- **Valor de pedir o segundo exame:** +2,2 por caso, já descontado o custo.
- **Posterior estimada por regressão logística** com tudo o que o agente observa: é plana, com probabilidade máxima mediana de 0,52.
- **Parar só compensaria em 0,4% dos negativos no Rio**, com ganho de menos de 1 ponto por surto. O agente está certo em não selecionar.
- **Para a seleção do segundo exame virar decisão real**, o simulador precisa de informação por caso: sintomas, dias desde o início dos sintomas. Isso está nos microdados do Recife e foi para trabalho futuro.

### 3.6 Sensibilidade aos parâmetros simulados (Figura 11: `Artigo/img/fig_sensitivity.png`)

| Parâmetro | Faixa | Agente do Rio − regra |
|---|---|---|
| Sensibilidade do RT-PCR | 0,80–0,99 | empate em todas (~45 exames a menos) |
| Acurácia do médico | 0,55–0,95 | empate em todas |
| Custo do exame | 2, 3, 4 | empate |
| Custo do exame | 6 | **+116** [+16, +212] |
| Custo do exame | 8 | **+213** [+116, +309] |

- **Retreinando com outro custo:**
  - com custo 2, o agente empata com a regra;
  - **com custo 8, colapsa:** 106 exames, acurácia de 0,63, pior que não testar (−1576 contra −368).
- **Com custo 8, não fazer nada vence todas as políticas que testam.** É o regime sem problema de alocação, como o artigo já previa.

### 3.7 Confirmação em surtos novos (sementes 4001–4030)

A regra derivada e a correção do simulador foram desenhadas olhando os surtos 3001–3030. Para tirar esse viés, rodamos de novo a matriz inteira e os custos 6 e 8 em 30 surtos que nunca tinham sido usados: 214 avaliações, cerca de 2 h de máquina.

| Afirmação | Surtos 3001–3030 | Surtos novos 4001–4030 | Veredito |
|---|---|---|---|
| C vence A em todas as combinações | +1215 a +4275 | +1280 a +4142 | ✅ confirmado |
| Agente do Rio empata com a regra no Rio | +19 [−89, +121] | −9 [−109, +89] | ✅ confirmado |
| Agente do Rio empata no Recife 2016 / 2021 | −91 / −48 (empate) | **−177 [−373, −8] / −150 [−332, −5]** | ⚠️ perda pequena, mas detectável (~7–8%) |
| Agente do Recife empata nos 4 testes | −28 a +27 | −104 a −22 (IC encostando no zero) | ✅ confirmado, no limite |
| Idealizado vence no idealizado | +627 | +605 [+501, +702] | ✅ confirmado |
| Mistura abaixo da regra | −197 a −274 | −299 a −420 | ✅ confirmado (negativo) |
| Regra derivada ≈ regra clínica, com menos exames | +56 [−26, +138] | +8 [−90, +109], 40 exames a menos | ✅ confirmado |
| Regra derivada melhor com custo 6 | +133 [+47, +219] | +88 [−14, +193] | ⚠️ não reproduzido |
| Regra derivada / agente melhores com custo 8 | +209 / +213 | +167 [+61, +276] / +186 [+101, +277] | ✅ confirmado |
| Retreino com custo 8 colapsa | sim | sim | ✅ confirmado |

O artigo ganhou uma seção ("Confirmation on fresh outbreaks") que diz exatamente isso. Abstract, transferência, sensibilidade e limitações foram ajustados às duas ressalvas.

### 3.8 Casos proporcionais à população (08/10)

Na escala física comum (200 m por célula) o Recife, 5× menor, ficava 5× mais denso, porque o número de casos por surto era o mesmo. Agora cada cidade recebe `episize × população / população do Rio` (Censo 2022: Rio 6.211.423, Recife 1.488.920), o que dá 72 para o Recife. O Rio não muda.

- **Densidade medida** (casos por 1000 células habitadas): Rio 17,2; Recife 89,5 antes e **18,5** depois.
- **O agente do Rio deixa de falhar no Recife:**

| Agente do Rio (5 sementes) − regra | Recife 2016 | Recife 2021 |
|---|---|---|
| densidade original | −525 [−958, −128] | −576 [−1031, −176] |
| proporcional à população | **−38** [−84, +2] | **−26** [−70, +11] |

  A perda cai de 24–26% para 6–9% da recompensa da regra, com as 5 sementes entre 376 e 440 (regra: 446). A densidade, e não a geografia, explicava a falha.
- **Resultado negativo:** o agente treinado nos surtos pequenos do Recife (5 sementes) não melhora em casa (−41 [−103, +18]; −49 [−105, +6]), pede menos exames (71–91 contra 132) e erra mais (acurácia 0,84–0,88 contra 0,94). Também **não transfere para surtos maiores**: −2330 [−2679, −2006] no Rio. O agente treinado no Recife denso (3 sementes) empata no Recife nas duas densidades e perde 143 no Rio. Hipótese não testada: a recompensa por episódio é 5× menor, enquanto os hiperparâmetros foram ajustados na escala do Rio.
- **Ressalva:** feito nos surtos 3001–3030 (já olhados); não repetido em 4001–4030.
- Reproduzir: `python -m experiments.fila --pop` e `python -m experiments.avaliacao_v7 --conjunto pop --workers 10` (e `--analisa`, que grava em `results/v6_avaliacao/v7_pop/`). Código: `scaled_popsize` em `dengue_envs/wrappers/factory.py`; cenários `cenario_recifefispop` e `teste_recife_{2016,2021}_fispop`.

## 4. Mudanças no artigo (`Artigo/main.tex`)

- **Resumo e contribuições:**
  - números do v7 (5 sementes);
  - a regra derivada como "frugal";
  - robustez aos parâmetros;
  - a mistura e a transferência.
- **Seção nova:** trabalhos relacionados (Eva/COVID na Grécia, diagnóstico com custo, atribuição de crédito multiagente, generalização e atalhos, geoestatística).
- **Seção nova:** "What the agent learned", com a análise caso a caso, a prova de que o segundo exame é ótimo e a regra derivada.
- **Transferência reescrita:**
  - matriz no ambiente corrigido;
  - o defeito como diagnóstico, com a figura da primeira rodada;
  - a escala física;
  - a mistura como resultado negativo.
- **Seção nova:** sensibilidade a custo, sensibilidade do exame e acurácia do médico.
- **Protocolo:**
  - 5 sementes;
  - ambiente corrigido;
  - a regra de área habitada;
  - as varreduras.
- **Referências do RT-PCR** e a varredura de custo no ambiente completo, os dois TODOs do texto que estavam abertos.
- **Limitações novas:**
  - os surtos de teste foram reusados para desenhar a regra derivada e a correção (agora confirmados em surtos novos, com duas ressalvas);
  - o colapso com custo 8;
  - a estrutura (não só o nível) dos parâmetros simulados.
- **Figuras novas ou refeitas:**
  - `fig_transfer`, `fig_transfer_corrected`, `fig_policies_full`;
  - `fig_cases`, `fig_sensitivity`.

## 5. Dados e a reunião com a Fiocruz

- **Pedido:** microdados do Rio de outros anos.
- **Apresentação:** pronta, em 3 slides (problema, solução, dados).
- **Proposta de anonimização:**
  - coordenada → centro de uma célula de 100 m (usamos 500 m);
  - datas → semana epidemiológica e intervalos em dias;
  - sem identificadores;
  - checagem de k-anonimato do lado deles.
- **Os microdados abertos do Recife já têm, pelos nomes das colunas, tudo o que pedimos:**
  - agravo notificado, classificação, critério;
  - RT-PCR, NS1, IgM;
  - sintomas, datas.

  Dá para calibrar o simulador (acerto do médico, sensibilidade por dia de coleta, sintomas) antes mesmo do dado do Rio. Ainda não medimos quanto desses campos está preenchido.
- **Atenção:** `dengue_envs/data/zikario.gpkg` (notificações individuais do Rio com data de nascimento e coordenada da residência) está versionado no GitHub. É preciso decidir com o Flávio: tirar do repositório e do histórico, ou confirmar que o repositório é privado e o uso autorizado.

## 6. Pendências antes de submeter

1. ~~Conjunto de confirmação em surtos novos~~ **feito** (seção 3.7).
2. **Anos novos do Rio (Fiocruz)** como teste fora do treino.
3. **Calibração com o Recife:** acerto real do médico, sensibilidade do RT-PCR por dia de coleta, sintomas.
4. ~~Número de casos proporcional à população~~ **feito** (seção 3.8).
5. ~~Conferir as referências~~ **feito** (08/10): as 7 marcadas foram corrigidas (Moreira 2023 ganhou título e os 26 autores; Santiago, 10 autores; Aldstadt, 11; volumes e páginas de Agarwal e Saravanan) e as 17 de trabalhos relacionados conferem com as fontes. Só as páginas do Agarwal et al. (NeurIPS 34) não foram confirmadas e ficaram de fora.

## 7. Onde está cada coisa

| O quê | Onde |
|---|---|
| Artigo e figuras | `Artigo/main.tex`, `Artigo/main.pdf`, `Artigo/img/`; figuras geradas por `python -m experiments.figuras_artigo` |
| Notebook para os orientadores | `notebooks/apresentacao_orientadores.ipynb` (e `.html`) |
| Cenários de treino e teste | `experiments/cenarios.py` → `experiments/configs/env/` |
| Fila de treinos (retomável) | `python -m experiments.fila --v7` |
| Avaliação (retomável, com vigia) | `python -m experiments.avaliacao_v7 --workers 10`; `--analisa` grava em `results/v6_avaliacao/v7/` |
| Confirmação (surtos 4001–4030) | `python -m experiments.avaliacao_v7 --conjunto confirmacao --workers 10` (e `--analisa`) → `results/v6_avaliacao/v7_confirmacao/` |
| Análise caso a caso | `python -m experiments.analise_casos [--analisa / --limiar]` |
| Máscaras de área habitada | `python -m dengue_envs.data.build_masks` |
| Recife (geocodificação e superfícies) | `dengue_envs/data/recife.py`, `dengue_envs/data/build_recife_surfaces.py` |
