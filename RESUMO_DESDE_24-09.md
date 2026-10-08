# O que foi feito desde 24/09

*Atualizado em 08/10/2026. Branch `krigin-env`. Artigo: `Artigo/main.pdf` (28 páginas). Detalhes dos números em `AVANCOS_ARTIGO.md`.*

## 0. Em uma página

Depois da reunião de 24/09 o trabalho seguiu quatro linhas:

1. **Treinar em cada cenário e na mistura, ver a mobilidade entre cenários e usar bootstrap** (v6, 26/09). O crédito por caso vence o GAE padrão em todas as combinações de treino × teste.
2. **Rastrear uma falha de transferência do Recife até um defeito do simulador** (26–27/09). Corrigido o defeito, Rio e Recife transferem nos dois sentidos.
3. **Resolver as três críticas que um revisor faria** (v7, 28/09–01/10): tudo é simulado, só havia 3 sementes e a matriz tinha sido feita antes de uma correção. Ficaram 5 sementes, o ambiente corrigido em tudo, varreduras de sensibilidade e uma confirmação em 30 surtos nunca olhados.
4. **Fechar pendências do artigo** (07–08/10): casos do surto proporcionais à população e conferência das referências.

**Resultado em uma frase:** nas cidades reais o agente treinado empata com a melhor regra de especialista usando cerca de 7% menos exames, e o crédito por caso é o que torna isso possível.

## 1. Cronologia

| Data | Commit | O que entrou |
|---|---|---|
| 26/09 | `66e6b71` | Experimento v6: treino por cenário, bootstrap espacial, avaliação cruzada |
| 26/09 | `3724b30` | Diagnóstico do Recife: atalho espacial e posicionamento aleatório |
| 26/09 | `59694b8` | Artigo: estudo de transferência e resultados do v6 |
| 27/09 | `7477220` | Ambiente corrigido: casos "outro" só na área habitada, escala física |
| 27/09 | `5856e7f` | Análise por caso e a regra derivada do agente |
| 27/09 | `fc7a91c` | Artigo: análise por caso, transferência corrigida, trabalhos relacionados |
| 01/10 | `4e7fe9b` | Simulador ~2× mais rápido, resultados idênticos bit a bit |
| 01/10 | `3fc0049` | Experimento v7: 5 sementes, ambiente corrigido, varreduras e confirmação |
| 01/10 | `47a6904` | Artigo com os resultados finais (v7) e resumo para o professor |
| 07–08/10 | *(sem commit)* | Casos proporcionais à população; conferência das referências |

## 2. v6 — treino por cenário e mobilidade (26/09)

**O que foi construído**
- **Bootstrap espacial:** 20 réplicas das superfícies por cidade e ano. As notificações são reamostradas e, no Recife, também a geocodificação. Serve para medir a incerteza das superfícies e dá variedade ao treino.
- **Cenários de treino:** sintético (posição acerta 94%), Rio (réplicas de 2015–16), Recife (réplicas de 2017–20 e 2022–25) e uma mistura de um terço de cada. Os testes usam sempre as superfícies originais; 2016 e 2021 do Recife ficam fora do treino.
- **Fila de treino retomável:** 24 treinos (4 cenários × braços A e C × 3 sementes), em paralelo, com guarda de RAM. O `agents/ppo/train.py` passou a retomar do último checkpoint de época.
- **Avaliação cruzada:** 30 surtos pareados inéditos por teste (sementes 3001–3030), bootstrap hierárquico e pareado.

**Resultado:** o C vence o A nas 16 combinações (P entre 0,94 e 1,00). O C treinado no Rio empata com a regra no Rio e em anos inéditos do Recife, com menos exames. O C treinado no sintético supera a regra onde a posição informa, mas desaba nas cidades reais. O Recife → Rio falhava, e a mistura ficava abaixo da regra.

## 3. A falha do Recife era um defeito do simulador (26–27/09)

- **Sintoma:** o agente treinado no Recife falhava até nos próprios anos guardados e desabava no Rio (−1643, desvio entre sementes de 722).
- **Ablação sem retreino:** o agente do Rio ignora o espaço. O do Recife colapsa sem a posição do caso (+2135 → −4631) e transfere melhor para o Rio sem o mapa (+464 → +1514). Leitura: ele memorizava o contorno do município.
- **Primeiro remédio (sortear escala e posição da cidade a cada episódio): piorou** (Rio −3491), o que é coerente com o diagnóstico, porque aumentava a área vazia.
- **Causa:** os casos "outro" caíam uniformes no grid inteiro, mas o município ocupa só 44% dele. Um caso fora da cidade era "outro" com certeza.
- **Correção:** máscara de área habitada pela mesma regra nas duas cidades (recupera o polígono oficial do Recife e dá 1160 km² para o Rio), e casos "outro" só dentro dela. Com isso, Rio e Recife transferem nos dois sentidos, com todas as células empatadas com a regra.
- **Escala física comum (200 m por célula):** testada, mas sozinha quebrou o Rio → Recife, porque o número de casos por surto era fixo e o Recife, cinco vezes menor, ficava cinco vezes mais denso. Isso foi resolvido em 07–08/10 (seção 7).

## 4. O que o agente aprendeu, caso a caso (27/09)

- Para suspeitas de dengue ou chik, o agente faz o mesmo que a regra: primeiro exame da doença suspeita (99%) e segundo exame depois de todo negativo (~100%).
- **A economia vem de não testar 94% dos casos que o médico rotula como "não é arbovirose".**
- Escrito como regra fixa, isso virou a `sequencial_confia_outro`: +56 sobre a regra clínica, com 38 exames a menos (empate estatístico no custo padrão). Foi escrita depois de olhar o agente, e o artigo diz isso.
- **O segundo exame é sempre ótimo neste ambiente.** A posterior após um primeiro negativo é plana (probabilidade máxima mediana de 0,52). Parar só compensaria em 0,4% dos negativos no Rio.
- O agente do cenário idealizado faz outra coisa: testa só 46% dos casos e deixa sem exame os casos longe da fronteira entre os focos.

## 5. Simulador ~2× mais rápido (01/10)

O treino passava ~90% do tempo em operações de pandas refeitas a cada dia simulado. Agora a tabela de casos é montada uma vez por surto, os mapas usam `np.bincount` e a observação é vetorizada. Conferido: a regra e o PPO em 4 cenários dão métricas idênticas às de antes, e as 915 observações de mapa comparadas passo a passo são iguais.

## 6. v7 — protocolo final (28/09–01/10)

**Protocolo**
- 5 sementes de treino por braço e geografia (eram 3).
- Ambiente corrigido em tudo. Rio e sintético não mudam com a correção, então as sementes antigas foram reaproveitadas.
- 30 surtos pareados por teste, bootstrap hierárquico e pareado com 10 mil réplicas.
- Matriz treino × teste: Rio, Recife, idealizado e mistura (com o dobro de passos), braços A e C.
- Varreduras no Rio, sem retreino: custo do exame (2, 3, 6, 8), sensibilidade do RT-PCR (0,80, 0,90, 0,99) e acurácia do médico (0,55 a 0,95), além de agentes retreinados com custo 2 e 8. Parâmetros do RT-PCR ancorados em Santiago et al. 2013 e Waggoner et al. 2016.
- Volume: 46 treinos e 307 avaliações, cerca de 9.200 episódios de 300 dias.

**Resultados principais**
- **Rio, ambiente completo:** o C faz +2168 [+2042, +2290] com 651 exames. A regra derivada faz +2205 e a sequencial guiada pelo médico +2149. O C × regra é um empate (+19 [−89, +121]) com 49 exames a menos (−7%). O C × A é +3911, P = 1,00.
- **O C vence o A em todas as 16 células** da matriz (+1215 a +4275).
- **Rio ↔ Recife transfere** nos dois sentidos. O agente do Recife corrigido é o mais estável (desvio entre sementes de 21 no Rio, contra 722 antes).
- **Idealizado:** aprender vence a regra onde a posição informa (+627), mas não transfere para cidades reais.
- **Mistura: resultado negativo.** Fica abaixo da regra nas três cidades reais (−197 a −274) e é instável. Com 3 sementes parecia empatar; com 5, não se sustenta.
- **Sensibilidade:** o agente empata com a regra em toda a faixa de sensibilidade do exame e de acurácia do médico. Com custo 6 e 8, ele passa a vencer (+116 e +213). **Retreinado com custo 8, colapsa** (106 exames, pior que não testar), e com esse custo não fazer nada vence todas as políticas que testam.

**Confirmação em 30 surtos nunca olhados (4001–4030).** A regra derivada e a correção do simulador foram desenhadas olhando os surtos 3001–3030, então repetimos a matriz e os custos 6 e 8 em surtos novos.
- Confirmado: C > A em tudo, empate no Rio, idealizado vence no idealizado, mistura abaixo da regra, colapso com custo 8.
- **Duas ressalvas:** o agente do Rio perde cerca de 7–8% no Recife (−177 e −150, intervalos excluindo o zero por pouco), e a vantagem da regra derivada no custo 6 não se reproduziu.

## 7. Casos proporcionais à população (07–08/10)

Na escala física comum, o número de casos por surto era o mesmo nas duas cidades, e o Recife (cinco vezes menor) ficava cinco vezes mais denso.

- **Implementação:** `episize` vale para o Rio; cada cidade recebe `episize × população / população do Rio` (Censo 2022: Rio 6.211.423, Recife 1.488.920). O Recife fica com 72 casos de referência em vez de 300. Chaves de config `surface_population` e `population_ref`; função `scaled_popsize` em `dengue_envs/wrappers/factory.py`; 5 testes novos.
- **Densidade medida** (casos por 1000 células habitadas): Rio 17,2; Recife 89,5 antes e 18,5 depois.
- **Treino e avaliação:** 7 treinos novos e 95 avaliações em 30 surtos pareados.

| Agente do Rio − regra | Recife 2016 | Recife 2021 |
|---|---|---|
| Densidade original | −525 [−958, −128] | −576 [−1031, −176] |
| Proporcional à população | −38 [−84, +2] | −26 [−70, +11] |

- **A densidade, e não a geografia, explicava a falha.** A perda cai de 24–26% para 6–9% da recompensa da regra.
- **Resultado negativo:** o agente treinado nos surtos pequenos do Recife (5 sementes) não melhora em casa (−41 e −49), pede menos exames (71–91 contra 132) e erra mais. Também não transfere para surtos maiores (−2330 no Rio). O agente treinado no Recife denso (3 sementes) é mais robusto. A causa de o agente pequeno pedir menos exames não foi testada.
- **Ressalva:** feito nos surtos 3001–3030 (já olhados); não repetido em 4001–4030.

## 8. Referências (08/10)

- As 7 entradas marcadas com `% TODO` no `refs.bib` foram verificadas contra fontes primárias (Crossref, PubMed/Europe PMC, PMC) e corrigidas. Moreira 2023 ganhou título e os 26 autores; o R0 de 1,56 [1,46–1,67] com tempo de geração de 14 dias, que o artigo cita, confere com o texto do artigo original. Santiago 2013 tem os 10 autores e Aldstadt 2012 os 11. Agarwal ganhou o volume 34 e Saravanan, volume, número e páginas.
- As 17 referências de trabalhos relacionados que tinham entrado de memória conferem com as fontes.
- **Não confirmado:** as páginas do Agarwal et al. (NeurIPS 34) ficaram de fora.

## 9. Mudanças no artigo

- Resumo e contribuições com os números do v7 (5 sementes) e a regra derivada como "frugal".
- Seções novas: trabalhos relacionados, "What the agent learned", sensibilidade a custo/sensibilidade do exame/acurácia do médico, confirmação em surtos novos e a tabela de escala por população.
- Transferência reescrita: matriz no ambiente corrigido, o defeito do simulador como diagnóstico, a escala física e a mistura como resultado negativo.
- Limitações novas: reuso dos surtos de teste, colapso com custo 8, estrutura (não só nível) dos parâmetros simulados, escala por população.
- Figuras novas ou refeitas: `fig_transfer`, `fig_transfer_corrected`, `fig_policies_full`, `fig_cases`, `fig_sensitivity`.

## 10. O que ainda está aberto

**Marcado para não fazer neste momento:**
- Anos novos do Rio (Fiocruz) como teste fora do treino.
- Calibração com os microdados do Recife (acerto real do médico, sensibilidade do RT-PCR por dia de coleta, sintomas).
- Outras cidades via synthetic-pop.
- Sazonalidade e capacidade diária de laboratório.

**Do pedido original do professor (24/09), ainda sem fazer:**
- **Clamping por quantis e mistura com a uniforme** existem só como código e teste. Nenhum resultado deles foi rodado.
- A varredura de temperatura nunca foi refeita no ambiente corrigido (v7), e ninguém treinou A e C em pontos intermediários da curva.
- Confirmar com o professor se "desvios de uniforme" era essa leitura espacial.

**Outras pendências:**
- Repetir os resultados de escala por população nos surtos de confirmação (4001–4030), cerca de 1 h de avaliação.
- O `dengue_envs/data/zikario.gpkg` (notificações individuais do Rio, com data de nascimento e coordenada da residência) está versionado no GitHub. É preciso decidir com o Flávio se sai do repositório e do histórico.

## 11. Como reproduzir

```bash
# testes (314 passando)
.venv/Scripts/python.exe -m pytest -q

# v7: treinos e avaliação (retomáveis)
.venv/Scripts/python.exe -m experiments.fila --v7 --paralelo 4
.venv/Scripts/python.exe -m experiments.avaliacao_v7 --workers 10
.venv/Scripts/python.exe -m experiments.avaliacao_v7 --analisa

# confirmação em surtos novos
.venv/Scripts/python.exe -m experiments.avaliacao_v7 --conjunto confirmacao --workers 10

# casos proporcionais à população
.venv/Scripts/python.exe -m experiments.fila --pop --paralelo 3
.venv/Scripts/python.exe -m experiments.avaliacao_v7 --conjunto pop --workers 10
.venv/Scripts/python.exe -m experiments.avaliacao_v7 --conjunto pop --analisa

# análise caso a caso e figuras do artigo
.venv/Scripts/python.exe -m experiments.analise_casos
.venv/Scripts/python.exe -m experiments.figuras_artigo
```

Resultados brutos: `results/ppo_v6/` (treinos), `results/v6_avaliacao/` (avaliações; `v7/`, `v7_confirmacao/` e `v7_pop/` têm as tabelas de bootstrap).
