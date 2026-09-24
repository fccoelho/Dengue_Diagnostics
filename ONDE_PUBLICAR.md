# Onde publicar

Rascunho para discutir com o orientador (24/09/2026). A escolha da revista
depende mais de **qual história o artigo conta** do que do tema. Escopo,
formatos de artigo e taxas de publicação (APC) mudam: **confira no site de cada
revista antes de decidir.** Nada disso foi verificado para esta nota.

---

## 1. Estado do resultado que sustenta a escolha

| resultado | força | onde está |
|---|---|---|
| Crédito por caso × GAE padrão: o PPO usual colapsa e o crédito por caso o destrava | **Forte.** C × A = +2009 (v8, 4 seeds) e +4359 (v9, 1 seed), P = 1,00 | `RESUMO_SESSAO.md` §13, §16 |
| O agente contra a melhor regra fixa | **Modesto.** No v9, o C empata com o `sequencial_clinico` (+2387 contra +2330, diferença +57 [−128, +221]), usando 54 exames a menos (−7%) | commit `8f2064d` |
| Robustez sem retreino | Boa. Estável entre anos e em toda a faixa de dificuldade espacial (medido no v8) | `RESUMO_SESSAO.md` §14 |
| Validação fora do Rio | **Não existe ainda** | pendente (synthetic-pop) |
| Seeds no ambiente corrigido (v9) | **1 por braço** | pendente |

---

## 2. Três formas de contar a história

1. **Método de RL.** "A atribuição de crédito por caso faz o PPO funcionar num
   problema em que as decisões sobre pacientes se intercalam no tempo; o GAE
   padrão colapsa." É a contribuição mais sólida hoje.
2. **Epidemiologia e decisão em saúde.** "RL descobre políticas de testagem
   para a vigilância de arboviroses." Hoje o agente *iguala* a melhor regra
   fixa com menos exames. É honesto, mas ainda fraco como achado
   epidemiológico.
3. **Ambiente e benchmark.** Um simulador aberto e calibrado (SEIR com
   parâmetros de literatura, kriging sobre notificações reais, casos
   "outro"), com baselines fortes e um protocolo estatístico, como
   contribuição em si.

A leitura escolhida define o público, e o público define a revista.

---

## 3. Opções

| revista / evento | o que valoriza | leitura | encaixe hoje | o que faltaria |
|---|---|---|---|---|
| **PLOS Computational Biology** | Método computacional com insight biológico ou epidemiológico, validação forte, código aberto. Tem o formato de artigo "Methods". | 1 + 2 | Médio: "iguala a regra fixa" é pouco para essa revista | Várias seeds; cenários além do Rio (synthetic-pop); análise de *quais* casos o agente testa e por quê; sensibilidade a custo e qualidade do médico no v9 |
| **PLOS Neglected Tropical Diseases** | Relevância para o controle de dengue e chik; o contexto brasileiro é um ponto forte | 2 | Médio: precisa falar a língua da vigilância, não do RL | Tradução para decisão real: exames economizados, em que cenários, e ligação com o fluxo da secretaria de saúde |
| **Epidemics** | Modelagem de doenças infecciosas | 2 + 3 | Bom | Mais seeds; alguma validação espacial fora do Rio |
| **Infectious Disease Modelling** | Modelagem de doenças infecciosas, acesso aberto | 2 + 3 | Bom | O mesmo que Epidemics |
| **Journal of the Royal Society Interface** | Modelagem interdisciplinar | 1 + 2 | Médio a bom | Seeds; uma mensagem geral além do caso do Rio |
| **Medical Decision Making** | Políticas de decisão em saúde comparadas com regras | 2 | Bom: a comparação com regras fixas é o que avaliam | Análise econômica clara (custo por diagnóstico correto); sensibilidade ao custo do exame |
| **Health Care Management Science** | Alocação de recursos escassos em saúde | 2 | Bom | O mesmo que Medical Decision Making |
| **AAAI** (trilha AI for Social Impact) | Método de IA com impacto social; aceita simulação | 1 | **Melhor encaixe hoje** | Mais seeds; ablações de crédito (já existem) |
| **CHIL**, **ML4H**, **MLHC** | ML para saúde, com rigor metodológico | 1 | **Bom** | Mais seeds; talvez um segundo domínio com casos intercalados, para mostrar que o método é geral |
| **Artificial Intelligence in Medicine** | IA aplicada à saúde | 1 + 2 | Bom | Seeds; análise interpretável da política |
| **Journal of Biomedical Informatics** | Informática em saúde | 1 + 2 | Bom | O mesmo que AI in Medicine |
| **BRACIS** (Brasil) | IA, comunidade nacional | 1 | **Bom já** | Pouca coisa |
| **SBCAS** (Brasil) | Computação aplicada à saúde | 1 + 2 | **Bom já** | Pouca coisa |


---

## 4. Recomendação

**Curto prazo: um artigo de conferência com a leitura 1** (BRACIS, SBCAS ou um
workshop como o ML4H). O resultado central já está pronto. Marca a autoria do
método e dá retorno de revisores enquanto a versão longa amadurece.

**Versão longa: PLOS Computational Biology, se o trabalho ganhar três coisas:**

1. **Seeds suficientes** no v9 (≥ 4 por braço), para o bootstrap ter
   resolução no nível das seeds.
2. **Cenários em outras cidades via synthetic-pop.** Era sugestão do
   orientador e é a validação espacial que essa revista cobra.
3. **Uma análise interpretável de quais casos o agente escolhe testar.** É o
   que transforma "iguala a regra fixa" em insight epidemiológico. Por exemplo:
   "com médico ruim ele testa mais cedo; perto de focos confirmados ele
   economiza".

**Sem esses três itens, os alvos mais realistas para a versão longa são
Epidemics ou Infectious Disease Modelling.** PLOS NTDs é a alternativa se o
orientador preferir o público de vigilância ao público de método.

---

## 5. Checklist do que qualquer revista vai cobrar

- [x] Baselines fortes, incluindo a regra sequencial (`sequencial`,
  `sequencial_clinico`)
- [x] Intervalos de confiança com bootstrap hierárquico e pareado
- [x] Ablação que isola o efeito do crédito (braços A, B e C)
- [x] Código e configs versionados (`experiments/configs/env/` agora está no git)
- [ ] ≥ 4 seeds por braço no v9
- [ ] Bootstrap, anos e temperatura refeitos sobre o v9
- [ ] Análise de quais casos o agente testa duas vezes
- [ ] Sensibilidade ao custo do exame e à qualidade do médico no v9
- [ ] Validação fora do Rio (synthetic-pop)
- [ ] Revisão das referências marcadas como `% TODO conferir` no `Artigo/refs.bib`
- [ ] Atualizar artigo e apresentação com o empate contra o `sequencial_clinico`
  (eles ainda dizem "supera a melhor política fixa por +1163")

---

## 6. Para decidir com o orientador

1. **Qual das três leituras (§2) ele quer.** Isso define a revista.
2. **Onde o grupo já publica.** O histórico de publicações do projeto
   Mosqlimate pode apontar a revista mais natural.
3. **Conferência primeiro ou direto para revista.** Vale conferir se a revista
   escolhida aceita trabalhos já publicados em conferência e em que condições.
4. **Prazo.** Ele define se dá tempo de fazer o synthetic-pop antes da
   submissão.
