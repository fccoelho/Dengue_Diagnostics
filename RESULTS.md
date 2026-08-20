# Resultados do benchmark — Clínico × Random × DQN

Comparação de três políticas de diagnóstico de dengue (na presença de
chikungunya) no ambiente `DengueDiag-v0`, avaliadas em **duas distribuições
espaciais**: `synthetic` (in-distribution, onde o DQN treinou) e `kriging`
(out-of-distribution, dados reais do Rio 2015–16) — um teste de generalização.

- **10 seeds** por agente; a mesma seed gera o mesmo mundo para os três agentes,
  então as comparações são **pareadas** (mais robustas).
- Métrica primária de ranqueamento: **Recompensa Total** (a mais fiel ao objetivo).
- Reprodução: ver [Como reproduzir](#como-reproduzir).

## Agentes

| Agente | O que faz |
|--------|-----------|
| `clinical` | Aceita o diagnóstico clínico, **sem testar** — piso de referência |
| `random` | Ações aleatórias (testa e decide ao acaso) — controle |
| `dqn` | Política aprendida (Deep Q-Network, Tianshou) — testa seletivamente e decide |

O **DQN** foi treinado apenas no `synthetic_default` (400×400, 150 casos, 60 dias,
`reward_delay_days=5`, `clinical_specificity ∈ [0.5, 0.95]`), rede `DengueNet`
(~1,25 M parâmetros, checkpoint de 5 MB), `buffer_size=5000`, 50 épocas.

## Resultados

### Synthetic (in-distribution: treino = teste)

| Agente | Recompensa | Acurácia multiclasse | F1 | Sensib. (dengue) | Especif. | Testes/ep. | Custo/acerto |
|--------|-----------:|---------------------:|-----:|-----------------:|---------:|-----------:|-------------:|
| **dqn** | **−395,9** | **0,753** | **0,744** | **0,717** | 0,793 | 103,4 | 0,488 |
| clinical | −702,8 | 0,690 | 0,692 | 0,691 | 0,699 | 0,0 | — |
| random | −2288,3 | 0,633 | 0,695 | 0,625 | 0,826 | 90,8 | 0,451 |

*Desvio-padrão da recompensa: dqn 827 · clinical 447 · random 442.*

### Kriging (out-of-distribution: teste de generalização)

| Agente | Recompensa | Acurácia multiclasse | F1 | Sensib. (dengue) | Especif. | Testes/ep. | Custo/acerto |
|--------|-----------:|---------------------:|-----:|-----------------:|---------:|-----------:|-------------:|
| **dqn** | **−44,9** | **0,782** | **0,779** | **0,749** | 0,830 | 74,6 | 0,341 |
| clinical | −576,3 | 0,731 | 0,734 | 0,733 | 0,742 | 0,0 | — |
| random | −2150,8 | 0,660 | 0,721 | 0,647 | 0,851 | 90,8 | 0,437 |

*Desvio-padrão da recompensa: dqn 904 · clinical 391 · random 460.*

### Significância (teste pareado por seed)

| Comparação | Δ Recompensa | Vitórias | t | Δ Acurácia | Vitórias | t |
|------------|-------------:|:--------:|----:|-----------:|:--------:|----:|
| Synthetic — dqn vs clinical | +306,9 | 8/10 | 2,52 | +0,063 | 10/10 | 4,75 |
| Kriging — dqn vs clinical | +531,4 | 8/10 | 3,24 | +0,051 | 10/10 | 4,68 |
| dqn vs random (ambos) | — | 10/10 | — | — | 10/10 | — |

Em ambas as distribuições, o DQN supera o clínico com significância estatística
(p ≲ 0,03 na recompensa; p ≲ 0,001 na acurácia) e vence o random em 10/10 seeds.

## Conclusões

1. **O DQN é o melhor agente** nas duas distribuições — e a vantagem sobre o
   clínico é estatisticamente significativa (análise pareada).

2. **O DQN generaliza.** Treinado só no `synthetic`, ele continua em 1º no
   `kriging` (distribuição espacial real que nunca viu). A política aprendida não
   está viciada no layout sintético: ela captura algo transferível — testar onde
   há ambiguidade, confiar no clínico onde não há.

3. **Cuidado na leitura absoluta:** o `kriging` é intrinsecamente **mais fácil**
   (todos os agentes melhoram nele — o clínico sobe de 0,690 → 0,731 de acurácia
   sem mudar de estratégia). A prova de generalização é a **margem relativa**
   DQN-vs-clínico, que se mantém/cresce (+306,9 → +531,4 de recompensa).

4. **Eficiência:** no `kriging` o DQN faz **menos testes** (74,6 vs 103,4) e ainda
   assim tem **acurácia maior** (0,782 vs 0,753) — coerente com uma política que
   responde à observação, e não que testa uma fração fixa cega.

5. **Testar sem inteligência é pior que não agir:** o `random` testa quase tanto
   quanto o DQN, mas suas decisões finais aleatórias disparam as penalidades do
   delay (confirmar errado −20, descartar doente real −30), afundando a
   recompensa (−2150 a −2288). Isso valida o desenho da função de recompensa.

## Limitações

- **10 seeds.** Os testes pareados já são significativos; rodar 30 seeds apertaria
  os intervalos de confiança.
- **Variância alta do DQN** (std da recompensa ~830–900): o ganho se concentra nos
  cenários fáceis/médios (alta `clinical_specificity`); nos difíceis (especif. ~0,5)
  ele empata ou perde de pouco para o clínico. Treino mais longo / buffer maior /
  redução da resolução do mapa devem reduzir isso.
- **Comparação entre distribuições:** como o `kriging` é mais fácil, compare sempre
  a **margem sobre o clínico**, nunca o valor absoluto entre `synthetic` e `kriging`.

## Glossário das métricas

- **Recompensa Total** — objetivo real (custo de testes + acerto/erro das decisões
  com delay + placar final). Métrica primária.
- **Acurácia multiclasse** — fração de casos com a doença correta (dengue/chik/outro).
- **F1 / Sensibilidade / Especificidade / Precisão** — métricas binárias (dengue vs. não).
- **Testes/ep.** — nº médio de testes de laboratório por episódio (custo operacional).
- **Custo/acerto** — custo de testes dividido por diagnóstico correto (eficiência).

## Arquivos gerados

```
results/baseline/          # benchmark synthetic
results/baseline_kriging/  # benchmark kriging
  benchmark_raw.csv        # uma linha por (agente, seed)
  benchmark_mean.csv       # média por agente
  benchmark_std.csv        # desvio-padrão por agente
  confusion_maps/          # matriz de confusão por agente/seed
  epidemic_maps/           # mapa da verdade por seed
results/dqn/policy_best.pth # checkpoint DQN (~5 MB)
results/kriging/*.npz       # superfícies Ordinary Kriging (Rio 2015–16)
```

## Como reproduzir

```bash
# 1) Treinar o DQN (synthetic, ~várias horas em GPU)
poetry run python agents/deepq/train.py --config experiments/configs/train/dqn_delay5.yaml

# 2) Benchmark in-distribution (synthetic)
poetry run python experiments/evaluate.py --config experiments/configs/benchmark.yaml

# 3) Gerar as superfícies Kriging (uma vez) e rodar o benchmark de generalização
poetry run python -m dengue_envs.data.build_kriging_surfaces
poetry run python experiments/evaluate.py --config experiments/configs/benchmark_kriging.yaml
```

Ambiente e semente idênticos entre os agentes garantem comparabilidade. As seeds
usadas: `[100, 150, 200, 250, 300, 350, 400, 450, 500, 550]`.
