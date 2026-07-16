# Plano — estado atual

Documento curto. Histórico detalhado das rodadas: `old/plano_historico.md`.
Resumo da faxina de legado: `REFACTOR.md`.

## O que está pronto

- Ambiente modular (`core` + wrappers + `make_env`), recompensa com delay, lab delay, horizonte derivado
- Geradores: `synthetic` e `kriging` (dengue + chik; Zika fora do treino)
- Baselines no benchmark: `clinical`, `random`, `qlearning` (com Q-table)
- Q-Learning tabular com estado `rich_v1`, treino/watch/YAML
- Visualização de geradores (`view_generator`) e export de superfícies Kriging
- Legado fora do path ativo → pasta `old/` (DQN/PPO não foram mexidos)

## Próximas prioridades

1. **Congelar baseline** — treinar Q-Learning (synthetic e/ou kriging) + benchmark com as 3 políticas; guardar CSVs de referência
2. **Registrar DQN no `AGENT_REGISTRY`** — mesmo contrato `EpisodeRunner` / checkpoint `.pth` (código atual em `agents/deepq/` fica até lá)
3. **`epi_confirm` com superfície Kriging** — usar intensidade / `P(disease|cell)` no lugar do limiar de contagem
4. **PPO no mesmo pipeline** (depois do DQN)
5. **`Rio2016Generator`** — série temporal real além da densidade espacial

## Como rodar (atalho)

```bash
poetry install
poetry run pytest
poetry run python -m dengue_envs.data.build_kriging_surfaces
poetry run python agents/qlearning/train.py --config experiments/configs/train/qlearning_kriging.yaml
poetry run python experiments/evaluate.py --config experiments/configs/benchmark.yaml
```

Detalhes: `README.md`.
