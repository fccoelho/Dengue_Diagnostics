# Plano — estado atual

Documento curto. O histórico detalhado das rodadas e o registro da faxina de
legado (pasta `old/`, agora removida) vivem no histórico do git.

## O que está pronto

- Ambiente modular (`core` + wrappers + `make_env`), recompensa com delay, lab delay, horizonte derivado
- Geradores: `synthetic` e `kriging` (dengue + chik; Zika fora do treino)
- Baselines no benchmark: `clinical`, `random`, `qlearning` (com Q-table)
- Q-Learning tabular com estado `rich_v1`, treino/watch/YAML
- Visualização de geradores (`view_generator`) e export de superfícies Kriging
- Legado removido do working tree (antiga pasta `old/`); preservado no git (PPO ainda não foi mexido)
- **DQN integrado** ao framework (`agents/deepq/agent.py`, `network.py`,
  `train.py`) e registrado no `AGENT_REGISTRY` do benchmark (mesmo contrato
  `EpisodeRunner`, checkpoint `.pth`)

## Próximas prioridades

1. **Congelar baseline** — treinar Q-Learning e DQN (synthetic e/ou kriging),
   rodar o benchmark com as 4 políticas (`clinical`, `random`, `qlearning`,
   `dqn`) e guardar CSVs de referência (descomentar `dqn` em `benchmark.yaml`)
2. **`epi_confirm` com superfície Kriging** — usar intensidade / `P(disease|cell)` no lugar do limiar de contagem
3. **PPO no mesmo pipeline** — adaptar `agents/ppo` ao `CaseByCaseWrapper` e registrar no `AGENT_REGISTRY`
4. **`Rio2016Generator`** — série temporal real além da densidade espacial (hoje é stub que levanta `NotImplementedError`)

## Como rodar (atalho)

```bash
poetry install
poetry run pytest
poetry run python -m dengue_envs.data.build_kriging_surfaces
poetry run python agents/qlearning/train.py --config experiments/configs/train/qlearning_kriging.yaml
poetry run python agents/deepq/train.py --config experiments/configs/train/dqn_delay5.yaml
poetry run python experiments/evaluate.py --config experiments/configs/benchmark.yaml
```

Detalhes: `README.md`.
