# DQN (Deep Q-Network)

Agente deep RL integrado ao ambiente atual (`make_env` + `map_tensor` + `case_by_case`).
Uma ação por caso (`Discrete(6)`), no mesmo formato do random / Q-Learning.

Usa **Tianshou 2.x** (`DiscreteQLearningPolicy` + `DQN` + `OffPolicyTrainer`) e a
rede `DengueNet` (CNN no mapa + MLP nas coordenadas do caso).

## Arquivos

| Arquivo | Papel |
|---------|-------|
| `network.py` | `DengueNet` |
| `agent.py` | `DQNAgent`, `DQNAgentRunner`, load de `.pth` |
| `train.py` | Treino CLI (YAML) |
| `watch.py` | Visualização Pygame |

Scripts legados (treino/watch antigos) foram removidos do working tree —
continuam disponíveis no histórico do git. Restam apenas os shims em
`agents/deepq/files/` (ver README de lá) — **não** use no path ativo.

## Treinar

```bash
poetry run python agents/deepq/train.py --config experiments/configs/train/dqn_delay5.yaml

# smoke rápido
poetry run python agents/deepq/train.py --epochs 1 --output-dir results/dqn_smoke
```

Saídas em `results/dqn/`: `policy_best.pth`, `policy_final.pth`, `logs/`.

Com `reward_delay_days: 5`, use `n_step >= 5` (já é o default em `dqn_delay5.yaml`).

## Benchmark / watch

```bash
# Após treinar, descomente/adicione dqn em experiments/configs/benchmark.yaml
poetry run python experiments/evaluate.py --config experiments/configs/benchmark.yaml

poetry run python agents/deepq/watch.py --policy results/dqn/policy_best.pth --seed 100
```

## Checkpoint

`results/dqn/policy_best.pth` — `state_dict` da `DiscreteQLearningPolicy`.
Checkpoints legados (só pesos da rede) também são aceitos no load.
