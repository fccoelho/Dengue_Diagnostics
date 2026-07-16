# Dengue_Diagnostics

Gymnasium environment for learning policies that diagnose dengue in the presence of chikungunya.

## About

In `DengueDiag-v0`, two epidemics run in the same city. For each reported case the agent chooses: lab test (dengue or chik), epidemiological confirm, accept the clinical diagnosis, or a final decision (confirm / discard).

Cases come from a **plugable generator** (`env.generator` in YAML):

| Generator | Spatial layout | Time |
|-----------|----------------|------|
| `synthetic` (default) | SIR + truncated normals around foci | SIR |
| `kriging` | Ordinary Kriging on Rio 2015–16 notifications (`P(cell \| disease)`) | SIR |

Only **dengue** and **chikungunya** are modeled (no Zika in training).

Active agents use `make_env` + wrappers. Archived code lives in `old/` (see `REFACTOR.md`). DQN/PPO are unchanged and not yet on the benchmark registry.

## Reward model

Computed by `dengue_envs.core.reward.RewardEngine` in three layers:

1. **Immediate cost** (same day):

   | Action | Cost |
   |--------|------|
   | Test dengue (0) / chik (1) | 1.0 |
   | Epi confirm (2) | 0.5 |
   | Do nothing (3) | 0.1 |
   | Confirm (4) / Discard (5) | 0.0 |

   Lab results arrive after `lab_delay_days` (default = `reward_delay_days`). Use `0` for immediate results.

2. **Delayed decision** (`reward_delay_days`, default **5**): confirm/discard pay `+10` if correct; wrong confirm `-20`; discarding a real case `-30`.

3. **Final score**: `+1` per correct diagnosis; `-10` per wrong case that was never tested.

Weights are configurable via `make_env` / YAML. Episode ends at `(epilength - 1) + settle_days`.

### Observation / action

Dict of per-case sequences (`clinical_diagnostic`, `testd`, `testc`, `epiconf`, `tnot`), wrapped for RL into a 4-channel map + one case at a time (`map_tensor` + `case_by_case`).

Actions (per case): test dengue (0), test chik (1), epi confirm (2), do nothing (3), confirm (4), discard (5).

## Installation

Requires **Python >=3.12, <3.13**:

```bash
poetry env use python3.12
poetry install
poetry run pytest
```

## Project layout

```
dengue_envs/      # env, core, wrappers, generators, rendering, metrics
agents/           # clinical, random, qlearning (+ deepq/ppo not yet unified)
experiments/      # YAML + evaluate.py
old/              # archived legacy (not imported by the active path)
REFACTOR.md       # what moved / what is active
plano.md          # current priorities
```

## Quick commands

**Benchmark** (clinical + random; uncomment `qlearning` after training):

```bash
poetry run python experiments/evaluate.py --config experiments/configs/benchmark.yaml
```

**Q-Learning** — synthetic / Kriging:

```bash
poetry run python agents/qlearning/train.py --config experiments/configs/train/qlearning_default.yaml
poetry run python -m dengue_envs.data.build_kriging_surfaces
poetry run python agents/qlearning/train.py --config experiments/configs/train/qlearning_kriging.yaml
```

**Visualize generators** (green = dengue, red = chik):

```bash
poetry run python -m dengue_envs.data.view_generator --generator both --seed 42
```

`--size` = map grid; `--episize` ≈ number of cases (SIR population).

**Watch**:

```bash
poetry run python agents/qlearning/watch.py --q-table results/qlearning/q_table.pkl
```

More detail: `agents/qlearning/README.md`, `experiments/README.md`.

### Agents

| Agent | Train | Benchmark |
|-------|-------|-----------|
| Clinical / Random | — | yes |
| Q-Learning | `agents/qlearning/train.py` | yes (needs `.pkl`) |
| DQN / PPO | scripts under `agents/deepq`, `agents/ppo` | not registered yet |

## TODO / roadmap

- [x] Reward/lab delay, discard semantics, final score, episode horizon
- [x] Kriging generator (dengue + chik) + `view_generator`
- [x] Legacy out of the active path (`old/`)
- [ ] Freeze reproducible baseline CSVs
- [ ] DQN / PPO on the benchmark registry
- [ ] Kriging intensity inside `epi_confirm`
- [ ] Compare against the health department workflow
