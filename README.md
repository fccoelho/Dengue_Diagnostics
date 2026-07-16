# Dengue_Diagnostics

This is a Gymnasium environment for learning optimal policies for accurately detecting dengue cases in the presence of
other arbovirus cases such as chikungunya.

## About this environment

In the `DengueDiag-v0` environment, there are two epidemics going on in the same city, dengue and chikungunya. For every
case reported, the agent has to decide whether to run a lab test (dengue or chik), confirm the case based on
epidemiological evidence, accept the clinical diagnosis, or make a final decision (confirm as a true positive / discard
as a false positive).

At every time step, the agent receives the list of cases reported up to that time and decides on an action for every new
case reported at time step $t$ (assuming decisions for cases reported before $t$ were already made).

## Reward model

The reward is computed by `dengue_envs.core.reward.RewardEngine` and has **three layers**:

1. **Immediate cost** (paid the same day) — every action pays its operational cost:

   | Action | Cost |
   |--------|------|
   | Test for dengue (0) | 1.0 |
   | Test for chik (1) | 1.0 |
   | Epi confirm (2) | 0.5 |
   | Do nothing (3) | 0.1 |
   | Confirm (4) | 0.0 |
   | Discard (5) | 0.0 |

   Lab tests also model a **turnaround delay** (`lab_delay_days`, defaults to
   `reward_delay_days`): when a test is ordered the sample is taken *today*, but
   the result (and the update to `testd`/`testc`/`agent_diagnosis`) only becomes
   available `lab_delay_days` later. Set `lab_delay_days=0` for immediate results.

2. **Delayed decision outcome** (paid `reward_delay_days` later, default **5**) — the decisive actions only reveal
   whether they were good after the lab/epidemiological result "matures":

   - **Confirm**: correct when `agent_diagnosis == true disease` → `+reward_correct_decision` (default `+10`);
     otherwise → `penalty_incorrect_decision` (default `-20`, a false positive).
   - **Discard**: correct when the true disease is *other* → `+reward_correct_decision`; discarding a real
     arbovirus case is a false negative and gets `penalty_missed_case` (default `-30`, the most dangerous error in
     surveillance).

3. **Final score** (paid once at the end of the episode):

   - `+final_correct_bonus` (default `+1`) for every case whose `agent_diagnosis` matches the truth — the sum is
     equivalent to `episize - misdiagnosed`.
   - `penalty_untested_misdiagnosed` (default `-10`) for every case that ends up wrong **and was never tested**.

All weights are parameters of `DengueDiagnosticsEnv.__init__` (and can be set through `make_env`/config), so the whole
reward can be tuned without touching the environment loop. Setting `reward_delay_days=0` pays the decision outcomes
immediately.

### Observation Space

The observation space is a `Dict` of sequences (one entry per reported case):

```python
{
    "clinical_diagnostic": ((x, y, disease), ...),  # disease: 0 dengue, 1 chik, 2 other
    "testd": ((case_id, status), ...),               # status: 0 not tested, 1 neg, 2 pos, 3 inconclusive
    "testc": ((case_id, status), ...),
    "epiconf": ((case_id, confirmed), ...),          # confirmed: 0 no, 1 yes
    "tnot": ((case_id, day), ...),                    # day the case was reported
}
```

For RL training this Dict is turned into a 4-channel tensor by `DengueWrapper` (see `dengue_envs/wrappers`).

### Action Space

There are 6 possible actions per case:

- Test for dengue (0)
- Test for chik (1)
- Epi confirm (2): confirm based on epidemiological evidence
- Do nothing (3): accept the clinical diagnosis
- Confirm (4): confirm the case as a true positive
- Discard (5): discard the case as a false positive

## Installation

The project requires **Python >=3.12, <3.13**. To install:

```bash
poetry env use python3.12
poetry install
```

Run the test suite with:

```bash
poetry run pytest
```

## Project layout

```
dengue_envs/
├── data/        # world/epidemic generators (synthetic today; rio2016/kriging planned)
├── core/        # clinical model, epi confirm, case store, RewardEngine, LabResultQueue
├── envs/        # DengueDiagnosticsEnv (orchestrates step/reset, delegates rendering)
├── wrappers/    # map_tensor, case_by_case and the make_env factory
├── metrics/     # episode metrics (binary + multiclass) / confusion map
├── rendering/   # PygameRenderer, sprites, image assets (assets/*.png)
└── tests/
agents/          # RL agents (DQN via Tianshou, random, clinical, qlearning, ppo)
```

### Agents

| Agent | Module | Train | Benchmark | Watch |
|-------|--------|-------|-----------|-------|
| Clinical (baseline) | `agents/clinical/` | — | yes | — |
| Random (baseline) | `agents/random/` | — | yes | yes |
| Q-Learning (tabular) | `agents/qlearning/` | `agents/qlearning/train.py` | yes* | yes |
| DQN (deep) | `agents/deepq/` | `agents/deepq/files/agent_train.py` | planned | yes |

\* Q-Learning no benchmark requer Q-table treinada (`results/qlearning/q_table.pkl`). Ver `agents/qlearning/README.md`.

Benchmark all agents:

```bash
poetry run python experiments/evaluate.py --config experiments/configs/benchmark.yaml
```

Train Q-Learning (not run automatically):

```bash
poetry run python agents/qlearning/train.py --config experiments/configs/train/qlearning_default.yaml
```

### Rendering

Rendering lives entirely in `dengue_envs/rendering/` (not in the env):

- `assets/` — image icons (`dengue_test.png`, `chick_test.png`, ...); resolved via `rendering/assets.py` (`ASSETS_DIR`, `asset_path`, cached `load_image`).
- `sprites.py` — `CaseSprite` / `CaseGroup`.
- `pygame_renderer.py` — `PygameRenderer`: owns the window/surfaces and draws the map, the reward plot and the accuracy plot. The env only calls `create_sprites` / `update_sprites` / `render`.

The accuracy plot shows the **multiclass** accuracy (exact class match over dengue/chik/other), not the lenient binary one.

## TODO / roadmap

- [x] Delay in the reward (implemented in `RewardEngine`, active by default with `reward_delay_days=5`)
- [x] Delay in the lab test results (`LabResultQueue`, `lab_delay_days`, defaults to the reward delay)
- [x] Fix the discard reward semantics
- [x] Final reward model (bonus `episize - misdiagnosed` + `-10` for untested & misdiagnosed)
- [x] Principled episode horizon (`(epilength - 1) + settle_days` instead of a magic `+10`)
- [ ] Simulated data based on 2016 outbreak (`Rio2016Generator`)
- [ ] Kriging density surface for `epi_confirm` (no distribution assumption)
- [ ] Compare against the health department's real decision workflow (introduction/methodology)
- [ ] Evaluate over longer intervals
