# SEPRSO

Multi-agent RL research on SESiL, over **PettingZoo** and **OpenSpiel**, with
room for several methods (PSRO, PPO, GEMS, ...) compared under one budget and
one evaluation protocol. Developed for Linux (native, a remote server, or WSL).

Status:
- **`psro`** works. It follows OpenSpiel's `psro_v2` in structure and defaults
  (re-implemented on this codebase's env interface, so it also runs on
  PettingZoo), with psro_v2's three oracles: RL with policy gradient (`pg`),
  RL with DQN (`dqn`), and exact best response (`br`, OpenSpiel only).
- `ppo` (self-play) and `gems` are registered placeholders that raise
  `NotImplementedError`.
- `random` is a pipeline check, not a method.

## Installation (Linux / WSL)

```bash
conda create -n seprso python=3.11 -y && conda activate seprso
pip install torch --index-url https://download.pytorch.org/whl/cpu    # or a CUDA build
pip install -r requirements.txt
pytest -q
```

Without conda, `python3 -m venv .venv && source .venv/bin/activate` works the
same. OpenSpiel ships Linux and macOS wheels only, so on a Windows machine run
everything inside WSL.

## How to run

```bash
python main.py --method random --env kuhn_poker --budget 100000 --seed 0

# PSRO as in psro_v2_example.py, with its three oracles
python main.py --method psro --env kuhn_poker --psro-oracle br  --budget 100000000 --eval-metrics nash_conv
python main.py --method psro --env kuhn_poker --psro-oracle pg  --budget 100000000 --eval-metrics nash_conv
python main.py --method psro --env kuhn_poker --psro-oracle dqn --budget 100000000 --eval-metrics nash_conv

bash run.sh --method psro --env leduc_poker --budget 100000000 --seeds 0,1,2
```

`--eval-metrics nash_conv` gives exact NashConv on OpenSpiel turn-based games
and on the native matrix games. PettingZoo games have no exact equivalent; use
`vs_random` / `self_play` there.

Output goes to `results/<exp-name>/<env>/<method>/seed<seed>/`:
`config.json`, `eval.jsonl` (one record per evaluation watermark, same schema
for every method) and `train.jsonl` (method internals: `psro_iteration`
records with meta-strategies and payoffs, `oracle_episode` records with losses).

## PSRO and psro_v2

Defaults are `psro_v2_example.py`'s: PG oracle with QPG loss, a 4x256 ReLU MLP,
Adam, critic lr 1e-2, policy lr 1e-3, entropy cost 0.001, 8 critic updates per
policy update; 10,000 training episodes per response; 1000 simulations per
payoff entry; PRD with 50,000 iterations and gamma 1e-10; at most 100
iterations. The DQN oracle uses the example's 4x256 network, batch 32, lr 1e-2,
target update every 1000 steps and learning every 10.

Where this differs from psro_v2, on purpose:

| | psro_v2 | here | why |
|---|---|---|---|
| Learners | JAX (`open_spiel/python/jax`) | PyTorch ports, same algorithms | one framework for the codebase |
| Default meta-solver | `alpharank` | `auto` (Nash LP for 2-player zero-sum, else PRD) | alpharank not ported yet |
| Default oracle | `BR` | `pg` | BR exists only for OpenSpiel; `pg` runs everywhere |
| PG policy loss | softmax over all actions | illegal actions masked | untrained Q of illegal actions otherwise pulls probability to them |
| Stopping | iterations | iterations **or** env-step budget | the budget is the cross-method currency |

Not ported yet: alpharank, `rm` / `uniform_biased` meta-solvers, rectified
training, symmetric-game mode, `number_policies_selected > 1`.

## Layout

```
config.py                 single source of truth: ENV_PRESETS + every hyper-parameter
main.py                   single entry point; builds shared pieces, runs --method
run.sh                    launcher over methods x seeds
seprso/
  envs/
    base.py               MultiAgentEnv / EnvSpec / TimeStep -- the one interface
    openspiel_env.py      OpenSpiel adapter (sequential + simultaneous, chance resolved)
    pettingzoo_env.py     PettingZoo adapter (AEC + Parallel, any module path)
    native_env.py         dependency-free matrix games (rps, ...) for debugging
  policies.py             Policy interface, RandomPolicy, MixturePolicy
  rollout.py              play_episode(): the only env loop; per-agent credit assignment
  meta_game.py            Population + cached empirical PayoffTable
  meta_solvers.py         payoff tensor -> meta-strategy (uniform, nash, prd, auto)
  oracles.py              RLOracle (pg / dqn) and exact BestResponseOracle
  learners/
    pg.py                 QPG / RPG / RM / A2C (port of OpenSpiel's JAX policy_gradient)
    dqn.py                DQN (port of OpenSpiel's JAX dqn)
    ppo.py, networks.py   PPO, for the future self-play `ppo` method
  exploitability.py       exact NashConv; mixture -> behaviour policy by reach
  budget.py               BudgetTracker, in env steps
  evaluation.py           Evaluator + metric registry (vs_random, self_play)
  logging.py              JSONL eval/train streams
  registry.py             METHODS / METRICS registries
  methods/
    base.py               Method contract
    random_baseline.py    pipeline check
    psro.py               PSRO
    ppo.py gems.py        placeholders
tests/test_core.py        env adapters, rollout, budget, payoff table, pipeline
tests/test_psro.py        meta-solvers, oracles, PSRO, NashConv (closed-form checks)
```

## Design rules

- **One env interface.** Every step takes `{agent: action}`; every `TimeStep`
  says which agents act next (`acting`). Turn-based and simultaneous games are
  the same loop. Rewards are reported for *all* agents on every transition.
- **Budget = env steps.** Every training step a method takes is charged,
  including payoff-table simulation. Evaluation is not charged.
- **Same evaluation for everyone.** Methods expose `policies()` (one `Policy`
  per seat); the `Evaluator` scores it at the same watermarks with the same
  metrics, so `eval.jsonl` files from different methods line up.
- **Registries, not edits.** A new method is a file in `seprso/methods/` with
  `@METHODS.register('name')`, imported in `methods/__init__.py`, plus its
  argument group in `config.py`. A new metric is `@METRICS.register('name')`.
- **Environments by preset.** Add frequently used envs to `ENV_PRESETS`; try
  others ad hoc with `--env openspiel:<game>` or
  `--env pettingzoo:<module>[@parallel]`.
