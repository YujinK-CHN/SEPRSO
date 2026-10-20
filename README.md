# SEPRSO

Multi-agent RL research on SESiL, over **PettingZoo** and **OpenSpiel**, with
room for several methods (PSRO, PPO, GEMS, ...) compared under one budget and
one evaluation protocol.

Status:
- **`psro`** works: standard PSRO with a PPO best-response oracle, its design
  following OpenSpiel's `psro_v2` (not a port). On Kuhn poker NashConv falls
  from 0.917 (uniform) to about 0.13 within 200k env steps.
- `ppo` (self-play) and `gems` are registered placeholders that raise
  `NotImplementedError`.
- `random` is a pipeline check, not a method.

## Installation

```bash
conda create -n seprso python=3.10 -y && conda activate seprso
pip install torch --index-url https://download.pytorch.org/whl/cu121   # when methods need it
pip install -r requirements.txt
```

OpenSpiel has wheels for Linux and macOS only; on Windows use WSL. Everything
else imports without it.

## How to run

```bash
python main.py --method random --env kuhn_poker --budget 100000 --seed 0
python main.py --method psro --env kuhn_poker --budget 200000 --eval-metrics nash_conv
bash run.sh --method psro --env leduc_poker --budget 1000000 --seeds 0,1,2
pytest -q
```

`--eval-metrics nash_conv` gives exact NashConv on OpenSpiel turn-based games
and on the native matrix games. PettingZoo games have no exact equivalent; use
`vs_random` / `self_play` there.

Output goes to `results/<exp-name>/<env>/<method>/seed<seed>/`:
`config.json`, `eval.jsonl` (one record per evaluation watermark, same schema
for every method) and `train.jsonl` (method internals).

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
  meta_solvers.py         payoff tensor -> meta-strategy (uniform, nash LP, prd, auto)
  oracles.py              best-response oracles (PPOOracle)
  learners/
    networks.py           masked actor-critic MLP
    ppo.py                CleanRL-style PPO for one seat vs fixed opponents
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
tests/test_psro.py        meta-solvers, oracle, PSRO, NashConv (closed-form checks)
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
