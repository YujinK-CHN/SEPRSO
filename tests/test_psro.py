"""
PSRO pieces, each checked against something known in closed form.

    meta-solvers   Nash of RPS / matching pennies is uniform; PRD finds it too
    oracle         against 'always rock', the best response is 'paper'
    PSRO           on RPS the meta-strategy approaches uniform
    NashConv       Kuhn poker: the uniform policy's NashConv is a known constant,
                   and a mixture is aggregated by reach, not by naive averaging
"""

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import get_config, parse_env, resolve  # noqa: E402
from seprso.budget import BudgetTracker  # noqa: E402
from seprso.envs import make_env  # noqa: E402
from seprso.meta_solvers import META_SOLVERS, is_two_player_zero_sum  # noqa: E402
from seprso.policies import FixedActionPolicy, MixturePolicy, RandomPolicy  # noqa: E402

torch = pytest.importorskip('torch')

HAS_OS = True
try:
    import pyspiel  # noqa: F401
except ImportError:
    HAS_OS = False


def _matrix(name):
    return make_env(parse_env(name)).payoffs


# ------------------------------------------------------------ meta-solvers
@pytest.mark.parametrize('game', ['rps', 'matching_pennies'])
@pytest.mark.parametrize('solver', ['nash', 'prd', 'auto'])
def test_meta_solvers_find_uniform_nash(game, solver):
    payoffs = _matrix(game)
    for dist in META_SOLVERS.get(solver)(payoffs):
        np.testing.assert_allclose(dist, np.full(len(dist), 1 / len(dist)), atol=0.02)


def test_nash_on_biased_rps():
    # Rock beating scissors pays 2. Checked by the indifference condition (every
    # strategy in the support earns the same) rather than a hard-coded vector.
    payoffs = _matrix('rps').copy()
    payoffs[0, 2] = [2, -2]
    payoffs[2, 0] = [-2, 2]
    x, y = META_SOLVERS.get('nash')(payoffs)
    A = payoffs[..., 0]
    row_values = A @ y            # every row in the support earns the same
    col_values = x @ A
    support = x > 1e-6
    assert np.ptp(row_values[support]) < 1e-6
    assert np.ptp(col_values[y > 1e-6]) < 1e-6


def test_nash_refuses_general_sum():
    with pytest.raises(ValueError):
        META_SOLVERS.get('nash')(_matrix('prisoners_dilemma'))
    assert not is_two_player_zero_sum(_matrix('prisoners_dilemma'))
    # auto falls back to PRD, which finds the dominant strategy (defect = 1).
    x, y = META_SOLVERS.get('auto')(_matrix('prisoners_dilemma'), iterations=20_000, dt=1e-2)
    assert x[1] > 0.95 and y[1] > 0.95


def test_prd_three_players():
    # Three-player game where action 1 strictly dominates for every seat.
    payoffs = np.zeros((2, 2, 2, 3))
    for idx in np.ndindex(2, 2, 2):
        payoffs[idx] = [float(a) for a in idx]
    dists = META_SOLVERS.get('prd')(payoffs, iterations=20_000, dt=1e-2)
    assert all(d[1] > 0.95 for d in dists)


# ------------------------------------------------------------------ oracle
def _args(*extra):
    return resolve(get_config().parse_args(['--env', 'rps', *extra]))


def test_ppo_oracle_best_responds():
    from seprso.learners.ppo import PPOConfig
    from seprso.oracles import PPOOracle

    args = _args('--ppo-rollout-steps', '128', '--ppo-lr', '3e-3')
    env = make_env(args.env_preset)
    oracle = PPOOracle(env.spec, PPOConfig.from_args(args), seed=0)
    budget = BudgetTracker(10**6)
    policy, updates = oracle(env, 'player_1', {'player_0': FixedActionPolicy(0)},
                             np.random.default_rng(0), budget, steps=3000)
    assert updates > 0 and budget.used >= 3000
    probs = policy.action_probs(np.ones(1, np.float32), np.ones(3, bool))
    assert probs[1] > 0.9, probs     # paper beats rock


def test_ppo_respects_action_mask():
    from seprso.learners.ppo import PPOLearner, PPOConfig
    learner = PPOLearner((1,), 3, PPOConfig())
    policy = learner.policy()
    mask = np.array([False, True, False])
    rng = np.random.default_rng(0)
    assert all(policy.act(np.ones(1, np.float32), mask, rng) == 1 for _ in range(50))
    assert policy.action_probs(np.ones(1, np.float32), mask)[[0, 2]].sum() == 0


# -------------------------------------------------------------------- PSRO
def test_psro_rps_end_to_end(tmp_path):
    import main
    main.main(['--method', 'psro', '--env', 'rps', '--budget', '30000', '--seed', '0',
               '--psro-br-steps', '2000', '--psro-sims-per-entry', '50',
               '--ppo-rollout-steps', '128', '--ppo-lr', '3e-3',
               '--psro-prd-iterations', '2000', '--eval-episodes', '50',
               '--eval-metrics', 'nash_conv', '--eval-interval', '1',
               '--results-dir', str(tmp_path)])
    run_dir = tmp_path / 'check' / 'rps' / 'psro' / 'seed0'
    train = [json.loads(line) for line in (run_dir / 'train.jsonl').read_text().splitlines()]
    iterations = [r for r in train if r['kind'] == 'psro_iteration']
    assert len(iterations) >= 3
    last = iterations[-1]
    # Population grew by one response per seat per iteration.
    assert last['population_sizes'] == {'player_0': last['iteration'] + 1,
                                         'player_1': last['iteration'] + 1}
    evals = [json.loads(line) for line in (run_dir / 'eval.jsonl').read_text().splitlines()]
    assert evals[-1]['final']
    assert any(r['kind'] == 'ppo_update' for r in train)
    # The random seed policy is already Nash in RPS (NashConv 0). The first best
    # response moves away from it; PSRO must then converge back towards it.
    nash_convs = [e['nash_conv'] for e in evals]
    assert nash_convs[-1] < 0.25 and nash_convs[-1] < 0.6 * max(nash_convs), nash_convs


def test_matrix_nash_conv():
    from seprso.exploitability import matrix_nash_conv
    env = make_env(parse_env('rps'))
    uniform = {a: RandomPolicy() for a in env.spec.agents}
    rock = {a: FixedActionPolicy(0) for a in env.spec.agents}
    assert abs(matrix_nash_conv(env, uniform)) < 1e-12
    assert abs(matrix_nash_conv(env, rock) - 2.0) < 1e-12   # each seat gains 1 by paper


def test_psro_budget_is_charged(tmp_path):
    import main
    main.main(['--method', 'psro', '--env', 'rps', '--budget', '3000',
               '--psro-br-steps', '1000', '--psro-sims-per-entry', '10',
               '--ppo-rollout-steps', '64', '--psro-prd-iterations', '100',
               '--eval-episodes', '10', '--results-dir', str(tmp_path)])
    run_dir = tmp_path / 'check' / 'rps' / 'psro' / 'seed0'
    final = [json.loads(line) for line in (run_dir / 'eval.jsonl').read_text().splitlines()][-1]
    assert final['budget_used'] >= 3000


# ---------------------------------------------------------------- NashConv
@pytest.mark.skipif(not HAS_OS, reason='open_spiel not installed')
def test_nash_conv_uniform_kuhn_matches_openspiel():
    from open_spiel.python import policy as os_policy
    from open_spiel.python.algorithms import exploitability
    from seprso.exploitability import nash_conv

    env = make_env(parse_env('kuhn_poker'))
    ours = nash_conv(env, {a: RandomPolicy() for a in env.spec.agents}, None, 0)['nash_conv']
    reference = exploitability.nash_conv(env.game, os_policy.UniformRandomPolicy(env.game))
    assert abs(ours - reference) < 1e-9
    assert abs(ours - 0.9166666666666666) < 1e-6   # = 11/12, the known value


@pytest.mark.skipif(not HAS_OS, reason='open_spiel not installed')
def test_mixture_is_aggregated_by_reach():
    """'Always bet' + 'always check', mixed 50/50 per episode, is not 'bet with p=0.5' everywhere.

    After player 0 has bet, only the always-bet member can be playing, so at
    player 0's later decisions the aggregate must follow always-bet exactly.
    """
    from seprso.exploitability import aggregate

    env = make_env(parse_env('kuhn_poker'))
    mix = MixturePolicy([FixedActionPolicy(1), FixedActionPolicy(0)], [0.5, 0.5])
    tabular = aggregate(env, {'player_0': mix, 'player_1': RandomPolicy()})
    # Kuhn actions: 0 = pass, 1 = bet. Deal, player 0 decides, then after
    # pass / bet player 0 decides again -- a state only always-check reaches.
    for card in range(3):
        state = env.game.new_initial_state()
        state.apply_action(card)            # chance: player 0's card
        state.apply_action((card + 1) % 3)  # chance: player 1's card
        first = tabular.policy_for_key(state.information_state_string(0))
        np.testing.assert_allclose(first, [0.5, 0.5])
        state.apply_action(0)   # player 0 passes
        state.apply_action(1)   # player 1 bets
        second = tabular.policy_for_key(state.information_state_string(0))
        np.testing.assert_allclose(second, [1.0, 0.0])   # always-check passes (folds)


@pytest.mark.skipif(not HAS_OS, reason='open_spiel not installed')
def test_psro_kuhn_reduces_nash_conv(tmp_path):
    import main
    main.main(['--method', 'psro', '--env', 'kuhn_poker', '--budget', '60000',
               '--psro-br-steps', '8000', '--psro-sims-per-entry', '100',
               '--ppo-rollout-steps', '256', '--ppo-lr', '1e-3',
               '--eval-metrics', 'nash_conv', '--eval-interval', '20000',
               '--results-dir', str(tmp_path)])
    run_dir = tmp_path / 'check' / 'kuhn_poker' / 'psro' / 'seed0'
    evals = [json.loads(line) for line in (run_dir / 'eval.jsonl').read_text().splitlines()]
    assert evals[-1]['nash_conv'] < evals[0]['nash_conv'] * 0.6, [e['nash_conv'] for e in evals]
