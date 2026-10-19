"""
Checks of the shared scaffold. Backend-specific tests skip when that backend is
not installed, so `pytest` is meaningful on any machine.

    pytest -q
"""

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import ENV_PRESETS, parse_env  # noqa: E402
from seprso.budget import BudgetTracker  # noqa: E402
from seprso.envs import make_env  # noqa: E402
from seprso.meta_game import PayoffTable, Population  # noqa: E402
from seprso.policies import FixedActionPolicy, MixturePolicy, RandomPolicy  # noqa: E402
from seprso.rollout import play_episode  # noqa: E402


def _has(module):
    try:
        __import__(module)
        return True
    except ImportError:
        return False


HAS_PZ = _has('pettingzoo')
HAS_OS = _has('pyspiel')


def _random_joint(env):
    return {a: RandomPolicy() for a in env.spec.agents}


def _check_episodes(env, n=20):
    """Generic invariants every backend must satisfy."""
    rng = np.random.default_rng(0)
    budget = BudgetTracker(10**9)
    for _ in range(n):
        ep = play_episode(env, _random_joint(env), rng, budget=budget, record=True)
        assert ep.terminated or ep.truncated
        assert set(ep.returns) == set(env.spec.agents)
        for agent, transitions in ep.trajectories.items():
            for t in transitions:
                assert t.legal_mask[t.action]
                assert t.obs.shape == env.spec.obs_shape[agent]
        if env.spec.zero_sum:
            assert abs(sum(ep.returns.values())) < 1e-6
    assert budget.used > 0
    env.close()


# ----------------------------------------------------------------- native
def test_rps_payoffs():
    env = make_env(parse_env('rps'))
    rng = np.random.default_rng(0)
    ep = play_episode(env, {'player_0': FixedActionPolicy(0), 'player_1': FixedActionPolicy(2)}, rng)
    assert ep.returns == {'player_0': 1.0, 'player_1': -1.0}
    assert ep.length == 1


def test_native_invariants():
    _check_episodes(make_env(parse_env('rps')))
    _check_episodes(make_env(parse_env('matching_pennies')))


def test_payoff_table_rps():
    env = make_env(parse_env('rps'))
    pops = {a: Population([FixedActionPolicy(i) for i in range(3)]) for a in env.spec.agents}
    budget = BudgetTracker(10**6)
    table = PayoffTable(env, pops, episodes_per_entry=1)
    assert table.update(np.random.default_rng(0), budget=budget) == 9
    assert budget.used == 9   # empirical-game simulation is charged
    np.testing.assert_allclose(table.tensor(), env.payoffs)
    pops['player_0'].add(RandomPolicy())
    assert table.update(np.random.default_rng(0)) == 3   # only the new row


def test_mixture_samples_per_episode():
    rng = np.random.default_rng(0)
    mix = MixturePolicy([FixedActionPolicy(0), FixedActionPolicy(1)], [0.25, 0.75])
    picks = []
    for _ in range(4000):
        mix.reset(rng)
        picks.append(mix.act(None, np.ones(2, bool), rng))
    assert abs(np.mean(picks) - 0.75) < 0.03


def test_truncation():
    env = make_env(parse_env('connect_four' if HAS_PZ else 'rps'), max_episode_steps=1)
    ep = play_episode(env, _random_joint(env), np.random.default_rng(0))
    assert ep.length == 1
    assert ep.truncated or ep.terminated


def test_parse_env_forms():
    assert parse_env('kuhn_poker')['backend'] == 'openspiel'
    p = parse_env('pettingzoo:mpe2.simple_tag_v3@parallel')
    assert p == {'backend': 'pettingzoo', 'id': 'mpe2.simple_tag_v3', 'api': 'parallel',
                 'name': 'simple_tag_v3'}


# ------------------------------------------------------------- pettingzoo
@pytest.mark.skipif(not HAS_PZ, reason='pettingzoo not installed')
@pytest.mark.parametrize('name', ['connect_four', 'tic_tac_toe', 'pz_rps'])
def test_pettingzoo_invariants(name):
    env = make_env(parse_env(name))
    assert env.spec.turn_based
    _check_episodes(env)


@pytest.mark.skipif(not HAS_PZ, reason='pettingzoo not installed')
def test_pettingzoo_winner_gets_reward():
    env = make_env(parse_env('connect_four'))
    rng = np.random.default_rng(1)
    ep = play_episode(env, _random_joint(env), rng, record=True)
    # Connect Four: +1 / -1 or a draw, and the final reward reaches BOTH
    # agents' last transitions even though only one of them made the last move.
    assert sorted(ep.returns.values()) in ([-1.0, 1.0], [0.0, 0.0])
    for agent, transitions in ep.trajectories.items():
        assert transitions[-1].terminated
        assert transitions[-1].reward == ep.returns[agent]


@pytest.mark.skipif(not HAS_PZ, reason='pettingzoo not installed')
def test_pettingzoo_parallel_api():
    env = make_env({'backend': 'pettingzoo', 'id': 'pettingzoo.classic.rps_v2',
                    'api': 'parallel', 'name': 'rps_parallel'})
    assert not env.spec.turn_based
    _check_episodes(env, n=3)


# -------------------------------------------------------------- openspiel
@pytest.mark.skipif(not HAS_OS, reason='open_spiel not installed')
@pytest.mark.parametrize('name', ['kuhn_poker', 'leduc_poker', 'goofspiel', 'os_tic_tac_toe'])
def test_openspiel_invariants(name):
    env = make_env(parse_env(name))
    _check_episodes(env)


# ------------------------------------------------------------- end to end
def test_main_random_end_to_end(tmp_path):
    import main
    env = 'connect_four' if HAS_PZ else 'rps'
    main.main(['--method', 'random', '--env', env, '--budget', '300',
               '--eval-episodes', '5', '--results-dir', str(tmp_path)])
    run_dir = tmp_path / 'check' / env / 'random' / 'seed0'
    records = [json.loads(line) for line in (run_dir / 'eval.jsonl').read_text().splitlines()]
    assert records[0]['budget_used'] == 0
    assert records[-1]['final'] and records[-1]['budget_used'] >= 300
    assert {'method', 'env', 'seed', 'vs_random/mean'} <= set(records[-1])
    assert (run_dir / 'config.json').exists()


@pytest.mark.parametrize('method', ['psro', 'ppo', 'gems'])
def test_stub_methods_fail_loudly(method, tmp_path):
    import main
    with pytest.raises(NotImplementedError):
        main.main(['--method', method, '--env', 'rps', '--budget', '10',
                   '--results-dir', str(tmp_path)])


def test_every_preset_is_well_formed():
    for name, preset in ENV_PRESETS.items():
        assert preset['backend'] in ('native', 'openspiel', 'pettingzoo'), name
        assert 'id' in preset, name
