"""
Meta-solvers: empirical payoff tensor -> one distribution per seat.

Design follows OpenSpiel's psro_v2/meta_strategies.py: a meta-solver is a pure
function of the payoff tensor, so the PSRO loop never knows which one it runs
and a new solver is one registered function.

Input  payoffs   shape [n_0, ..., n_{N-1}, N]; payoffs[..., k] is seat k's payoff
                 (the layout of meta_game.PayoffTable.tensor()).
Output list of N probability vectors, entry k of length n_k.

    uniform   every member equally -> PSRO reduces to fictitious play
    nash      exact Nash by linear programming; two-player zero-sum only
    prd       projected replicator dynamics (Lanctot et al., 2017); any N, any sum
    auto      nash if the table is two-player zero-sum, else prd
"""

from __future__ import annotations

import numpy as np

from .registry import Registry

META_SOLVERS = Registry('meta-solver')


def is_two_player_zero_sum(payoffs, tol=1e-6):
    if payoffs.shape[-1] != 2:
        return False
    scale = max(1.0, float(np.abs(payoffs).max()))
    return float(np.abs(payoffs.sum(axis=-1)).max()) <= tol * scale


@META_SOLVERS.register('uniform')
def uniform(payoffs, **_):
    return [np.full(n, 1.0 / n) for n in payoffs.shape[:-1]]


@META_SOLVERS.register('nash')
def nash(payoffs, **_):
    """Maximin LP for each player of a two-player zero-sum game.

    Estimated tables are only approximately zero-sum, so the game solved is
    A = (P_0 - P_1) / 2, the zero-sum part of the table.
    """
    from scipy.optimize import linprog

    if not is_two_player_zero_sum(payoffs, tol=1e-2):
        raise ValueError("Meta-solver 'nash' needs a two-player zero-sum table; use 'prd'.")
    A = (payoffs[..., 0] - payoffs[..., 1]) / 2.0
    return [_maximin(A, linprog), _maximin(-A.T, linprog)]


def _maximin(A, linprog):
    """argmax_x min_j (x^T A)_j over the simplex. Variables: x (m), v."""
    m, n = A.shape
    c = np.zeros(m + 1)
    c[-1] = -1.0                                     # maximise v
    A_ub = np.hstack([-A.T, np.ones((n, 1))])        # v - (x^T A)_j <= 0
    A_eq = np.hstack([np.ones((1, m)), np.zeros((1, 1))])
    bounds = [(0, None)] * m + [(None, None)]
    res = linprog(c, A_ub=A_ub, b_ub=np.zeros(n), A_eq=A_eq, b_eq=[1.0],
                  bounds=bounds, method='highs')
    if not res.success:
        raise RuntimeError(f'Nash LP failed: {res.message}')
    x = np.clip(res.x[:m], 0.0, None)
    return x / x.sum()


@META_SOLVERS.register('prd')
def prd(payoffs, iterations=100_000, dt=1e-3, gamma=1e-6, average_last=None, **_):
    """Projected replicator dynamics.

    Each step moves every seat's distribution along the replicator flow and
    projects it back onto the gamma-floored simplex, so no member's mass ever
    reaches zero (PSRO needs every member to stay reachable). The returned
    strategy is the average of the last `average_last` iterates (default: the
    last 10%), which damps the cycling replicator dynamics are prone to.
    """
    num_seats = payoffs.shape[-1]
    strategies = [np.full(n, 1.0 / n) for n in payoffs.shape[:-1]]
    if average_last is None:
        average_last = max(1, iterations // 10)
    window = [np.zeros(n) for n in payoffs.shape[:-1]]

    for t in range(iterations):
        new = []
        for k in range(num_seats):
            values = _values_against(payoffs[..., k], strategies, k)
            delta = values - values @ strategies[k]
            updated = strategies[k] + dt * strategies[k] * delta
            new.append(_project(updated, gamma))
        strategies = new
        if t >= iterations - average_last:
            for k in range(num_seats):
                window[k] += strategies[k]
    return [w / w.sum() for w in window]


def _values_against(tensor, strategies, seat):
    """Expected payoff of each of seat's pure members against the others' mixtures."""
    out = tensor
    # Contract the highest axes first so the remaining axis indices stay valid.
    for axis in reversed(range(len(strategies))):
        if axis != seat:
            out = np.tensordot(out, strategies[axis], axes=([axis], [0]))
    return out


def _project(x, gamma):
    x = np.maximum(x, gamma)
    return x / x.sum()


@META_SOLVERS.register('auto')
def auto(payoffs, **kwargs):
    if is_two_player_zero_sum(payoffs):
        return nash(payoffs, **kwargs)
    return prd(payoffs, **kwargs)
