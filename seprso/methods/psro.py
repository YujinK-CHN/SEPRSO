"""
PSRO -- Policy-Space Response Oracles (Lanctot et al., 2017).

Not a port of OpenSpiel's psro_v2, but the same decomposition, so its knobs
mean the same thing here:

    populations          one Population per seat, seeded with RandomPolicy
    empirical game       meta_game.PayoffTable, --psro-sims-per-entry episodes
                         per entry, charged to the budget
    meta-solver          meta_solvers.META_SOLVERS[--psro-meta-solver]
    training selector    who the best response trains against
                           probabilistic  the opponents' meta-strategy (standard PSRO)
                           uniform        every opponent member equally
    oracle               oracles.PPOOracle, --psro-br-steps env steps per response

One iteration:
    1. for every seat, train a best response against the other seats'
       training mixtures -- all seats against the SAME current meta-strategy
       (psro_v2's simultaneous update);
    2. add the responses to their populations;
    3. simulate only the new entries of the payoff tensor;
    4. re-solve for the meta-strategy.

Output (policies()): per seat, the MixturePolicy of its population under the
current meta-strategy.

Budget: an iteration that starts is completed, so the run can overshoot the
budget by one payoff-table update. The final eval record logs the real
budget_used.
"""

from __future__ import annotations

import numpy as np

from ..meta_game import PayoffTable, Population
from ..meta_solvers import META_SOLVERS
from ..policies import MixturePolicy, RandomPolicy
from ..registry import METHODS
from .base import Method

# Log the full payoff tensor while it is small enough to be readable.
_LOG_PAYOFFS_MAX_ENTRIES = 400


@METHODS.register('psro')
class PSRO(Method):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from ..learners.ppo import NetworkPolicy, PPOConfig
        from ..oracles import PPOOracle

        a = self.args
        self._network_policy_cls = NetworkPolicy
        self.env = self.env_fn()
        self.agents = self.spec.agents
        self.populations = {agent: Population() for agent in self.agents}
        for pop in self.populations.values():
            pop.add(RandomPolicy(), kind='random', iteration=0)
        self.table = PayoffTable(self.env, self.populations, a.psro_sims_per_entry)
        self.solver = META_SOLVERS.get(a.psro_meta_solver)
        self.oracle = PPOOracle(self.spec, PPOConfig.from_args(a), device=a.device, seed=a.seed)
        self.meta = None
        self.iteration = 0

    # -------------------------------------------------------------- loop
    def train(self):
        self._update_meta_game()
        while not self.budget.exhausted:
            self.evaluator.maybe_evaluate(self)
            self.iteration += 1

            responses = {}
            for seat in self.agents:
                opponents = {agent: self._training_mixture(agent)
                             for agent in self.agents if agent != seat}
                policy, updates = self.oracle(
                    self.env, seat, opponents, self.rng, self.budget,
                    steps=self.args.psro_br_steps,
                    init_policy=self._warm_start(seat),
                    on_update=lambda stats, seat=seat: self.logger.train(
                        'ppo_update', {'iteration': self.iteration, 'seat': seat,
                                       'budget_used': self.budget.used, **stats}))
                if policy is not None:
                    responses[seat] = (policy, updates)

            if not responses:
                break   # budget ran out before any response was trained
            for seat, (policy, updates) in responses.items():
                self.populations[seat].add(policy, kind='ppo_br', iteration=self.iteration,
                                           updates=updates, budget_used=self.budget.used)
            self._update_meta_game()
        self.env.close()

    def policies(self):
        return {agent: MixturePolicy(self.populations[agent].policies, self.meta[agent])
                for agent in self.agents}

    # ----------------------------------------------------------- helpers
    def _update_meta_game(self):
        new_entries = self.table.update(self.rng, budget=self.budget)
        payoffs = self.table.tensor()
        a = self.args
        distributions = self.solver(payoffs, iterations=a.psro_prd_iterations,
                                    dt=a.psro_prd_dt, gamma=a.psro_prd_gamma)
        self.meta = dict(zip(self.agents, distributions))

        record = {'iteration': self.iteration, 'budget_used': self.budget.used,
                  'new_entries': new_entries,
                  'population_sizes': {agent: len(pop) for agent, pop in self.populations.items()},
                  'meta_strategy': {agent: dist for agent, dist in self.meta.items()}}
        if int(np.prod(payoffs.shape[:-1])) <= _LOG_PAYOFFS_MAX_ENTRIES:
            record['payoffs'] = payoffs
        self.logger.train('psro_iteration', record)

    def _training_mixture(self, agent):
        policies = self.populations[agent].policies
        if self.args.psro_training_selector == 'uniform':
            weights = np.ones(len(policies))
        else:
            weights = self.meta[agent]
        return MixturePolicy(policies, weights)

    def _warm_start(self, seat):
        if self.args.psro_br_init != 'latest':
            return None
        latest = self.populations[seat].policies[-1]
        return latest if isinstance(latest, self._network_policy_cls) else None
