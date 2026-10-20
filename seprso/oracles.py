"""
Best-response oracles.

Design follows OpenSpiel's psro_v2/rl_oracle.py: an oracle is handed one seat
and a fixed strategy for every other seat, and returns a new frozen policy for
that seat that (approximately) best-responds to it. The PSRO loop never knows
how the response was computed, so an exact oracle (for small OpenSpiel games)
or a DQN oracle can be swapped in behind the same call.

The opponents arrive as Policies -- normally MixturePolicy over a population
with meta-strategy weights, which samples one member per episode. That is how
psro_v2 plays a meta-strategy during best-response training too.
"""

from __future__ import annotations


class PPOOracle:
    """Approximate best response by training a fresh (or warm-started) PPO learner."""

    def __init__(self, spec, ppo_config, device='cpu', seed=0):
        from .learners.ppo import PPOLearner  # torch is needed only from here on
        self._learner_cls = PPOLearner
        self.spec = spec
        self.config = ppo_config
        self.device = device
        self.seed = seed
        self.calls = 0

    def __call__(self, env, seat, opponents, rng, budget, steps, init_policy=None,
                 on_update=None):
        """Return (policy, updates). policy is None if the budget allowed no update."""
        self.calls += 1
        learner = self._learner_cls(self.spec.obs_shape[seat], self.spec.num_actions[seat],
                                    self.config, device=self.device,
                                    seed=self.seed * 100_003 + self.calls)
        if init_policy is not None:
            learner.load_policy(init_policy)
        updates = learner.train(env, seat, opponents, rng, budget, steps, on_update=on_update)
        return (learner.policy() if updates else None), updates
