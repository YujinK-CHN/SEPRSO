"""
PSRO -- Policy-Space Response Oracles. Placeholder.

Planned shape (to be decided together before writing it):
    * seprso.meta_game.Population per seat, seeded with RandomPolicy;
    * seprso.meta_game.PayoffTable for the empirical game (charged to budget);
    * a meta-solver over the payoff tensor (Nash / PRD / alpha-rank / uniform);
    * a best-response oracle (e.g. PPO or DQN) trained against the
      MixturePolicy of the opponents' meta-strategy;
    * policies() returns the MixturePolicy per seat.
"""

from __future__ import annotations

from ..registry import METHODS
from .base import NotImplementedMethod


@METHODS.register('psro')
class PSRO(NotImplementedMethod):
    reference = 'Lanctot et al., 2017, "A Unified Game-Theoretic Approach to MARL"'
