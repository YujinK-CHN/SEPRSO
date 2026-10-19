"""
GEMS. Placeholder.

Population-based like PSRO, so it is expected to reuse seprso.meta_game for
its empirical game and seprso.policies.MixturePolicy for its output.
"""

from __future__ import annotations

from ..registry import METHODS
from .base import NotImplementedMethod


@METHODS.register('gems')
class GEMS(NotImplementedMethod):
    reference = 'GEMS'
