"""
Networks shared by the learners.

Observations are flattened, so one MLP serves every backend (OpenSpiel
information-state vectors, PettingZoo board planes). Swap in a CNN or a
recurrent core here when an environment needs one.

Illegal actions are masked by setting their logits to a large negative number
before the softmax, so they get (numerically) zero probability and zero
gradient -- both in acting and in the PPO ratio.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

MASK_VALUE = -1e9


def _layer(in_dim, out_dim, gain):
    layer = nn.Linear(in_dim, out_dim)
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.zeros_(layer.bias)
    return layer


class ActorCritic(nn.Module):
    """Separate actor and critic MLPs (CleanRL's default layout)."""

    def __init__(self, obs_shape, num_actions, hidden=(128, 128)):
        super().__init__()
        self.obs_dim = int(np.prod(obs_shape))
        self.num_actions = int(num_actions)

        def body():
            layers, last = [], self.obs_dim
            for width in hidden:
                layers += [_layer(last, width, np.sqrt(2)), nn.Tanh()]
                last = width
            return layers, last

        actor, last = body()
        self.actor = nn.Sequential(*actor, _layer(last, self.num_actions, 0.01))
        critic, last = body()
        self.critic = nn.Sequential(*critic, _layer(last, 1, 1.0))

    def forward(self, obs, mask):
        obs = obs.reshape(obs.shape[0], -1)
        logits = self.actor(obs).masked_fill(~mask, MASK_VALUE)
        value = self.critic(obs).squeeze(-1)
        return logits, value
