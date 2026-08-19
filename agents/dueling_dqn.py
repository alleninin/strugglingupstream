"""Dueling Q-Network.

The network is fed the concatenated ``(state, move)`` vector used everywhere else in
this repo, but splits it internally:

    common features  = MLP(state)            # the shared "common feature layers"
    V(s)             = value_stream(common)  # scalar, state-value
    A(s, a)          = adv_stream(common, move_encoder(move))  # scalar, advantage

and combines them with the standard stable aggregation

    Q(s, a) = V(s) + (A(s, a) - mean_a A(s, a)).

Because actions here are structured/per-pair rather than a fixed discrete set, the
advantage mean is taken over the *legal moves of the state* (exact at decision time,
and over the stored legal set during training). See ``duel`` below.
"""

import numpy as np
import torch
import torch.nn as nn


def _mlp(input_dim: int, hidden, out_dim: int) -> nn.Sequential:
    layers = []
    prev = input_dim
    for h in hidden:
        layers.append(nn.Linear(prev, h))
        layers.append(nn.ReLU())
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class DuelingQNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden=(128, 64),
                 move_hidden=(64,)):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Common feature layers operate on the STATE only, so V(s) stays
        # independent of the chosen move.
        self.common = _mlp(state_dim, hidden, hidden[-1])
        h_dim = hidden[-1]

        # Advantage stream also reads an encoding of the move.
        self.move_enc = _mlp(action_dim, move_hidden, move_hidden[-1])
        m_dim = move_hidden[-1]

        self.value_stream = _mlp(h_dim, hidden, 1)
        self.adv_stream = _mlp(h_dim + m_dim, hidden, 1)

    def forward(self, x):
        s = x[:, :self.state_dim]
        m = x[:, self.state_dim:]
        h = self.common(s)
        v = self.value_stream(h)                       # [B, 1]
        me = self.move_enc(m)
        a = self.adv_stream(torch.cat([h, me], dim=1))  # [B, 1]
        return v, a


def duel(v, a, group_ids):
    """Apply Q = V + (A - mean_a A) for a batch.

    ``v``, ``a`` are numpy arrays of shape ``[B]``. ``group_ids`` is an array of
    shape ``[B]`` (e.g. state hashes / indices) that assigns each row to a state;
    the advantage mean is computed within each group so the constant offset cancels
    correctly for argmax/value at the state level.
    """
    v = np.asarray(v, dtype=np.float64).ravel()
    a = np.asarray(a, dtype=np.float64).ravel()
    group_ids = np.asarray(group_ids).ravel()
    q = np.empty(len(v), dtype=np.float64)
    for gid in np.unique(group_ids):
        mask = group_ids == gid
        q[mask] = v[mask] + a[mask] - a[mask].mean()
    return q
