"""Scalar Q network with explicit, public after-move hand features.

These are counts, not an action policy or a search through opponents' hands.
They expose combo preservation to the learner without requiring it to discover
integer subtraction, counting thresholds and runs from sparse game outcomes.
"""
import torch
from torch import nn
from .dqn_agent import QNetwork


class HandQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.state_dim = state_dim
        self.register_buffer('feature_schema', torch.tensor([state_dim, action_dim]))
        self.q = QNetwork(state_dim + action_dim + 26)

    def encode(self, x):
        state, action = x[:, :self.state_dim], x[:, self.state_dim:]
        before, played = state[:, :15], action[:, :15]
        after = (before - played).clamp_min(0)
        normal = after[:, :12]  # Straights never include 2 or jokers.
        ranks = after[:, :13]  # Jokers never form pairs, triples or bombs.
        stats = torch.stack((
            after.sum(1) / 54,
            (after > 0).sum(1) / 15,
            ((ranks == 1).sum(1) + after[:, 13:].sum(1)) / 15,
            (ranks >= 2).sum(1) / 15,
            (ranks >= 3).sum(1) / 15,
            (ranks >= 4).sum(1) / 15,
            (normal.unfold(1, 5, 1) >= 1).all(2).sum(1) / 8,
            (normal.unfold(1, 3, 1) >= 2).all(2).sum(1) / 10,
            (normal.unfold(1, 2, 1) >= 3).all(2).sum(1) / 11,
            after[:, 11:].sum(1) / 8,
            ((before[:, :13] >= 4) & (ranks < 4) & (played[:, :13] < 4)).sum(1) / 15,
        ), dim=1)
        normalized_state = torch.cat((before / 4, state[:, 15:]), dim=1)
        normalized_action = torch.cat((played / 4, action[:, 15:]), dim=1)
        return torch.cat((normalized_state, normalized_action, after / 4, stats), dim=1)

    def forward(self, x):
        return self.q(self.encode(x))
