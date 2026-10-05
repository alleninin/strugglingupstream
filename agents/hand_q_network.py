"""Scalar Q network with explicit, public after-move hand features.

These are own-hand counts/partitions, not an action policy or opponent search.
They expose combo preservation to the learner without requiring it to discover
integer subtraction, counting thresholds and runs from sparse game outcomes.
"""

import torch
from torch import nn

from .networks import QNetwork


class HandQNetwork(nn.Module):
    def __init__(
        self, state_dim, action_dim, partition_features=False, history_features=False
    ):
        super().__init__()
        self.state_dim = state_dim
        self.partition_features = partition_features
        schema = (
            [state_dim, action_dim, 2]
            if partition_features
            else [state_dim, action_dim]
        )
        if history_features:
            schema = [state_dim, action_dim, 3, int(partition_features)]
        self.register_buffer("feature_schema", torch.tensor(schema))
        self.q = QNetwork(
            state_dim + action_dim + 26 + (5 if partition_features else 0)
        )

    def encode(self, x):
        state, action = x[:, : self.state_dim], x[:, self.state_dim :]
        before, played = state[:, :15], action[:, :15]
        after = (before - played).clamp_min(0)
        normal = after[:, :12]
        ranks = after[:, :13]
        stats = torch.stack(
            (
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
                ((before[:, :13] >= 4) & (ranks < 4) & (played[:, :13] < 4)).sum(1)
                / 15,
            ),
            dim=1,
        )
        normalized_state = torch.cat((before / 4, state[:, 15:]), dim=1)
        normalized_action = torch.cat((played / 4, action[:, 15:]), dim=1)
        parts = [normalized_state, normalized_action, after / 4, stats]
        if self.partition_features:
            from env.hand_structure import hand_structure

            counts = (
                torch.cat((before, after), dim=0)
                .detach()
                .to("cpu")
                .numpy()
                .round()
                .astype("int16")
            )
            structures = [hand_structure(tuple(row.tolist())) for row in counts]
            structure = x.new_tensor(structures)
            old, new = structure[: len(x)], structure[len(x) :]
            parts.append(
                torch.stack(
                    (
                        old[:, 0] / 15,
                        new[:, 0] / 15,
                        (old[:, 0] - new[:, 0]) / 4,
                        new[:, 1] / 15,
                        (old[:, 1] - new[:, 1]) / 4,
                    ),
                    dim=1,
                )
            )
        return torch.cat(parts, dim=1)

    def forward(self, x):
        return self.q(self.encode(x))
