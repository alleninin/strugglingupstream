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
    def __init__(
        self, state_dim: int, action_dim: int, hidden=(128, 64), move_hidden=(64,)
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.common = _mlp(state_dim, hidden, hidden[-1])
        h_dim = hidden[-1]

        self.move_enc = _mlp(action_dim, move_hidden, move_hidden[-1])
        m_dim = move_hidden[-1]

        self.value_stream = _mlp(h_dim, hidden, 1)
        self.adv_stream = _mlp(h_dim + m_dim, hidden, 1)

    def forward(self, x):
        s = x[:, : self.state_dim]
        m = x[:, self.state_dim :]
        h = self.common(s)
        v = self.value_stream(h)
        me = self.move_enc(m)
        a = self.adv_stream(torch.cat([h, me], dim=1))
        return v, a

    def forward_grouped(self, states, moves, owners):
        """Encode each state once, then score its variable number of moves.

        owners[j] is the state index for moves[j]. Parameter names and shapes
        match forward(), so existing checkpoints remain compatible.
        """
        h = self.common(states)
        values = self.value_stream(h).squeeze(-1)
        encoded_moves = self.move_enc(moves)
        advantages = self.adv_stream(
            torch.cat([h.index_select(0, owners), encoded_moves], dim=1)
        ).squeeze(-1)
        return values, advantages


class QNetwork(nn.Module):
    """Scalar action-value network shared by current DDQN representations."""

    def __init__(self, input_dim, hidden=(128, 64)):
        super().__init__()
        self.net = _mlp(input_dim, hidden, 1)

    def forward(self, x):
        return self.net(x)
