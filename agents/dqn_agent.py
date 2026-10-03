import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List

from .base import BaseAgent
from .runtime import resolve_device
from game.moves import Move


class QNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden=(128, 64)):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class DQNAgent(BaseAgent):
    def __init__(self, state_dim: int, action_dim: int, lr: float = 1e-3,
                 gamma: float = 0.95, epsilon: float = 0.5,
                 epsilon_decay: float = 0.995, min_epsilon: float = 0.05,
                 buffer_size: int = 20000, batch_size: int = 64,
                 target_update: int = 200, seed: int = None, device="auto"):
        self.input_dim = state_dim + action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.batch_size = batch_size
        self.target_update = target_update
        self.rng = np.random.default_rng(seed)
        if seed is not None:
            torch.manual_seed(seed)

        self.device = resolve_device(device)

        self.policy_net = QNetwork(self.input_dim).to(self.device)
        self.target_net = QNetwork(self.input_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

        self.buffer = []
        self.buffer_size = buffer_size
        self.buffer_pos = 0
        self.step_count = 0
        self.demonstrations = None

    def _q_batch(self, phis, net):
        if len(phis) == 0:
            return torch.zeros(0)
        x = torch.from_numpy(np.stack(phis)).to(self.device)
        return net(x).squeeze(-1)

    def act(self, obs, legal_moves: List[Move]) -> Move:
        if not legal_moves:
            return None
        if self.rng.random() < self.epsilon:
            return legal_moves[int(self.rng.integers(len(legal_moves)))]
        phis = [self._phi(obs, m) for m in legal_moves]
        with torch.no_grad():
            qvals = self._q_batch(phis, self.policy_net).cpu().numpy()
        return legal_moves[int(np.argmax(qvals))]

    def observe(self, transition) -> None:
        state, action, reward, next_state, done, next_legal = transition
        phi = self._phi(state, action)
        next_phis = None if (done or not next_legal) else [self._phi(next_state, m) for m in next_legal]
        item = (phi, float(reward), next_phis, bool(done))
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(item)
        else:
            self.buffer[self.buffer_pos] = item
        self.buffer_pos = (self.buffer_pos + 1) % self.buffer_size

        self.step_count += 1
        if self.step_count % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        self._learn()

    def _learn(self) -> None:
        if len(self.buffer) < self.batch_size:
            return
        idxs = self.rng.integers(0, len(self.buffer), size=self.batch_size)
        batch = [self.buffer[i] for i in idxs]
        states = np.stack([b[0] for b in batch])
        rewards = np.array([b[1] for b in batch], dtype=np.float32)

        targets = rewards.copy()
        non_terminal = [(i, b[2]) for i, b in enumerate(batch)
                        if not b[3] and b[2] is not None and len(b[2]) > 0]
        if non_terminal:
            all_next = [p for _, nph in non_terminal for p in nph]
            with torch.no_grad():
                nq = self._q_batch(all_next, self.target_net).cpu().numpy()
            offset = 0
            for i, nph in non_terminal:
                end = offset + len(nph)
                targets[i] += self.gamma * nq[offset:end].max()
                offset = end

        sx = torch.from_numpy(states).to(self.device)
        pred = self.policy_net(sx).squeeze(-1)
        loss = nn.functional.smooth_l1_loss(pred, torch.from_numpy(targets).to(self.device))
        if self.demonstrations is not None and self.demonstrations.weight:
            loss = loss + self.demonstrations.weight * self.demonstrations.loss(self)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()

    def save(self, path: str) -> None:
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path: str) -> None:
        self.policy_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.policy_net.state_dict())
