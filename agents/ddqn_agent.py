from typing import List, NamedTuple, Optional

import numpy as np
import torch
import torch.optim as optim

from .base import BaseAgent
from .dueling_dqn import DuelingQNetwork
from .prioritized_replay import PrioritizedReplayBuffer
from .runtime import resolve_device
from env import features
from game.moves import Move


class ReplayTransition(NamedTuple):
    state: np.ndarray
    moves: np.ndarray
    chosen: int
    reward: float
    next_state: Optional[np.ndarray]
    next_moves: Optional[np.ndarray]


class DDQNAgent(BaseAgent):
    def __init__(self, state_dim: int, action_dim: int, lr: float = 1e-3,
                 gamma: float = 0.95, epsilon: float = 0.5,
                 epsilon_decay: float = 0.995, min_epsilon: float = 0.05,
                 buffer_size: int = 20000, batch_size: int = 64,
                 target_update: int = 200, alpha: float = 0.6, beta: float = 0.4,
                 beta_anneal_steps: int = 100000, seed: int = None, device="auto"):
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
        self.policy_net = DuelingQNetwork(state_dim, action_dim).to(self.device)
        self.target_net = DuelingQNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.buffer = PrioritizedReplayBuffer(
            capacity=buffer_size, alpha=alpha, beta=beta,
            beta_anneal_steps=beta_anneal_steps, rng=self.rng)
        self.step_count = 0
        self.demonstrations = None

    def _t(self, arr):
        return torch.as_tensor(np.asarray(arr, dtype=np.float32), device=self.device)

    def _indices(self, arr):
        return torch.as_tensor(arr, dtype=torch.long, device=self.device)

    @staticmethod
    def _move_vectors(moves):
        return np.stack([features.move_vector(move) for move in moves])

    def _pack_moves(self, move_sets):
        lengths = np.array([len(moves) for moves in move_sets], dtype=np.int64)
        offsets = np.cumsum(lengths) - lengths
        owners = self._indices(np.repeat(np.arange(len(lengths)), lengths))
        return self._t(np.concatenate(move_sets)), owners, lengths, offsets

    @staticmethod
    def _mean_advantage(advantages, owners, lengths):
        # One differentiable reduction for the whole batch, not one graph per
        # replay item. Gradients through the legal-action mean are essential.
        sums = advantages.new_zeros(len(lengths)).index_add(0, owners, advantages)
        return sums / torch.as_tensor(lengths, device=advantages.device, dtype=advantages.dtype)

    def _best_indices(self, advantages, lengths, offsets):
        # Pad only scalar scores, never whole state/action feature tensors. This
        # keeps argmax on the device and preserves first-index tie breaking.
        columns = np.arange(lengths.max())
        indices = offsets[:, None] + columns
        valid = columns < lengths[:, None]
        indices = np.minimum(indices, len(advantages) - 1)
        scores = advantages[self._indices(indices)]
        mask = torch.as_tensor(valid, dtype=torch.bool, device=self.device)
        best = scores.masked_fill(~mask, -torch.inf).argmax(dim=1)
        return self._indices(offsets) + best

    def act(self, obs, legal_moves: List[Move]) -> Move:
        if not legal_moves:
            return None
        if self.rng.random() < self.epsilon:
            return legal_moves[int(self.rng.integers(len(legal_moves)))]
        with torch.no_grad():
            _, advantages = self.policy_net.forward_grouped(
                self._t(np.asarray(obs)[None, :]), self._t(self._move_vectors(legal_moves)),
                torch.zeros(len(legal_moves), dtype=torch.long, device=self.device))
            # V(s) and the mean advantage are constant across this state's moves.
            chosen = int(advantages.argmax().item())
        return legal_moves[chosen]

    def observe(self, transition) -> None:
        state, action, reward, next_state, done, next_legal, cur_legal = transition
        if not cur_legal:
            raise ValueError("DDQN requires the current legal moves for dueling advantages")
        chosen = cur_legal.index(action)
        terminal = done or not next_legal
        # Replay stores one state per decision and one contiguous matrix of move
        # features, avoiding duplicated states and repeated stacking when sampled.
        self.buffer.push(ReplayTransition(
            np.array(state, dtype=np.float32, copy=True), self._move_vectors(cur_legal),
            chosen, float(reward),
            None if terminal else np.array(next_state, dtype=np.float32, copy=True),
            None if terminal else self._move_vectors(next_legal)))
        self.step_count += 1
        if self.step_count % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        self._learn()

    def _targets(self, data):
        targets = self._t([item.reward for item in data])
        nonterminal = [i for i, item in enumerate(data) if item.next_moves is not None]
        if not nonterminal:
            return targets
        items = [data[i] for i in nonterminal]
        moves, owners, lengths, offsets = self._pack_moves([item.next_moves for item in items])
        states = self._t(np.stack([item.next_state for item in items]))
        with torch.no_grad():
            _, online_a = self.policy_net.forward_grouped(states, moves, owners)
            target_v, target_a = self.target_net.forward_grouped(states, moves, owners)
            chosen = self._best_indices(online_a, lengths, offsets)
            next_q = target_v + target_a[chosen] - self._mean_advantage(target_a, owners, lengths)
            targets[self._indices(nonterminal)] += self.gamma * next_q
        return targets

    def _predictions(self, data):
        moves, owners, lengths, offsets = self._pack_moves([item.moves for item in data])
        states = self._t(np.stack([item.state for item in data]))
        values, advantages = self.policy_net.forward_grouped(states, moves, owners)
        chosen = self._indices(offsets + np.array([item.chosen for item in data]))
        return values + advantages[chosen] - self._mean_advantage(advantages, owners, lengths)

    def _learn(self) -> None:
        if len(self.buffer) < self.batch_size:
            return
        sample = self.buffer.sample(self.batch_size)
        if sample is None:
            return
        data, idxs, weights = sample
        targets = self._targets(data)
        predictions = self._predictions(data)
        td = predictions - targets
        loss = (self._t(weights) * torch.nn.functional.smooth_l1_loss(
            predictions, targets, reduction="none")).mean()
        if self.demonstrations is not None and self.demonstrations.weight:
            loss = loss + self.demonstrations.weight * self.demonstrations.loss(self)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()
        self.buffer.update_priorities(idxs, td.detach().abs().cpu().numpy())
        self.buffer.anneal_beta()

    def save(self, path: str) -> None:
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path: str) -> None:
        self.policy_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.policy_net.state_dict())
