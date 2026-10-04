from typing import List, NamedTuple, Optional
from collections import deque

import numpy as np
import torch
import torch.optim as optim

from .base import BaseAgent
from .dueling_dqn import DuelingQNetwork
from .dqn_agent import QNetwork
from .hand_q_network import HandQNetwork
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
    discount: Optional[float] = None


class DDQNAgent(BaseAgent):
    def __init__(self, state_dim: int, action_dim: int, lr: float = 1e-3,
                 gamma: float = 0.95, epsilon: float = 0.5,
                 epsilon_decay: float = 0.995, min_epsilon: float = 0.05,
                 buffer_size: int = 20000, batch_size: int = 64,
                 target_update: int = 200, alpha: float = 0.6, beta: float = 0.4,
                 beta_anneal_steps: int = 100000, seed: int = None, device="auto",
                 dueling=False, n_step=1, learning_starts=64, train_every=1,
                 planning_features=None, partition_features=False):
        if n_step < 1 or train_every < 1 or learning_starts < 0:
            raise ValueError("invalid replay update schedule")
        self.input_dim = state_dim + action_dim
        self.state_dim, self.action_dim = state_dim, action_dim
        self.dueling = dueling
        if planning_features is None:
            planning_features = not dueling and min(state_dim, action_dim) >= 15
        self.planning_features = planning_features
        self.partition_features = partition_features
        if partition_features and not planning_features:
            raise ValueError("partition features require hand features")
        if planning_features and (dueling or min(state_dim, action_dim) < 15):
            raise ValueError("hand features require scalar Q and rank-count observations")
        self.n_step, self.learning_starts, self.train_every = n_step, learning_starts, train_every
        self.pending = deque()
        self.learning_updates = 0
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
        self.policy_net = self._network().to(self.device)
        self.target_net = self._network().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.buffer = PrioritizedReplayBuffer(
            capacity=buffer_size, alpha=alpha, beta=beta,
            beta_anneal_steps=beta_anneal_steps, rng=self.rng)
        self.step_count = 0
        self.demonstrations = None

    def _network(self):
        if self.planning_features:
            return HandQNetwork(self.state_dim, self.action_dim, self.partition_features)
        return (DuelingQNetwork(self.state_dim, self.action_dim) if self.dueling
                else QNetwork(self.state_dim + self.action_dim))

    def _q_batch(self, phis, net):
        return net(self._t(phis)).squeeze(-1)

    def _scores(self, net, states, moves, owners, lengths):
        if not self.dueling:
            return net(torch.cat((states.index_select(0, owners), moves), dim=1)).squeeze(-1)
        values, advantages = net.forward_grouped(states, moves, owners)
        return values[owners] + advantages - self._mean_advantage(advantages, owners, lengths)[owners]

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
            scores = self._scores(self.policy_net,
                self._t(np.asarray(obs)[None, :self.state_dim]), self._t(self._move_vectors(legal_moves)),
                torch.zeros(len(legal_moves), dtype=torch.long, device=self.device), [len(legal_moves)])
            chosen = int(scores.argmax().item())
        return legal_moves[chosen]

    def observe(self, transition) -> None:
        state, action, reward, next_state, done, next_legal, cur_legal = transition
        if not cur_legal:
            raise ValueError("DDQN requires the current legal moves for dueling advantages")
        chosen = cur_legal.index(action)
        current_moves = self._move_vectors(cur_legal if self.dueling else [action])
        if not self.dueling:
            chosen = 0
        terminal = done or not next_legal
        # Replay stores one state per decision and one contiguous matrix of move
        # features, avoiding duplicated states and repeated stacking when sampled.
        self.pending.append(ReplayTransition(
            np.array(state[:self.state_dim], dtype=np.float32, copy=True), current_moves,
            chosen, float(reward),
            None if terminal else np.array(next_state[:self.state_dim], dtype=np.float32, copy=True),
            None if terminal else self._move_vectors(next_legal)))
        if len(self.pending) >= self.n_step:
            self._store_pending()
        if terminal:
            while self.pending:
                self._store_pending()
        self.step_count += 1
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        if self.step_count >= self.learning_starts and self.step_count % self.train_every == 0:
            self._learn()

    def _store_pending(self):
        items = list(self.pending)[:self.n_step]
        first, last = items[0], items[-1]
        reward = sum(self.gamma ** i * item.reward for i, item in enumerate(items))
        self.buffer.push(first._replace(reward=reward, next_state=last.next_state,
                                        next_moves=last.next_moves, discount=self.gamma ** len(items)))
        self.pending.popleft()

    def reset_episode(self):
        # Normal terminal transitions flush the queue. An abandoned deal must
        # not leak unfinished returns into a different deal.
        self.pending.clear()

    def _targets(self, data):
        targets = self._t([item.reward for item in data])
        nonterminal = [i for i, item in enumerate(data) if item.next_moves is not None]
        if not nonterminal:
            return targets
        items = [data[i] for i in nonterminal]
        moves, owners, lengths, offsets = self._pack_moves([item.next_moves for item in items])
        states = self._t(np.stack([item.next_state for item in items]))
        with torch.no_grad():
            online_q = self._scores(self.policy_net, states, moves, owners, lengths)
            chosen = self._best_indices(online_q, lengths, offsets)
            if self.dueling:
                next_q = self._scores(self.target_net, states, moves, owners, lengths)[chosen]
            else:
                next_q = self.target_net(torch.cat((states, moves[chosen]), dim=1)).squeeze(-1)
            discounts = self._t([self.gamma if item.discount is None else item.discount for item in items])
            targets[self._indices(nonterminal)] += discounts * next_q
        return targets

    def _predictions(self, data):
        if not self.dueling:
            states = self._t(np.stack([item.state for item in data]))
            moves = self._t(np.stack([item.moves[item.chosen] for item in data]))
            return self.policy_net(torch.cat((states, moves), dim=1)).squeeze(-1)
        moves, owners, lengths, offsets = self._pack_moves([item.moves for item in data])
        states = self._t(np.stack([item.state for item in data]))
        scores = self._scores(self.policy_net, states, moves, owners, lengths)
        chosen = self._indices(offsets + np.array([item.chosen for item in data]))
        return scores[chosen]

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
        self.learning_updates += 1
        if self.learning_updates % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        self.buffer.update_priorities(idxs, td.detach().abs().cpu().numpy())
        self.buffer.anneal_beta()

    def save(self, path: str) -> None:
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path: str) -> None:
        weights = torch.load(path, map_location=self.device, weights_only=True)
        self.dueling = 'common.0.weight' in weights
        self.planning_features = 'feature_schema' in weights
        schema = weights.get('feature_schema')
        self.partition_features = schema is not None and len(schema) == 3
        if self.partition_features and int(schema[2]) != 2:
            raise ValueError('unsupported hand feature schema')
        if self.planning_features and int(weights['feature_schema'][1]) != self.action_dim:
            raise ValueError("checkpoint has incompatible action features")
        saved_dim = (int(weights['feature_schema'][0]) if self.planning_features else
                     weights['common.0.weight'].shape[1] if self.dueling else
                     weights['net.0.weight'].shape[1] - self.action_dim)
        if saved_dim not in (self.state_dim, features.legacy_state_dim(self.state_dim)):
            raise ValueError("checkpoint has incompatible player count or feature dimensions")
        self.state_dim, self.input_dim = saved_dim, saved_dim + self.action_dim
        self.policy_net = self._network().to(self.device)
        self.target_net = self._network().to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.optimizer.param_groups[0]['lr'])
        self.policy_net.load_state_dict(weights)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.pending.clear()
        self.buffer = PrioritizedReplayBuffer(
            capacity=self.buffer.tree.capacity, alpha=self.buffer.alpha,
            beta=self.buffer.beta_start, beta_anneal_steps=self.buffer.beta_anneal_steps, rng=self.rng)
        self.step_count = self.learning_updates = 0
        self.demonstrations = None
