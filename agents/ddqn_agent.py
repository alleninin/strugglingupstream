"""DDQN agent with a Dueling Q-Network and Prioritized Experience Replay.

Combines three improvements over the vanilla ``DQNAgent``:
  * Double DQN (DDQN) targets: the greedy next action is selected with the online
    (policy) network but its value is read from the target network, which reduces
    the Q-value overestimation typical of vanilla DQN.
  * Dueling architecture: ``Q(s,a) = V(s) + (A(s,a) - mean_a A(s,a))``.
  * Prioritized Experience Replay (SumTree): samples important transitions more
    often and corrects the bias with importance-sampling weights (beta -> 1.0).

The replay transition is
    (state, action, reward, next_state, done, next_legal_moves, cur_legal_moves)
i.e. the same raw ``(state, move)`` form used by the other agents, plus the legal
moves available when the action was taken (needed for the exact dueling advantage
mean during training).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Any

from .base import BaseAgent
from .dueling_dqn import DuelingQNetwork
from .prioritized_replay import PrioritizedReplayBuffer
from game.moves import Move


class DDQNAgent(BaseAgent):
    def __init__(self, state_dim: int, action_dim: int, lr: float = 1e-3,
                 gamma: float = 0.95, epsilon: float = 0.5,
                 epsilon_decay: float = 0.995, min_epsilon: float = 0.05,
                 buffer_size: int = 20000, batch_size: int = 64,
                 target_update: int = 200, alpha: float = 0.6, beta: float = 0.4,
                 beta_anneal_steps: int = 100000, seed: int = None):
        self.input_dim = state_dim + action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.batch_size = batch_size
        self.target_update = target_update
        self.rng = np.random.default_rng(seed)

        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.policy_net = DuelingQNetwork(state_dim, action_dim).to(self.device)
        self.target_net = DuelingQNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

        self.buffer = PrioritizedReplayBuffer(
            capacity=buffer_size, alpha=alpha, beta=beta,
            beta_anneal_steps=beta_anneal_steps, rng=self.rng)
        self.step_count = 0

    def _t(self, arr):
        return torch.from_numpy(np.asarray(arr, dtype=np.float32)).to(self.device)

    def _q_over_moves(self, net, phis):
        """Dueling Q for a single state's legal moves (exact advantage mean)."""
        if not phis:
            return np.zeros(0)
        with torch.no_grad():
            v, a = net(self._t(np.stack(phis)))
        v = v.detach().cpu().numpy().ravel()
        a = a.detach().cpu().numpy().ravel()
        return v + a - a.mean()

    def act(self, obs, legal_moves: List[Move]) -> Move:
        if not legal_moves:
            return None
        phis = [self._phi(obs, m) for m in legal_moves]
        if self.rng.random() < self.epsilon:
            return legal_moves[int(self.rng.integers(len(legal_moves)))]
        with torch.no_grad():
            v, a = self.policy_net(self._t(np.stack(phis)))
        v = v.detach().cpu().numpy().ravel()
        a = a.detach().cpu().numpy().ravel()
        q = v + a - a.mean()  # single state -> exact dueling aggregation
        return legal_moves[int(np.argmax(q))]

    def observe(self, transition) -> None:
        # (state, action, reward, next_state, done, next_legal, cur_legal)
        state, action, reward, next_state, done, next_legal, cur_legal = transition
        phi = self._phi(state, action)
        next_phis = (None if (done or not next_legal)
                     else [self._phi(next_state, m) for m in next_legal])
        cur_phis = [self._phi(state, m) for m in cur_legal]
        self.buffer.push((phi, float(reward), next_phis, bool(done), cur_phis))

        self.step_count += 1
        if self.step_count % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        self._learn()

    def _learn(self) -> None:
        if len(self.buffer) < self.batch_size:
            return
        sample = self.buffer.sample(self.batch_size)
        if sample is None:
            return
        data, idxs, weights = sample
        B = len(data)

        # --- Targets (no grad): DDQN over next legal moves ---
        targets = np.zeros(B, dtype=np.float32)
        for i, (phi, reward, next_phis, done, cur_phis) in enumerate(data):
            if done or not next_phis:
                targets[i] = reward
            else:
                q_policy = self._q_over_moves(self.policy_net, next_phis)
                q_target = self._q_over_moves(self.target_net, next_phis)
                best = int(np.argmax(q_policy))
                targets[i] = reward + self.gamma * q_target[best]

        # --- Predictions (grad) for the played (s, a) ---
        all_phi = np.stack([d[0] for d in data])
        v_all, a_all = self.policy_net(self._t(all_phi))  # [B, 1], grad
        preds = []
        for i, (phi, reward, next_phis, done, cur_phis) in enumerate(data):
            if cur_phis:
                with torch.no_grad():
                    _, ca = self.policy_net(self._t(np.stack(cur_phis)))
                mean_a = ca.detach().mean()
            else:
                mean_a = a_all[i].detach()
            q = v_all[i] + a_all[i] - mean_a
            preds.append(q.squeeze(-1))
        q_pred = torch.cat(preds)  # [B]

        target_t = torch.from_numpy(targets).to(self.device)
        weight_t = torch.from_numpy(weights).to(self.device)
        loss = (weight_t * (q_pred - target_t) ** 2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        td = torch.abs(q_pred.detach() - target_t).cpu().numpy()
        self.buffer.update_priorities(idxs, td)
        self.buffer.anneal_beta()

    def save(self, path: str) -> None:
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path: str) -> None:
        self.policy_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.policy_net.state_dict())
