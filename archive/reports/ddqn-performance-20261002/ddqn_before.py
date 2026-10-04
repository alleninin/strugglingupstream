import numpy as np
import torch
import torch.optim as optim
from typing import List

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
        if seed is not None:
            torch.manual_seed(seed)

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

    def act(self, obs, legal_moves: List[Move]) -> Move:
        if not legal_moves:
            return None
        if self.rng.random() < self.epsilon:
            return legal_moves[int(self.rng.integers(len(legal_moves)))]
        phis = [self._phi(obs, m) for m in legal_moves]
        with torch.no_grad():
            v, a = self.policy_net(self._t(np.stack(phis)))
        v = v.detach().cpu().numpy().ravel()
        a = a.detach().cpu().numpy().ravel()
        q = v + a - a.mean()
        return legal_moves[int(np.argmax(q))]

    def observe(self, transition) -> None:
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
        targets = np.array([d[1] for d in data], dtype=np.float32)

        non_term = [(i, d[2], d[1]) for i, d in enumerate(data)
                    if not d[3] and d[2]]
        if non_term:
            all_next = np.concatenate([np.stack(nl) for (_, nl, _) in non_term])
            with torch.no_grad():
                vpn, apn = self.policy_net(self._t(all_next))
                vtn, atn = self.target_net(self._t(all_next))
            vpn = vpn.detach().cpu().numpy().ravel()
            apn = apn.detach().cpu().numpy().ravel()
            vtn = vtn.detach().cpu().numpy().ravel()
            atn = atn.detach().cpu().numpy().ravel()
            offset = 0
            for i, nl, reward in non_term:
                n = len(nl)
                s, e = offset, offset + n
                offset = e
                qp = vpn[s:e] + apn[s:e] - apn[s:e].mean()
                qt = vtn[s:e] + atn[s:e] - atn[s:e].mean()
                best = int(np.argmax(qp))
                targets[i] = reward + self.gamma * qt[best]

        # Center advantages within each state's legal set, with gradients through
        # the mean. Detaching it incorrectly trains A even in a one-action state.
        cur_sets = [np.stack(d[4]) for d in data]
        v_cur, a_cur = self.policy_net(self._t(np.concatenate(cur_sets)))
        predictions = []
        offset = 0
        for d, phis in zip(data, cur_sets):
            chosen = int(np.flatnonzero(np.all(phis == d[0], axis=1))[0])
            advantages = a_cur[offset:offset + len(phis), 0]
            predictions.append(v_cur[offset + chosen, 0]
                               + advantages[chosen] - advantages.mean())
            offset += len(phis)
        q_pred = torch.stack(predictions)

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
