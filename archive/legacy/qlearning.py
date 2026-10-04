import numpy as np
from typing import List

from agents.base import BaseAgent
from game.moves import Move


class QLearningAgent(BaseAgent):
    def __init__(self, state_dim: int, action_dim: int, alpha: float = 0.05,
                 gamma: float = 0.95, epsilon: float = 0.2,
                 epsilon_decay: float = 0.9995, min_epsilon: float = 0.02,
                 seed: int = None, partition_features=False):
        self.partition_features = partition_features
        self.base_dim = state_dim + action_dim
        self.state_dim, self.action_dim = state_dim, action_dim
        self.dim = self.base_dim + state_dim * action_dim + (5 if partition_features else 0)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.rng = np.random.default_rng(seed)
        self.w = np.zeros(self.dim, dtype=np.float32)
        self.demonstrations = None

    def _phi(self, state, move):
        state = state[:self.state_dim]
        base = super()._phi(state, move)
        # Additive w_s*s + w_a*a gives the same action preference in every
        # state. Cross terms let the linear learner condition a move on its hand
        # and the table. Bounding counts keeps multi-deck features well scaled.
        action = base[len(state):]
        interactions = np.outer(np.tanh(state), np.tanh(action)).ravel()
        parts = [base, interactions]
        if self.partition_features:
            from env.hand_structure import partition_vector
            parts.append(partition_vector(state[:15], np.maximum(0, state[:15] - action[:15])))
        return np.concatenate(parts).astype(np.float32)

    def q(self, phi) -> float:
        return float(np.dot(self.w, phi))

    def act(self, obs, legal_moves: List[Move]) -> Move:
        if not legal_moves:
            return None
        if self.rng.random() < self.epsilon:
            return legal_moves[int(self.rng.integers(len(legal_moves)))]
        phis = [self._phi(obs, m) for m in legal_moves]
        vals = [self.q(p) for p in phis]
        return legal_moves[int(np.argmax(vals))]

    def observe(self, transition) -> None:
        state, action, reward, next_state, done, next_legal = transition
        phi = self._phi(state, action)
        if done or not next_legal:
            target = float(reward)
        else:
            next_phis = [self._phi(next_state, m) for m in next_legal]
            target = float(reward) + self.gamma * max(self.q(p) for p in next_phis)
        td = target - self.q(phi)
        # Rank counts vary with deck/hand size; normalize the update so one
        # large hand does not multiply the effective learning rate unchecked.
        self.w += (self.alpha * td / max(1.0, float(np.dot(phi, phi)))) * phi
        if self.demonstrations is not None and self.demonstrations.weight:
            self.demonstrations.update_linear(self, self.demonstrations.weight)
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def save(self, path: str) -> None:
        with open(path, "wb") as checkpoint:
            np.save(checkpoint, self.w)

    def load(self, path: str) -> None:
        from env import features
        w = np.load(path)
        # Preserve .npy compatibility: the five appended features identify the
        # new schema by length. Old additive and interaction weights still load.
        for state_dim in (self.state_dim, features.legacy_state_dim(self.state_dim)):
            base_dim = state_dim + self.action_dim
            interaction_dim = base_dim + state_dim * self.action_dim
            if w.shape not in ((base_dim,), (interaction_dim,), (interaction_dim + 5,)):
                continue
            self.state_dim, self.base_dim = state_dim, base_dim
            self.partition_features = w.shape == (interaction_dim + 5,)
            self.dim = interaction_dim + (5 if self.partition_features else 0)
            if w.shape == (base_dim,):
                w = np.pad(w, (0, self.dim - base_dim))
            self.w = w.astype(np.float32)
            self.demonstrations = None
            return
        raise ValueError(f"checkpoint dim {w.shape} is incompatible with this agent")
