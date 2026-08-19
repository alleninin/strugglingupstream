import numpy as np
from typing import List

from .base import BaseAgent
from game.moves import Move


class QLearningAgent(BaseAgent):
    def __init__(self, state_dim: int, action_dim: int, alpha: float = 0.05,
                 gamma: float = 0.95, epsilon: float = 0.2,
                 epsilon_decay: float = 0.9995, min_epsilon: float = 0.02,
                 seed: int = None):
        self.dim = state_dim + action_dim
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.rng = np.random.default_rng(seed)
        self.w = np.zeros(self.dim, dtype=np.float32)

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
        self.w += self.alpha * td * phi
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def save(self, path: str) -> None:
        np.save(path, self.w)

    def load(self, path: str) -> None:
        self.w = np.load(path)
