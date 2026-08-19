"""Base agent interface.

An agent exposes ``act(obs, legal_moves) -> Move``. ``obs`` is the fixed-length
state feature vector; ``legal_moves`` is the list of ``Move`` objects valid this
turn (including the PASS sentinel when following). Learning agents additionally
implement ``observe(transition)`` where
``transition = (state, action, reward, next_state, done, next_legal_moves)``.
"""
from abc import ABC, abstractmethod
from typing import List, Any

from game.moves import Move


class BaseAgent(ABC):
    epsilon: float = 0.0

    @abstractmethod
    def act(self, obs, legal_moves: List[Move]) -> Move:
        ...

    def observe(self, transition: Any) -> None:
        """Called after ``env.step`` with the full transition (learning only)."""
        pass

    def reset_episode(self) -> None:
        pass

    def save(self, path: str) -> None:
        pass

    def load(self, path: str) -> None:
        pass
