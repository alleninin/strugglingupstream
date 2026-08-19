from abc import ABC, abstractmethod
from typing import List, Any

from game.moves import Move
from env import features


class BaseAgent(ABC):
    epsilon: float = 0.0

    def _phi(self, state, move):
        return features.combined_vector(state, features.move_vector(move))

    @abstractmethod
    def act(self, obs, legal_moves: List[Move]) -> Move:
        ...

    def observe(self, transition: Any) -> None:
        pass

    def reset_episode(self) -> None:
        pass

    def save(self, path: str) -> None:
        pass

    def load(self, path: str) -> None:
        pass
