from abc import ABC, abstractmethod
from typing import Any, List

from env import features
from game.moves import Move


class BaseAgent(ABC):
    epsilon: float = 0.0

    def _phi(self, state, move):
        state = state[: getattr(self, "state_dim", len(state))]
        return features.combined_vector(state, features.move_vector(move))

    @abstractmethod
    def act(self, obs, legal_moves: List[Move]) -> Move: ...

    def observe(self, transition: Any) -> None:
        pass

    def reset_episode(self) -> None:
        pass

    def save(self, path: str) -> None:
        pass

    def load(self, path: str) -> None:
        pass
