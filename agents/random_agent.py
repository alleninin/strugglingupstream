"""Uniformly random baseline agent."""
import random
from typing import List

from .base import BaseAgent
from game.moves import Move


class RandomAgent(BaseAgent):
    def __init__(self, seed: int = None):
        self.rng = random.Random(seed)

    def act(self, obs, legal_moves: List[Move]) -> Move:
        return self.rng.choice(legal_moves)
