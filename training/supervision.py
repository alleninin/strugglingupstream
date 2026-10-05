"""Temporary Greedy move-ranking supervision on ordinary training decisions."""

from collections import deque
from math import isfinite
from typing import NamedTuple

import numpy as np
import torch


class Example(NamedTuple):
    state: np.ndarray
    moves: np.ndarray
    chosen: int


class GreedySupervision:
    def __init__(
        self, weight=0.5, fraction=0.6, seed=None, capacity=4096, batch_size=16
    ):
        if not isfinite(weight) or weight < 0 or not 0 < fraction <= 1:
            raise ValueError(
                "supervision requires finite nonnegative weight and fraction in (0, 1]"
            )
        if capacity < 1 or batch_size < 1:
            raise ValueError("supervision capacity and batch size must be positive")
        self.initial_weight = weight
        self.fraction = fraction
        self.weight = weight
        self.batch_size = batch_size
        self.examples = deque(maxlen=capacity)
        self.rng = np.random.default_rng(seed)

    def schedule(self, episode, episodes):
        """Fade linearly to zero; the final episode always has zero supervision."""
        duration = max(1.0, (episodes - 1) * self.fraction)
        self.weight = self.initial_weight * max(0.0, 1.0 - episode / duration)
        if episode >= episodes - 1:
            self.weight = 0.0
        if not self.weight:
            self.examples.clear()

    def add(self, agent, state, legal, teacher):
        if self.weight and len(legal) > 1:
            action = teacher.act(state, legal)
            self.examples.append(
                Example(
                    np.array(state[: agent.state_dim], dtype=np.float32, copy=True),
                    agent._move_vectors(legal),
                    legal.index(action),
                )
            )

    def loss(self, agent):
        """Large-margin ranking stops pushing once the expert leads by 0.1 Q units."""
        if not self.examples:
            return next(agent.policy_net.parameters()).sum() * 0
        data = [
            self.examples[i]
            for i in self.rng.integers(len(self.examples), size=self.batch_size)
        ]
        moves, owners, lengths, offsets = agent._pack_moves(
            [item.moves for item in data]
        )
        states = agent._t(np.stack([item.state for item in data]))
        scores = agent._scores(agent.policy_net, states, moves, owners, lengths)
        expert = agent._indices(offsets + np.array([item.chosen for item in data]))
        margins = torch.full_like(scores, 0.1)
        margins[expert] = 0
        rivals = agent._best_indices(scores + margins, lengths, offsets)
        return (scores[rivals] + margins[rivals] - scores[expert]).mean()
