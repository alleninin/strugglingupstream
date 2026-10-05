"""Bounded opponent snapshots and practice positions from training games."""

import random
from collections import deque
from copy import copy, deepcopy
from functools import partial

from bots.greedy_bot import GreedyBot
from bots.random_bot import RandomAgent


class SelfPlayPool:
    def __init__(self, num_players, num_decks, seed=None, capacity=3):
        self.num_players = num_players
        self.num_decks = num_decks
        self.rng = random.Random(seed)
        self.snapshots = deque(maxlen=capacity)
        self.anchor = None

    def capture(self, agent, anchor=False):
        network = deepcopy(agent.policy_net).eval().requires_grad_(False)
        policy = partial(copy(agent).act, network=network)
        policy.history_features = getattr(agent, "history_features", False)
        if anchor:
            self.anchor = policy
        else:
            self.snapshots.append(policy)

    def policies(self, seed):
        """Sample each seat from 60% Greedy, 30% frozen DDQN and 10% Random."""
        policies = []
        opponents = tuple(self.snapshots) + ((self.anchor,) if self.anchor else ())
        for seat in range(1, self.num_players):
            draw = self.rng.random()
            if opponents and 0.6 <= draw < 0.9:
                policies.append(self.rng.choice(opponents))
            elif draw >= 0.9:
                policies.append(
                    RandomAgent(seed=None if seed is None else seed * 100 + seat).act
                )
            else:
                policies.append(
                    GreedyBot(
                        num_players=self.num_players, num_decks=self.num_decks
                    ).act
                )
        return policies


class EndgamePool:
    def __init__(self, seed=None, capacity=256):
        self.rng = random.Random(seed)
        self.positions = deque(maxlen=capacity)
        self.candidate = None

    def consider(self, env, legal):
        """Keep the first non-forced pressure decision, before later mistakes compound."""
        game = env.game
        opponents = [
            len(hand)
            for seat, hand in enumerate(game.hands)
            if seat != env.agent_seat and hand
        ]
        if (
            self.candidate is None
            and len(legal) > 1
            and len(game.hands[env.agent_seat]) <= 12
            and opponents
            and min(opponents) <= 8
        ):
            self.candidate = deepcopy(game)

    def finish(self, lost):
        if lost and self.candidate is not None:
            self.positions.append(self.candidate)
        self.candidate = None

    def sample(self, probability):
        if self.positions and self.rng.random() < probability:
            return self.rng.choice(self.positions)
        return None
