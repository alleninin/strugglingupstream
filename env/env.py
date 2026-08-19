import numpy as np
from typing import List, Optional, Callable

from game.rules import Game
from game.moves import Move
from . import features
from agents.random_agent import RandomAgent


class ZhengShangYouEnv:
    def __init__(self, num_players: int = 4, num_decks: int = 2,
                 opponent_policies: Optional[List[Callable]] = None, seed=None):
        self.num_players = num_players
        self.num_decks = num_decks
        self.opponent_policies = opponent_policies
        self.agent_seat = 0
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self._default_random = RandomAgent(seed=seed)
        self.game: Optional[Game] = None

    def _opponent_policy(self, seat: int) -> Callable:
        if self.opponent_policies is not None and seat - 1 < len(self.opponent_policies):
            pol = self.opponent_policies[seat - 1]
            if pol is not None:
                return pol
        return self._default_random.act

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            self.seed = seed
        starting = int(self.rng.integers(0, self.num_players))
        self.game = Game(num_players=self.num_players, num_decks=self.num_decks,
                         starting_player=starting, seed=self.seed)
        return self._roll_to_agent()

    def _roll_to_agent(self):
        while not self.game.done and self.game.current_player != self.agent_seat:
            seat = self.game.current_player
            legal = self.game.legal_moves(seat)
            obs = features.state_vector(self.game, seat)
            move = self._opponent_policy(seat)(obs, legal)
            if move not in legal:
                move = legal[0]
            self.game.apply_move(seat, move)
        if self.game.done:
            return None
        return features.state_vector(self.game, self.agent_seat)

    def get_legal_moves(self) -> List[Move]:
        return self.game.legal_moves(self.agent_seat)

    def step(self, action: Move):
        seat = self.agent_seat
        legal = self.game.legal_moves(seat)
        if action not in legal:
            action = legal[0]

        self.game.apply_move(seat, action)
        self._roll_to_agent()

        if self.game.done:
            return None, self._terminal_reward(seat), True, {"next_legal_moves": []}

        state = features.state_vector(self.game, seat)
        legal = self.game.legal_moves(seat)
        return state, 0.0, False, {"next_legal_moves": legal}

    def _terminal_reward(self, seat: int) -> float:
        try:
            rank = self.game.finish_order.index(seat) + 1
        except ValueError:
            rank = self.num_players
        return 1.0 - 2.0 * (rank - 1) / (self.num_players - 1)
