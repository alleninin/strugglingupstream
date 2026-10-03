import numpy as np
from typing import List, Optional, Callable

from game.rules import Game
from game.moves import Move
from . import features
from bots.greedy_bot import GreedyBot


class ZhengShangYouEnv:
    def __init__(self, num_players: int = 4, num_decks: int = 2,
                 opponent_policies: Optional[List[Callable]] = None, seed=None,
                 reward_scheme: str = "default", reward_discount: float = 1.0):
        self.num_players = num_players
        self.num_decks = num_decks
        self.opponent_policies = opponent_policies
        self.agent_seat = 0
        self.seed = seed
        self.reward_scheme = reward_scheme
        if not 0 < reward_discount <= 1:
            raise ValueError("reward_discount must be in (0, 1]")
        self.reward_discount = reward_discount
        self.rng = np.random.default_rng(seed)
        if reward_scheme not in ("default", "basic", "win"):
            raise ValueError("unknown reward scheme")
        self._default_opponent = GreedyBot(num_players=self.num_players,
                                           num_decks=self.num_decks, seed=seed)
        self.game: Optional[Game] = None
        self.last_trick_winners = []

    @property
    def done(self):
        # First place is decided as soon as any player empties their hand.
        return self.game is not None and (self.game.done or
            (self.reward_scheme == "win" and bool(self.game.finish_order)))

    def _potential(self):
        # With the learner's matching discount, potential shaping telescopes to
        # a fixed offset in discounted return, including terminal losses.
        return -len(self.game.hands[self.agent_seat]) / self.game.initial_hand_size

    def _opponent_policy(self, seat: int) -> Callable:
        if self.opponent_policies is not None and seat - 1 < len(self.opponent_policies):
            pol = self.opponent_policies[seat - 1]
            if pol is not None:
                return pol
        return self._default_opponent.act

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            self.seed = seed
            self.rng = np.random.default_rng(seed)
        starting = int(self.rng.integers(0, self.num_players))
        deal_seed = int(self.rng.integers(0, 2**63))
        self.game = Game(num_players=self.num_players, num_decks=self.num_decks,
                         starting_player=starting, seed=deal_seed)
        self.last_trick_winners = []
        return self._roll_to_agent()

    def _roll_to_agent(self):
        while not self.done and self.game.current_player != self.agent_seat:
            seat = self.game.current_player
            legal = self.game.legal_moves(seat)
            obs = features.state_vector(self.game, seat)
            move = self._opponent_policy(seat)(obs, legal)
            if move not in legal:
                move = legal[0]
            self.game.apply_move(seat, move)
            if self.game.last_trick_winner is not None:
                self.last_trick_winners.append(self.game.last_trick_winner)
        if self.done:
            return None
        return features.state_vector(self.game, self.agent_seat)

    def get_legal_moves(self) -> List[Move]:
        if self.done:
            return []
        return self.game.legal_moves(self.agent_seat)

    def step(self, action: Move):
        if self.game is None or self.done:
            raise RuntimeError("reset the environment before stepping")
        seat = self.agent_seat
        potential = self._potential() if self.reward_scheme == "win" else 0.0
        self.last_trick_winners = []
        self.game.apply_move(seat, action)
        if self.game.last_trick_winner is not None:
            self.last_trick_winners.append(self.game.last_trick_winner)
        state = self._roll_to_agent()

        card_reward = 0.1 * len(action.cards)

        if self.reward_scheme == "win":
            card_reward = (0.0 if self.done else self.reward_discount * self._potential()) - potential

        if self.done:
            return None, card_reward + self._terminal_reward(seat), True, \
                {"next_legal_moves": [], "trick_winners": self.last_trick_winners.copy()}

        legal = self.game.legal_moves(seat)
        return state, card_reward, False, {
            "next_legal_moves": legal, "trick_winners": self.last_trick_winners.copy()}

    def _terminal_reward(self, seat: int) -> float:
        if self.reward_scheme == "win":
            return 1.0 if self.game.finish_order[0] == seat else -1.0
        try:
            rank = self.game.finish_order.index(seat) + 1
        except ValueError:
            rank = self.num_players
        if self.reward_scheme == "basic":
            # 1.00 for winning (1st), 0.00 for every other placement.
            return 1.0 if rank == 1 else 0.0
        if self.num_players == 4:
            return {1: 1.0, 2: 0.3, 3: -0.3, 4: -1.0}.get(rank, -1.0)
        return 1.0 - 2.0 * (rank - 1) / (self.num_players - 1)
