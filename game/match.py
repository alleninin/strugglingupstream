import random
from typing import List, Optional

from .cards import Card, deal_hands
from .rules import Game


class Match:
    def __init__(self, num_players: int = 4, num_decks: int = 2,
                 rng=None, seed=None):
        self.num_players = num_players
        self.num_decks = num_decks
        self.rng = rng or random.Random(seed)
        self.hands: Optional[List[List[Card]]] = None
        self.last_transfers = []

    def next_deal(self, prev_finish_order: Optional[List[int]] = None) -> List[List[Card]]:
        if prev_finish_order is not None and sorted(prev_finish_order) != list(range(self.num_players)):
            raise ValueError("finish order must contain every seat exactly once")
        self.hands = deal_hands(self.num_players, self.num_decks, self.rng)
        self.last_transfers = []
        if prev_finish_order is not None:
            self._apply_penalty(prev_finish_order)
        return self.hands

    def _apply_penalty(self, finish_order: List[int]) -> None:
        count = min(2, self.num_players // 2)
        losers = finish_order[-count:]
        winners = finish_order[:count]
        transfers = []
        for i, loser in enumerate(reversed(losers)):
            winner = winners[i]
            if not self.hands[loser]:
                continue
            top = max(self.hands[loser], key=lambda c: c.rank)
            self.hands[loser].remove(top)
            self.hands[winner].append(top)
            transfers.append((loser, winner, top))
        self.last_transfers = transfers

    def play_match(self, agents, num_deals: int, starting_player: int = 0):
        from env import features
        finish_orders = []
        prev = None
        for _ in range(num_deals):
            hands = self.next_deal(prev)
            g = Game(num_players=self.num_players, num_decks=self.num_decks,
                     hands=hands, starting_player=starting_player, rng=self.rng)
            while not g.done:
                seat = g.current_player
                legal = g.legal_moves(seat)
                obs = features.state_vector(g, seat)
                move = agents[seat].act(obs, legal)
                if move not in legal:
                    move = legal[0]
                g.apply_move(seat, move)
            prev = g.finish_order
            finish_orders.append(prev)
        return finish_orders
