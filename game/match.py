"""Multi-deal match with the "Struggling Upstream" penalty.

After each deal the bottom-2 finishers must hand their single highest card
(face-up) to the top-2 winners for the next deal:

  * last place       -> 1st place
  * next-to-last     -> 2nd place

(With 3 players there is only one winner, so both losers give to 1st place.)
"""
import random
from typing import List, Optional

from .cards import Card, build_deck
from .rules import Game


class Match:
    def __init__(self, num_players: int = 4, num_decks: int = 1,
                 rng=None, seed=None):
        self.num_players = num_players
        self.num_decks = num_decks
        self.rng = rng or random.Random(seed)
        self.hands: Optional[List[List[Card]]] = None
        self.last_transfers = []

    def next_deal(self, prev_finish_order: Optional[List[int]] = None) -> List[List[Card]]:
        if prev_finish_order is None or self.hands is None:
            deck = build_deck(self.num_decks)
            self.rng.shuffle(deck)
            n = self.num_players
            base = len(deck) // n
            self.hands = [deck[i * base:(i + 1) * base] for i in range(n)]
            self.last_transfers = []
            return self.hands
        self._apply_penalty(prev_finish_order)
        return self.hands

    def _apply_penalty(self, finish_order: List[int]) -> None:
        """Bottom-two finishers hand their highest card to the top winners.

        last place       -> 1st place
        next-to-last     -> 2nd place (or 1st if there is no 2nd winner,
                            e.g. in a 3-player game where the bottom two are
                            2nd and 3rd and only 1st is a winner).
        """
        losers = finish_order[-2:]          # [next_to_last, last]
        winners = finish_order[:-2]         # everyone ranked above the bottom two
        transfers = []
        # reversed(losers) = [last, next_to_last]; pair worst loser with best winner.
        for i, loser in enumerate(reversed(losers)):
            winner = winners[i] if i < len(winners) else winners[0]
            if not self.hands[loser]:
                continue
            self.hands[loser].sort(key=lambda c: c.rank)
            top = self.hands[loser].pop()   # highest card
            self.hands[winner].append(top)
            transfers.append((loser, winner, top))
        self.last_transfers = transfers

    def play_match(self, agents, num_deals: int, starting_player: int = 0):
        """Play ``num_deals`` consecutive deals, applying penalties between them.

        ``agents`` is a list of agent objects (one per seat) exposing
        ``act(obs, legal_moves) -> Move``.
        Returns the list of finish orders, one per deal.
        """
        from env import features  # lazy import to avoid a package cycle
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
