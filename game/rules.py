import random
from collections import Counter
from typing import List, Optional

from .cards import deal_hands
from .moves import Move, generate_moves, beats, PASS_MOVE


class Game:
    def __init__(self, num_players: int = 4, num_decks: int = 2,
                 hands=None, starting_player: int = 0, rng=None, seed=None):
        self.num_players = num_players
        self.num_decks = num_decks
        self.rng = rng or random.Random(seed)
        if num_players < 2 or num_decks < 1:
            raise ValueError("at least two players and one deck are required")

        if hands is not None:
            if len(hands) != num_players or any(not h for h in hands):
                raise ValueError("provide one nonempty hand per player")
            self.hands = [list(h) for h in hands]
        else:
            self.hands = deal_hands(num_players, num_decks, self.rng)

        self.initial_hand_size = max(map(len, self.hands))
        self.current_player = starting_player % num_players
        self.table_move: Optional[Move] = None
        self.table_owner: int = -1
        self.passes_in_a_row = 0
        self.finish_order: List[int] = []
        self.done = False
        # Most recently resolved trick, consumed by the environment after each turn.
        self.last_trick_winner = None

    def _is_active(self, seat: int) -> bool:
        return len(self.hands[seat]) > 0

    @property
    def num_active(self) -> int:
        return sum(1 for h in self.hands if len(h) > 0)

    def legal_moves(self, seat: Optional[int] = None) -> List[Move]:
        if seat is None:
            seat = self.current_player
        if self.done or not self._is_active(seat):
            return []
        all_moves = generate_moves(self.hands[seat])
        if self.table_move is None:
            return all_moves
        beating = [m for m in all_moves if beats(m, self.table_move)]
        return beating + [PASS_MOVE]

    def _next_active(self, seat: int) -> int:
        n = self.num_players
        for step in range(1, n + 1):
            nxt = (seat + step) % n
            if self._is_active(nxt):
                return nxt
        return seat

    def apply_move(self, seat: int, move: Move) -> None:
        if self.done:
            raise RuntimeError("game is over")
        if seat != self.current_player:
            raise RuntimeError(f"not player {seat}'s turn (it is {self.current_player})")

        self._validate_move(seat, move)
        self.last_trick_winner = None
        if move.is_pass:
            self._apply_pass(seat)
            return

        remaining = list(self.hands[seat])
        for card in move.cards:
            remaining.remove(card)
        self.hands[seat] = remaining
        self.table_move = move
        self.table_owner = seat
        self.passes_in_a_row = 0

        if len(self.hands[seat]) == 0:
            self.finish_order.append(seat)

        if self.num_active <= 1:
            self._finish_game()
            return

        self.current_player = self._next_active(seat)

    def _validate_move(self, seat: int, move: Move) -> None:
        if move.is_pass:
            if move != PASS_MOVE or self.table_move is None:
                raise ValueError("cannot pass when leading or submit a malformed pass")
            return
        cards = Counter(move.cards)
        if not cards or cards - Counter(self.hands[seat]):
            raise ValueError("move contains cards not in the player's hand")
        # Validate the selected cards, not every combination of the whole hand.
        if not any(m.type == move.type and m.rank == move.rank
                   and m.length == move.length and m.is_bomb == move.is_bomb
                   and Counter(m.cards) == cards
                   for m in generate_moves(list(move.cards))):
            raise ValueError("invalid card combination or move metadata")
        if self.table_move is not None and not beats(move, self.table_move):
            raise ValueError("move does not beat the table")

    def _apply_pass(self, seat: int) -> None:
        self.passes_in_a_row += 1
        owner_active = self._is_active(self.table_owner)
        others = self.num_active - (1 if owner_active else 0)
        if others <= 0 or self.passes_in_a_row >= others:
            self.last_trick_winner = self.table_owner
            self.table_move = None
            self.passes_in_a_row = 0
            if owner_active:
                self.current_player = self.table_owner
            else:
                self.current_player = self._next_active(self.table_owner)
        else:
            self.current_player = self._next_active(seat)

    def _finish_game(self) -> None:
        self.done = True
        for s in range(self.num_players):
            if s not in self.finish_order:
                self.finish_order.append(s)
