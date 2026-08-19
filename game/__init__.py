"""Zheng Shang You (征上游) game engine.

Card ranking (low -> high): 3,4,5,6,7,8,9,10,J,Q,K,A,2,Black Joker,Red Joker.
Combinations: single, pair, triple, full house, straight (5+), four-of-a-kind bomb.
A play is beaten only by a higher combo of the same type/length, or by a bomb.
"""

from .cards import Card, build_deck, RANK_LABELS
from .moves import Move, MoveType, generate_moves, beats, PASS_MOVE
from .rules import Game
from .match import Match

__all__ = ["Card", "build_deck", "RANK_LABELS", "Move", "MoveType",
           "generate_moves", "beats", "PASS_MOVE", "Game", "Match"]
