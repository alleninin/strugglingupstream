from dataclasses import dataclass
from enum import IntEnum
from typing import List, Tuple

from .cards import Card

class MoveType(IntEnum):
    PASS = 0
    SINGLE = 1
    PAIR = 2
    TRIPLE = 3
    FULL_HOUSE = 4
    STRAIGHT = 5
    BOMB = 6

STRAIGHT_MIN = 3
STRAIGHT_MAX = 14
MIN_STRAIGHT_LEN = 5


@dataclass
class Move:
    type: MoveType
    cards: Tuple[Card, ...]
    rank: int
    length: int
    is_bomb: bool = False
    is_pass: bool = False

    def __str__(self) -> str:
        if self.is_pass:
            return "PASS"
        return f"{self.type.name}({','.join(c.label for c in self.cards)})"

    def __repr__(self) -> str:
        return self.__str__()


PASS_MOVE = Move(type=MoveType.PASS, cards=(), rank=0, length=0,
                 is_bomb=False, is_pass=True)


def _by_rank(hand: List[Card]):
    groups: dict = {}
    for c in hand:
        groups.setdefault(c.rank, []).append(c)
    return groups


def generate_moves(hand: List[Card]) -> List[Move]:
    moves: List[Move] = []
    groups = _by_rank(hand)
    ranks = sorted(groups)

    for r in ranks:
        moves.append(Move(MoveType.SINGLE, (groups[r][0],), r, 1))
    for r in ranks:
        if len(groups[r]) >= 2:
            moves.append(Move(MoveType.PAIR, tuple(groups[r][:2]), r, 2))
    for r in ranks:
        if len(groups[r]) >= 3:
            moves.append(Move(MoveType.TRIPLE, tuple(groups[r][:3]), r, 3))
    for r in ranks:
        if len(groups[r]) >= 4:
            moves.append(Move(MoveType.BOMB, tuple(groups[r][:4]), r, 4, is_bomb=True))

    triples = [r for r in ranks if len(groups[r]) >= 3]
    pairs = [r for r in ranks if len(groups[r]) >= 2]
    for t in triples:
        for p in pairs:
            if p == t:
                continue
            moves.append(Move(
                MoveType.FULL_HOUSE,
                tuple(groups[t][:3] + groups[p][:2]),
                t, 5,
            ))

    present = set(ranks)
    for start in range(STRAIGHT_MIN, STRAIGHT_MAX + 1):
        max_len = STRAIGHT_MAX - start + 1
        if max_len < MIN_STRAIGHT_LEN:
            break
        for length in range(MIN_STRAIGHT_LEN, max_len + 1):
            end = start + length - 1
            seq = list(range(start, end + 1))
            if all(r in present for r in seq):
                cards = tuple(groups[r][0] for r in seq)
                moves.append(Move(MoveType.STRAIGHT, cards, end, length))

    return moves


def beats(candidate: Move, current: Move) -> bool:
    if candidate.is_pass:
        return False
    if candidate.is_bomb and not current.is_bomb:
        return True
    if current.is_bomb and not candidate.is_bomb:
        return False
    if candidate.is_bomb and current.is_bomb:
        return candidate.rank > current.rank
    return (candidate.type == current.type
            and candidate.length == current.length
            and candidate.rank > current.rank)
