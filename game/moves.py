from dataclasses import dataclass
from enum import IntEnum
from itertools import combinations
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
    AIRPLANE = 7
    CONSEC_PAIRS = 8      # 连对: 3+ consecutive pairs (no wings)
    CONSEC_TRIPLES = 9    # 连飞机: 2+ consecutive triples (body, no wings)

STRAIGHT_MIN = 3
STRAIGHT_MAX = 14
MIN_STRAIGHT_LEN = 5


@dataclass(frozen=True)
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


def _consecutive_runs(ranks: List[int], min_len: int) -> List[List[int]]:
    """Yield every contiguous subsequence of length >= min_len from the sorted,
    unique ``ranks`` list (e.g. [3,4,5,7] with min_len=2 -> [[3,4],[4,5],[3,4,5]])."""
    runs: List[List[int]] = []
    if not ranks:
        return runs
    segs: List[List[int]] = []
    cur = [ranks[0]]
    for r in ranks[1:]:
        if r == cur[-1] + 1:
            cur.append(r)
        else:
            segs.append(cur)
            cur = [r]
    segs.append(cur)
    for seg in segs:
        L = len(seg)
        for length in range(min_len, L + 1):
            for start in range(0, L - length + 1):
                runs.append(seg[start:start + length])
    return runs


def generate_moves(hand: List[Card]) -> List[Move]:
    moves: List[Move] = []
    groups = _by_rank(hand)
    ranks = sorted(groups)

    for r in ranks:
        moves.append(Move(MoveType.SINGLE, (groups[r][0],), r, 1))
    for r in ranks:
        if r <= 15 and len(groups[r]) >= 2:
            moves.append(Move(MoveType.PAIR, tuple(groups[r][:2]), r, 2))
    for r in ranks:
        if r <= 15 and len(groups[r]) >= 3:
            moves.append(Move(MoveType.TRIPLE, tuple(groups[r][:3]), r, 3))
    for r in ranks:
        if r <= 15 and len(groups[r]) >= 4:
            moves.append(Move(MoveType.BOMB, tuple(groups[r][:4]), r, 4, is_bomb=True))

    triples = [r for r in ranks if r <= 15 and len(groups[r]) >= 3]
    pairs = [r for r in ranks if r <= 15 and len(groups[r]) >= 2]
    for t in triples:
        for p in pairs:
            if p == t:
                continue
            moves.append(Move(
                MoveType.FULL_HOUSE,
                tuple(groups[t][:3] + groups[p][:2]),
                t, 5,
            ))

    triple_ranks = [r for r in triples if r <= 14]
    pair_ranks = [r for r in pairs if r <= 15]
    for r1, r2 in combinations(triples, 2):
        body = tuple(groups[r1][:3] + groups[r2][:3])
        wing_pool = [r for r in pair_ranks if r not in (r1, r2)]
        for p1, p2 in combinations(wing_pool, 2):
            cards = body + tuple(groups[p1][:2] + groups[p2][:2])
            moves.append(Move(
                MoveType.AIRPLANE,
                cards,
                max(r1, r2), 10,
            ))

    # 连飞机 (airplane body): 2+ consecutive triples, no wings.
    for run in _consecutive_runs(sorted(triple_ranks), 2):
        body = tuple(c for r in run for c in groups[r][:3])
        moves.append(Move(
            MoveType.CONSEC_TRIPLES, body, max(run), len(run)))

    # 连对 (consecutive pairs): 3+ consecutive pairs, no wings.
    consec_pair_ranks = [r for r in pairs if r <= 14]
    for run in _consecutive_runs(sorted(consec_pair_ranks), 3):
        body = tuple(c for r in run for c in groups[r][:2])
        moves.append(Move(
            MoveType.CONSEC_PAIRS, body, max(run), len(run)))

    straight_ranks = [r for r in ranks if STRAIGHT_MIN <= r <= STRAIGHT_MAX]
    for run in _consecutive_runs(straight_ranks, MIN_STRAIGHT_LEN):
        cards = tuple(groups[r][0] for r in run)
        moves.append(Move(MoveType.STRAIGHT, cards, run[-1], len(run)))

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
