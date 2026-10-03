"""Combination-aware, deterministic opponent using only public observations.

Compare the remaining hand after legal plays rather than leading low singles by
rule. Small hands use exact minimum-combination search; larger hands use cached
partition estimates. Contest with intact combinations and high singles; allow
limited restructuring or a bomb when falling behind. This is a heuristic card-game policy,
not a search over opponents' hidden hands.
"""

import os
import sys

# Allow running this file standalone (the training scripts also prepend ROOT).
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
from typing import List, Optional, Tuple

import numpy as np

from game.cards import Card
from game.moves import Move, MoveType, generate_moves


# --------------------------------------------------------------------------- #
# Rank bounds (mirror game/cards.py + game/moves.py)
# --------------------------------------------------------------------------- #
MIN_RANK = 3
MAX_CARD_RANK = 15           # "2" is the highest normal rank
MAX_STRAIGHT_RANK = 14       # straights / tractors / consec triples exclude 2 & jokers
JOKER_LO = 16
JOKER_HI = 17


# --------------------------------------------------------------------------- #
# Hand partition
# --------------------------------------------------------------------------- #
@dataclass
class Group:
    """One combo group in a hand partition."""
    type: str                 # SINGLE / PAIR / TRIPLE / FULL_HOUSE / STRAIGHT /
                              # CONSEC_PAIRS / CONSEC_TRIPLES / AIRPLANE / BOMB
    rank: int                 # top rank of the group
    length: int              # type-dependent size (pairs count, run length, ...)
    ranks: List[int]          # the ranks that make up the group
    dangerous: bool           # bombs / 2s / jokers -- do not lead unless forced

    @property
    def n_cards(self) -> int:
        if self.type == "CONSEC_TRIPLES":
            return 3 * self.length
        if self.type == "CONSEC_PAIRS":
            return 2 * self.length
        return self.length    # single 1, pair 2, triple 3, full house 5,
                              # straight length, bomb 4


def _max_consec_run(c: dict, need: int, lo: int, hi: int,
                    min_len: int = 2, strict: bool = False) -> Optional[List[int]]:
    """Longest contiguous run of ranks r in [lo, hi] with c[r] >= need (or == need
    when ``strict``) and run length >= min_len.  Longest-first maximises packing."""
    best: Optional[List[int]] = None
    i = lo
    while i <= hi:
        ok = (c.get(i, 0) == need) if strict else (c.get(i, 0) >= need)
        if ok:
            j = i
            while j + 1 <= hi:
                ok2 = (c.get(j + 1, 0) == need) if strict else (c.get(j + 1, 0) >= need)
                if ok2:
                    j += 1
                else:
                    break
            run = list(range(i, j + 1))
            if len(run) >= min_len and (best is None or len(run) > len(best)):
                best = run
            i = j + 1
        else:
            i += 1
    return best


def _partition(hand_hist, strict_tractor: bool) -> Tuple[int, List[Group]]:
    """Greedily partition a rank-count vector (length 15, ranks 3..17) into the
    legal combos. This is a heuristic, not an exact minimum. ``strict_tractor`` reserves count>=3 ranks
    for triples / full houses by extracting tractors only from ranks with
    exactly 2 cards (used for the *strategy* partition).  The near-optimal M
    solver uses strict_tractor=False."""
    c = {r: int(round(hand_hist[r - MIN_RANK])) for r in range(MIN_RANK, JOKER_HI + 1)}
    groups: List[Group] = []

    while True:
        # 1. bombs: 4 of a kind (ranks 3..15; a "2" can bomb too)
        bomb = False
        for r in range(MIN_RANK, MAX_CARD_RANK + 1):
            if c[r] >= 4:
                c[r] -= 4
                groups.append(Group("BOMB", r, 4, [r], dangerous=True))
                bomb = True
                break
        if bomb:
            continue

        # A winged airplane can shed two triples and two distinct pairs at once.
        triples = [r for r in range(MIN_RANK, MAX_CARD_RANK + 1) if c[r] >= 3]
        airplane = None
        for body in combinations(triples, 2):
            wings = [r for r in range(MIN_RANK, MAX_CARD_RANK + 1)
                     if c[r] >= 2 and r not in body]
            if len(wings) >= 2:
                airplane = (*body, *wings[:2])
                break
        if airplane is not None:
            for r, count in zip(airplane, (3, 3, 2, 2)):
                c[r] -= count
            groups.append(Group("AIRPLANE", max(airplane[:2]), 10, list(airplane),
                                dangerous=MAX_CARD_RANK in airplane))
            continue

        # 2. consecutive triples (airplane body), 2+ in a row, ranks 3..14
        run = _max_consec_run(c, 3, MIN_RANK, MAX_STRAIGHT_RANK, min_len=2)
        if run:
            for r in run:
                c[r] -= 3
            groups.append(Group("CONSEC_TRIPLES", max(run), len(run), run,
                                 dangerous=False))
            continue

        # 3. tractors (consecutive pairs), 3+ in a row, ranks 3..14.
        #    strict_tractor keeps count>=3 ranks available for triples/full houses.
        run = _max_consec_run(c, 2, MIN_RANK, MAX_STRAIGHT_RANK,
                              min_len=3, strict=strict_tractor)
        if run:
            for r in run:
                c[r] -= 2
            groups.append(Group("CONSEC_PAIRS", max(run), len(run), run,
                                 dangerous=False))
            continue

        # 4. straights (5+ consecutive singles), ranks 3..14
        run = _max_consec_run(c, 1, MIN_RANK, MAX_STRAIGHT_RANK, min_len=5)
        if run:
            for r in run:
                c[r] -= 1
            groups.append(Group("STRAIGHT", max(run), len(run), run,
                                 dangerous=any(r >= MAX_CARD_RANK for r in run)))
            continue

        # 5. full house: triple + (different) pair
        tr = next((r for r in range(MIN_RANK, MAX_CARD_RANK + 1) if c[r] >= 3), None)
        if tr is not None:
            pr = next((r for r in range(MIN_RANK, MAX_CARD_RANK + 1)
                       if c[r] >= 2 and r != tr), None)
            if pr is not None:
                c[tr] -= 3
                c[pr] -= 2
                groups.append(Group("FULL_HOUSE", tr, 5, [tr, pr],
                                     dangerous=tr >= MAX_CARD_RANK))
                continue

        # 6. standalone triple
        tr = next((r for r in range(MIN_RANK, MAX_CARD_RANK + 1) if c[r] >= 3), None)
        if tr is not None:
            c[tr] -= 3
            groups.append(Group("TRIPLE", tr, 3, [tr],
                                 dangerous=tr >= MAX_CARD_RANK))
            continue

        # 7. standalone pair (ranks 3..15)
        pr = next((r for r in range(MIN_RANK, MAX_CARD_RANK + 1) if c[r] >= 2), None)
        if pr is not None:
            c[pr] -= 2
            groups.append(Group("PAIR", pr, 2, [pr],
                                 dangerous=pr >= MAX_CARD_RANK))
            continue

        # 8. single (incl. jokers and 2)
        sr = next((r for r in range(MIN_RANK, JOKER_HI + 1) if c[r] >= 1), None)
        if sr is not None:
            c[sr] -= 1
            groups.append(Group("SINGLE", sr, 1, [sr],
                                 dangerous=sr >= MAX_CARD_RANK))
            continue

        break

    return len(groups), groups


def partition_strategy(hand_hist) -> List[Group]:
    """Strategy partition: tractors reserve count>=3 ranks (spec's reserve rule)."""
    return _partition(hand_hist, strict_tractor=True)[1]


@lru_cache(maxsize=8192)
def _estimated_moves(counts: tuple) -> int:
    return _partition(counts, strict_tractor=False)[0]


def M(hand_hist) -> int:
    """Greedy estimate of the number of legal combos needed to clear the hand."""
    return _estimated_moves(tuple(int(round(c)) for c in hand_hist))


def O(hand_hist) -> List[Group]:
    """The greedy partition used by the move-count estimate."""
    return _partition(hand_hist, strict_tractor=False)[1]


@lru_cache(maxsize=65536)
def _exact_turns(counts: tuple) -> int:
    """Minimum legal plays to empty a small hand, ignoring opponents.

    Every partition contains a combo using the lowest remaining rank. Branching
    only on such combos avoids exploring every ordering of the same partition.
    All combinations come from the rules engine, including winged airplanes.
    """
    if not any(counts):
        return 0
    pivot = next(i for i, count in enumerate(counts) if count)
    hand = [Card(id=i * 32 + j, rank=i + MIN_RANK, suit=j % 4,
                 is_joker=i + MIN_RANK >= JOKER_LO)
            for i, count in enumerate(counts) for j in range(count)]
    moves = sorted(generate_moves(hand), key=lambda move: -len(move.cards))
    best = sum(counts)
    for move in moves:
        if not any(card.rank == pivot + MIN_RANK for card in move.cards):
            continue
        remaining = list(counts)
        for card in move.cards:
            remaining[card.rank - MIN_RANK] -= 1
        if not any(remaining):
            return 1
        best = min(best, 1 + _exact_turns(tuple(remaining)))
        if best == 2:
            break  # A one-play finish was already considered (largest moves first).
    return best


@lru_cache(maxsize=16384)
def _large_hand_turns(counts: tuple) -> int:
    return min(_partition(counts, strict_tractor=False)[0],
               _partition(counts, strict_tractor=True)[0])


class GreedyBot:
    """Combination-aware bot; no opponent cards or game object are consulted."""

    def __init__(self, state_dim=0, action_dim=0, num_players=4, num_decks=2,
                 block_threshold=5, endgame_threshold=12, seed=0, **kwargs):
        self.num_players = num_players
        self.num_decks = num_decks
        self.block_threshold = block_threshold
        self.endgame_threshold = endgame_threshold
        self.initial_hand_size = (num_decks * 54) // num_players
        self.epsilon = 0.0

    def reset_episode(self):
        pass

    @staticmethod
    def partition_strategy(hand_hist):
        return partition_strategy(hand_hist)

    @staticmethod
    def _after_hist(hand_hist, move):
        after = list(hand_hist)
        for card in move.cards:
            after[card.rank - MIN_RANK] -= 1
        return tuple(after)

    def _turns(self, counts):
        if sum(counts) <= self.endgame_threshold:
            return _exact_turns(tuple(counts))
        return _large_hand_turns(tuple(counts))

    @staticmethod
    def _control_cost(move):
        # Includes high pair wings, which move.rank alone does not describe.
        return 4 * move.is_bomb + sum(max(0, card.rank - 13) for card in move.cards)

    @staticmethod
    def _broken_bombs(before, after):
        return sum(old // 4 > new // 4 and old - new < 4
                   for old, new in zip(before[:13], after[:13]))

    def _lead(self, legal, hand_hist, opp_sizes=()):
        active = [size for size in opp_sizes if size > 0]
        one_card_threat = bool(active) and min(active) == 1

        def score(move):
            after = self._after_hist(hand_hist, move)
            turns = self._turns(after)
            # Do not offer a low single to someone holding their last card.
            exposed_single = one_card_threat and move.type == MoveType.SINGLE and move.rank < 17
            return (exposed_single, turns, self._broken_bombs(hand_hist, after),
                    self._control_cost(move), -len(move.cards), move.rank)

        return min((move for move in legal if not move.is_pass), key=score)

    def _follow(self, legal, table, opp_sizes, hand_hist):
        candidates = [move for move in legal if not move.is_pass]
        passing = next((move for move in legal if move.is_pass), None)
        if not candidates:
            return passing
        whole_hand = [move for move in candidates if len(move.cards) == sum(hand_hist)]
        if whole_hand:
            return whole_hand[0]

        before_turns = self._turns(hand_hist)
        scored = []
        for move in candidates:
            after = self._after_hist(hand_hist, move)
            scored.append((move, self._turns(after), self._broken_bombs(hand_hist, after)))
        active = [size for size in opp_sizes if size > 0]
        # Five cards is too late: eight cards could already be two bombs. Also
        # respond when an opponent pulls ahead, rather than waiting for an endgame.
        danger_size = max(self.block_threshold, (self.initial_hand_size + 2) // 3)
        lead_margin = max(3, self.initial_hand_size // 5)
        under_pressure = bool(active) and (
            min(active) <= danger_size or sum(hand_hist) - min(active) >= lead_margin)
        if table is not None:
            table_type, _, table_length = table
            multiplier = {MoveType.CONSEC_PAIRS: 2, MoveType.CONSEC_TRIPLES: 3}.get(table_type, 1)
            response_size = table_length * multiplier
        else:
            response_size = next((len(move.cards) for move in candidates if not move.is_bomb), 1)
        immediate_threat = bool(active) and min(active) <= response_size
        if immediate_threat:
            # Direct beats conserve bombs. Within the same response type, use
            # strength to deny a near-finished opponent an easy overcall.
            return min(scored, key=lambda item: (
                item[0].is_bomb, -item[0].rank, item[1],
                self._control_cost(item[0])))[0]

        # High cards are useful for taking the lead. Their cost is a tie-breaker,
        # not a veto: spending an intact ace/2/joker does not break a combination.
        intact = [item for item in scored
                  if not item[0].is_bomb and item[1] < before_turns and not item[2]]
        if intact:
            return min(intact, key=lambda item: (
                item[1], self._control_cost(item[0]), -len(item[0].cards), item[0].rank))[0]

        if under_pressure:
            # Paying at most one extra future play can be worthwhile to stop an
            # opponent's run. Whole bombs are available here, but don't dismantle
            # a bomb into singles/pairs unless there is an immediate finish threat.
            contest = [item for item in scored if not item[2] and item[1] <= before_turns + 1]
            if contest:
                return min(contest, key=lambda item: (
                    item[1], item[0].is_bomb, self._control_cost(item[0]),
                    -len(item[0].cards), item[0].rank))[0]

        # A bomb can also buy the lead for a short finishing sequence of our own.
        closing = [item for item in scored if not item[2] and item[1] < before_turns and item[1] <= 2]
        if closing:
            return min(closing, key=lambda item: (
                item[1], self._control_cost(item[0]), item[0].rank))[0]
        return passing if passing is not None else self._lead(candidates, hand_hist, opp_sizes)

    def act(self, obs, legal):
        if not legal:
            return None
        hand = tuple(int(round(value)) for value in obs[:15])
        own_size = sum(hand)
        # Winning now always takes precedence over conservation and blocking.
        finishing = [move for move in legal if not move.is_pass and len(move.cards) == own_size]
        if finishing:
            return finishing[0]
        scale = own_size / float(obs[28]) if obs[28] > 0 else self.initial_hand_size
        opponents = [int(round(float(value) * scale))
                     for value in obs[29:29 + self.num_players - 1]]
        if float(obs[27]) < .5:
            return self._lead(legal, hand, opponents)
        table = (MoveType(int(np.argmax(obs[15:25]))),
                 int(round(float(obs[25]) * 14)) + MIN_RANK,
                 int(round(float(obs[26]) * 15)))
        return self._follow(legal, table, opponents, hand)


# --------------------------------------------------------------------------- #
# Self test
# --------------------------------------------------------------------------- #
def _self_test():
    # Critical edge case from the spec:
    # hand 6,7,7,7,8,8,9,9,10,10  -> triple 777 + tractor 88991010 + orphan 6
    hand = [0] * 15
    for r, k in [(6, 1), (7, 3), (8, 2), (9, 2), (10, 2)]:
        hand[r - MIN_RANK] = k

    assert M(hand) == 3, f"expected M=3, got {M(hand)}"

    groups = partition_strategy(hand)
    assert any(g.type == "TRIPLE" and g.rank == 7 for g in groups), groups
    assert any(g.type == "CONSEC_PAIRS" and g.ranks == [8, 9, 10]
               for g in groups), groups
    orphans = [g for g in groups if g.type == "SINGLE"]
    assert len(orphans) == 1 and orphans[0].rank == 6, groups

    print("OK  greedy-bot self-test (M=3, triple 777, tractor 88991010, orphan 6)")


if __name__ == "__main__":
    _self_test()
