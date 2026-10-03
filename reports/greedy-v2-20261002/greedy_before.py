"""Deterministic rule-based Zheng Shang You bot.

A stable, non-learning opponent for self-play training (and for A/B comparison
against the RL agents).  The bot is a *pure* function of the observable game
state -- its own hand, the current trick, and the opponents' hand sizes -- so
it is fully reproducible and needs no checkpoint.

Strategy
--------
The hand is partitioned into combo ``Group`` s (singles / pairs / triples /
full houses / straights / tractors (consecutive pairs) / consecutive triples /
bombs).  A *reserve* rule keeps count>=3 ranks out of tractors so that, e.g.,
``{6,7,7,7,8,8,9,9,10,10}`` becomes ``triple 777 + tractor 88991010 + single 6``
rather than absorbing the 7s into a straight.

Leading: dump the lowest orphan single first, then the lowest-tier intact group.
Never lead 2s / bombs / jokers unless forced (only dangerous groups remain).

Following: contest cheap, safe beats (a spare single/pair that does not
fragment a larger combo) freely -- there is no cost to denying an opponent the
trick with a disposable card.  Only genuinely costly cards (2s / bombs /
jokers, or moves that fragment structure) are held back, and those are only
spent when an opponent is about to win (their hand size <= ``BLOCK_THRESHOLD``).

Endgame: once the bot's own hand is small (<= ``ENDGAME_THRESHOLD``) it switches
to a minimum-moves-to-empty policy, picking the legal move that leaves the
fewest combos needed to clear the rest of the hand.

Both thresholds are constructor parameters so the aggression of the opponent
can be tuned.
"""

import os
import sys

# Allow running this file standalone (the training scripts also prepend ROOT).
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from dataclasses import dataclass
from collections import Counter
from functools import lru_cache
from itertools import combinations
from typing import List, Optional, Tuple

import numpy as np

from game.moves import Move, MoveType


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


# Leading priority: groups we most want to dump first have the lowest value.
TIER = {
    "SINGLE": 0,
    "PAIR": 1,
    "TRIPLE": 2,
    "FULL_HOUSE": 2,
    "STRAIGHT": 3,
    "CONSEC_PAIRS": 4,
    "CONSEC_TRIPLES": 4,
    "AIRPLANE": 4,
    "BOMB": 5,
}

TYPE_TO_MOVETYPE = {
    "SINGLE": MoveType.SINGLE,
    "PAIR": MoveType.PAIR,
    "TRIPLE": MoveType.TRIPLE,
    "FULL_HOUSE": MoveType.FULL_HOUSE,
    "STRAIGHT": MoveType.STRAIGHT,
    "CONSEC_PAIRS": MoveType.CONSEC_PAIRS,
    "CONSEC_TRIPLES": MoveType.CONSEC_TRIPLES,
    "AIRPLANE": MoveType.AIRPLANE,
    "BOMB": MoveType.BOMB,
}


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


# --------------------------------------------------------------------------- #
# Bot
# --------------------------------------------------------------------------- #
class GreedyBot:
    """Deterministic, non-learning Zheng Shang You bot."""

    def __init__(self, state_dim: int = 0, action_dim: int = 0,
                 num_players: int = 4, num_decks: int = 2,
                 block_threshold: int = 5, endgame_threshold: int = 5,
                 seed: int = 0, **kwargs):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_players = num_players
        self.num_decks = num_decks
        self.block_threshold = block_threshold
        self.endgame_threshold = endgame_threshold
        self.seed = seed
        self.initial_hand_size = (num_decks * 54) // num_players
        self.epsilon = 0.0
        self.env = None

    # -- BaseAgent-compatible no-ops ---------------------------------------- #
    def set_env(self, env):
        self.env = env

    def reset_episode(self):
        pass

    def observe(self, transition):
        pass

    def save(self, path):
        pass

    def load(self, path):
        pass

    # -- helpers ------------------------------------------------------------- #
    @staticmethod
    def partition_strategy(hand_hist) -> List[Group]:
        """Strategy partition of a rank-count vector (see module docstring)."""
        return partition_strategy(hand_hist)

    @staticmethod
    def _pass_move(legal: List[Move]) -> Optional[Move]:
        for m in legal:
            if m.is_pass:
                return m
        return None

    @staticmethod
    def _match_move(legal: List[Move], mtype: MoveType,
                    rank: int, length: int, ranks=None) -> Optional[Move]:
        for m in legal:
            if m.is_pass:
                continue
            if m.type == mtype and m.rank == rank and m.length == length:
                if ranks is not None and mtype in (MoveType.FULL_HOUSE, MoveType.AIRPLANE):
                    counts = (3, 2) if mtype == MoveType.FULL_HOUSE else (3, 3, 2, 2)
                    if Counter(c.rank for c in m.cards) != dict(zip(ranks, counts)):
                        continue
                return m
        return None

    @staticmethod
    def _after_hist(hand_hist, move: Move) -> np.ndarray:
        after = np.array(hand_hist, dtype=float).copy()
        for cd in move.cards:
            after[cd.rank - MIN_RANK] -= 1
        return after

    def _is_safe(self, move: Move, hand_hist) -> bool:
        """A play is 'safe' if it lowers the estimated number of combos by one
        (i.e. it plays a complete group, not a fragment of a larger one)."""
        after = self._after_hist(hand_hist, move)
        return M(after) == M(hand_hist) - 1

    @staticmethod
    def _is_dangerous_move(move: Move) -> bool:
        """True for bombs and high cards (2s and up). Spending one of
        these is genuinely costly, so it should be threat-gated."""
        return move.type == MoveType.BOMB or move.rank >= MAX_CARD_RANK

    # -- decision logic ------------------------------------------------------ #
    def _endgame(self, legal: List[Move], hand_hist, is_leading: bool) -> Move:
        candidates = [m for m in legal if not m.is_pass]
        if not candidates:
            return self._pass_move(legal) or legal[0]
        best = min(candidates, key=lambda m: (
            M(self._after_hist(hand_hist, m)), m.rank, m.length))
        return best

    def _lead(self, legal: List[Move], hand_hist) -> Move:
        groups = partition_strategy(hand_hist)
        non_danger = [g for g in groups if not g.dangerous]
        if non_danger:
            g = min(non_danger, key=lambda g: (TIER[g.type], g.rank))
        else:
            g = min(groups, key=lambda g: g.rank)      # forced: only dangerous left
        mv = self._match_move(legal, TYPE_TO_MOVETYPE[g.type], g.rank, g.length, g.ranks)
        if mv is not None:
            return mv
        # fallback: lowest non-pass legal move
        non_pass = [m for m in legal if not m.is_pass]
        return min(non_pass, key=lambda m: (m.rank, m.length))

    def _follow(self, legal: List[Move], table, opp_sizes, hand_hist) -> Move:
        candidates = [m for m in legal if not m.is_pass]
        if not candidates:
            return self._pass_move(legal) or legal[0]

        # Tier 1: cheap, safe beats (non-dangerous cards that don't fragment a
        # larger combo). Contest these freely -- there is no cost to shedding a
        # spare single 5 over someone's single 4, and denying the trick is free.
        # This must NOT be gated by `threat`; otherwise the bot hoards even
        # disposable cards early and lets opponents build an uncontested lead.
        safe_beats = [m for m in candidates
                      if not self._is_dangerous_move(m) and self._is_safe(m, hand_hist)]
        if safe_beats:
            return min(safe_beats, key=lambda m: (m.rank, m.length))

        # Tier 2: only cards that would fragment structure, or genuinely costly
        # cards (2s / bombs / jokers). Spending these is expensive, so only do it
        # when an opponent is about to win (their hand is small).
        active_sizes = [size for size in opp_sizes if size > 0]
        threat = bool(active_sizes) and min(active_sizes) <= self.block_threshold
        if not threat:
            return self._pass_move(legal) or candidates[0]
        pool = [m for m in candidates if self._is_safe(m, hand_hist)] or candidates
        return min(pool, key=lambda m: (m.rank, m.length))

    def act(self, obs, legal: List[Move]) -> Move:
        hist = np.asarray(obs[:15], dtype=float)
        own_size = int(hist.sum())
        scale = own_size / float(obs[28]) if obs[28] > 0 else self.initial_hand_size
        opp_start = 29
        opp_sizes = [int(round(float(obs[opp_start + i]) * scale))
                     for i in range(self.num_players - 1)]
        present = float(obs[27]) >= 0.5

        if own_size <= self.endgame_threshold:
            return self._endgame(legal, hist, is_leading=not present)

        if not present:
            return self._lead(legal, hist)

        table_type = int(np.argmax(obs[15:25]))
        table_rank = int(round(float(obs[25]) * 14)) + MIN_RANK
        table_length = int(round(float(obs[26]) * 15))
        return self._follow(legal, (table_type, table_rank, table_length),
                            opp_sizes, hist)


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
