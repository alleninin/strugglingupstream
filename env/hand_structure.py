"""Cached partitions of a player's own hand; never consult opponents' cards."""
from functools import lru_cache

from bots.greedy_bot import _partition, _exact_turns


@lru_cache(maxsize=65536)
def hand_structure(counts):
    """Return estimated plays and leftover singles in a disjoint partition.

    Large hands use the better of two legal partitions. Small hands use exact
    minimum plays; the single count remains a heuristic partition statistic.
    """
    candidates = [_partition(counts, strict_tractor=strict) for strict in (False, True)]
    turns, groups = min(candidates, key=lambda item: (item[0], sum(
        group.type == 'SINGLE' for group in item[1])))
    if sum(counts) <= 8:
        turns = _exact_turns(counts)
    singles = sum(group.type == 'SINGLE' for group in groups)
    return turns, singles
