"""Card representation and deck construction for Zheng Shang You."""
from dataclasses import dataclass
from typing import List

# Rank numeric values (low -> high).
#   3..10 -> 3..10, J=11, Q=12, K=13, A=14, 2=15, Black Joker=16, Red Joker=17
NORMAL_RANKS = list(range(3, 16))          # 3..15
SUITS = list(range(4))                     # 0..3 (arbitrary suits for normal cards)

RANK_LABELS = {
    3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8", 9: "9", 10: "10",
    11: "J", 12: "Q", 13: "K", 14: "A", 15: "2", 16: "BJ", 17: "RJ",
}


@dataclass(frozen=True)
class Card:
    """A single card. Identity is the unique ``id``; suits are cosmetic except
    for distinguishing the two jokers (black vs red)."""

    id: int
    rank: int            # 3..17
    suit: int            # 0..3 normal; 0=black / 1=red for jokers
    is_joker: bool = False

    @property
    def label(self) -> str:
        return RANK_LABELS[self.rank]

    def __str__(self) -> str:
        return self.label

    def __repr__(self) -> str:
        return self.label


def build_deck(num_decks: int = 1) -> List[Card]:
    """Build ``num_decks`` standard 54-card decks (52 + 2 jokers each)."""
    cards: List[Card] = []
    cid = 0
    for _ in range(num_decks):
        for rank in NORMAL_RANKS:
            for suit in SUITS:
                cards.append(Card(id=cid, rank=rank, suit=suit, is_joker=False))
                cid += 1
        cards.append(Card(id=cid, rank=16, suit=0, is_joker=True))  # Black Joker
        cid += 1
        cards.append(Card(id=cid, rank=17, suit=1, is_joker=True))  # Red Joker
        cid += 1
    return cards
