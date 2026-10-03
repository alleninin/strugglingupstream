from dataclasses import dataclass
from typing import List

NORMAL_RANKS = list(range(3, 16))
SUITS = list(range(4))

RANK_LABELS = {
    3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8", 9: "9", 10: "10",
    11: "J", 12: "Q", 13: "K", 14: "A", 15: "2", 16: "BJ", 17: "RJ",
}


@dataclass(frozen=True)
class Card:
    id: int
    rank: int
    suit: int
    is_joker: bool = False

    @property
    def label(self) -> str:
        return RANK_LABELS[self.rank]

    def __str__(self) -> str:
        return self.label

    def __repr__(self) -> str:
        return self.label


def build_deck(num_decks: int = 1) -> List[Card]:
    if num_decks < 1:
        raise ValueError("num_decks must be positive")
    cards: List[Card] = []
    cid = 0
    for _ in range(num_decks):
        for rank in NORMAL_RANKS:
            for suit in SUITS:
                cards.append(Card(id=cid, rank=rank, suit=suit, is_joker=False))
                cid += 1
        cards.append(Card(id=cid, rank=16, suit=0, is_joker=True))
        cid += 1
        cards.append(Card(id=cid, rank=17, suit=1, is_joker=True))
        cid += 1
    return cards


def deal_hands(num_players: int, num_decks: int, rng) -> List[List[Card]]:
    """Shuffle a fresh deck and distribute every card, including any remainder."""
    deck = build_deck(num_decks)
    if not 2 <= num_players <= len(deck):
        raise ValueError("num_players must be between 2 and the number of cards")
    rng.shuffle(deck)
    base, extra = divmod(len(deck), num_players)
    hands = []
    start = 0
    for seat in range(num_players):
        end = start + base + (seat < extra)
        hands.append(deck[start:end])
        start = end
    return hands
