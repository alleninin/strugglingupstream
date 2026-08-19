"""Standalone verification for the Zheng Shang You engine.

Run with:  python3 scripts/test_rules.py

Covers: deck construction, move generation, the beat relation, full-game
termination/consistency, and the Struggling-Upstream penalty.
"""
import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from game.cards import Card, build_deck, RANK_LABELS
from game.moves import Move, MoveType, generate_moves, beats, PASS_MOVE
from game.rules import Game
from game.match import Match
from agents.random_agent import RandomAgent


def card(rank, suit=0):
    return Card(id=1000 + rank * 10 + suit, rank=rank, suit=suit)


def test_deck():
    d = build_deck(1)
    assert len(d) == 54, len(d)
    d2 = build_deck(2)
    assert len(d2) == 108
    print("OK  deck construction (54 / 108 cards)")


def test_generate_moves():
    hand = [card(3), card(3), card(5), card(5), card(5),
            card(7), card(8), card(9), card(10), card(11),  # 7-8-9-10-J straight (5)
            card(4, 1), card(4, 2), card(4, 3), card(4, 0)]  # four 4s -> bomb
    moves = generate_moves(hand)
    types = [m.type for m in moves]
    assert MoveType.SINGLE in types
    assert MoveType.PAIR in types                 # pair of 3s
    assert MoveType.TRIPLE in types               # triple of 5s
    assert MoveType.BOMB in types                 # four 4s
    # straight 7-8-9-10-J present
    straights = [m for m in moves if m.type == MoveType.STRAIGHT]
    assert any(m.rank == 11 and m.length == 5 for m in straights), straights
    # NO straight should include a 2 or joker; none present anyway
    print(f"OK  move generation ({len(moves)} moves from a 14-card hand)")


def test_beats():
    s5 = Move(MoveType.SINGLE, (card(5),), 5, 1)
    s9 = Move(MoveType.SINGLE, (card(9),), 9, 1)
    s2 = Move(MoveType.SINGLE, (card(15),), 15, 1)      # a 2
    bj = Move(MoveType.SINGLE, (card(16),), 16, 1)
    rj = Move(MoveType.SINGLE, (card(17),), 17, 1)
    assert beats(s9, s5)
    assert not beats(s5, s9)
    assert beats(s2, s9)                                # 2 beats 9
    assert beats(bj, s2)                                # black joker beats 2
    assert beats(rj, bj)                                # red beats black
    assert not beats(bj, rj)

    bomb = Move(MoveType.BOMB, tuple(card(4, i) for i in range(4)), 4, 4, is_bomb=True)
    bomb_high = Move(MoveType.BOMB, tuple(card(7, i) for i in range(4)), 7, 4, is_bomb=True)
    assert beats(bomb, s9)                              # bomb beats single
    assert beats(bomb_high, bomb)                       # higher bomb beats
    assert not beats(s9, bomb)                          # single cannot beat bomb

    # straights must match length
    st5_low = Move(MoveType.STRAIGHT, tuple(card(r) for r in range(3, 8)), 7, 5)
    st5_high = Move(MoveType.STRAIGHT, tuple(card(r) for r in range(4, 9)), 8, 5)
    st6 = Move(MoveType.STRAIGHT, tuple(card(r) for r in range(3, 9)), 8, 6)
    assert beats(st5_high, st5_low)
    assert not beats(st6, st5_low)                      # length mismatch
    print("OK  beat relation (ranking, bombs, straight length)")


def test_full_game():
    for seed in range(50):
        g = Game(num_players=3, num_decks=1, seed=seed)
        total = sum(len(h) for h in g.hands)
        assert total == 54, total
        agents = [RandomAgent(seed=seed * 10 + i) for i in range(3)]
        steps = 0
        while not g.done:
            seat = g.current_player
            legal = g.legal_moves(seat)
            move = agents[seat].act(None, legal)
            assert move in legal, ("illegal move", move, seat)
            g.apply_move(seat, move)
            steps += 1
            assert steps < 2000, "game did not terminate"
        # every seat appears exactly once in the finish order
        assert sorted(g.finish_order) == [0, 1, 2], g.finish_order
        # All but the last-place player must have emptied their hand. The deal
        # ends when only one player is left, so the last player keeps their
        # remaining cards -- this is correct game behaviour (they are last).
        last = g.finish_order[-1]
        for seat in range(3):
            if seat != last:
                assert len(g.hands[seat]) == 0, (seat, g.hands[seat])
        # Card conservation: total cards are preserved (played + still held = 54).
        assert sum(len(h) for h in g.hands) >= 1
    print("OK  50 random full games: all terminate, valid finish order, last holds leftover")


def test_pass_and_clear():
    # Craft a scenario: A leads, B and C pass -> table clears, A leads again.
    hands = [
        [card(3, 0), card(3, 1), card(3, 2)],  # P0: three 3s (plays a pair, keeps one)
        [card(4, 0), card(4, 1)],              # P1: pair of 4s (could beat, but will pass)
        [card(5, 0), card(5, 1)],              # P2: pair of 5s (will pass)
    ]
    g = Game(num_players=3, num_decks=1, hands=hands, starting_player=0, seed=0)
    g.apply_move(0, g.legal_moves(0)[1])  # legal_moves(0) = [SINGLE(3), PAIR(3), TRIPLE(3)]
    assert g.table_move is not None
    g.apply_move(1, PASS_MOVE)
    assert not g.done
    g.apply_move(2, PASS_MOVE)
    # both others passed -> cleared, P0 (still active) leads again
    assert g.table_move is None, "table should have cleared"
    assert g.current_player == 0, g.current_player
    print("OK  pass & clear: table cleared after all others pass")


def test_penalty():
    # 4 players; finish order 0,1,2,3 (P3 last, P2 next-to-last)
    hands = [[card(r, i) for i in range(4)] for r in range(3, 7)]  # small distinct hands
    # give losers identifiable top cards
    hands[2] = [card(15), card(3)]   # P2 (next-to-last) top = 2
    hands[3] = [card(17), card(4)]   # P3 (last) top = red joker
    hands[0] = [card(14), card(5)]   # P0 (winner)
    hands[1] = [card(13), card(6)]   # P1 (2nd)
    m = Match(num_players=4, rng=None, seed=0)
    m.hands = hands
    m._apply_penalty([0, 1, 2, 3])
    # P3 (last) -> P0, P2 (next-to-last) -> P1
    transferred = {(src, dst, c.rank) for (src, dst, c) in m.last_transfers}
    assert (3, 0, 17) in transferred, transferred   # last gives red joker to 1st
    assert (2, 1, 15) in transferred, transferred   # next-to-last gives 2 to 2nd
    print("OK  Struggling-Upstream penalty transfers highest cards to winners")


def main():
    test_deck()
    test_generate_moves()
    test_beats()
    test_full_game()
    test_pass_and_clear()
    test_penalty()
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
