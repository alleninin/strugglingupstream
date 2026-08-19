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
            card(7), card(8), card(9), card(10), card(11),
            card(4, 1), card(4, 2), card(4, 3), card(4, 0)]
    moves = generate_moves(hand)
    types = [m.type for m in moves]
    assert MoveType.SINGLE in types
    assert MoveType.PAIR in types
    assert MoveType.TRIPLE in types
    assert MoveType.BOMB in types
    straights = [m for m in moves if m.type == MoveType.STRAIGHT]
    assert any(m.rank == 11 and m.length == 5 for m in straights), straights
    print(f"OK  move generation ({len(moves)} moves from a 14-card hand)")


def test_beats():
    s5 = Move(MoveType.SINGLE, (card(5),), 5, 1)
    s9 = Move(MoveType.SINGLE, (card(9),), 9, 1)
    s2 = Move(MoveType.SINGLE, (card(15),), 15, 1)
    bj = Move(MoveType.SINGLE, (card(16),), 16, 1)
    rj = Move(MoveType.SINGLE, (card(17),), 17, 1)
    assert beats(s9, s5)
    assert not beats(s5, s9)
    assert beats(s2, s9)
    assert beats(bj, s2)
    assert beats(rj, bj)
    assert not beats(bj, rj)

    bomb = Move(MoveType.BOMB, tuple(card(4, i) for i in range(4)), 4, 4, is_bomb=True)
    bomb_high = Move(MoveType.BOMB, tuple(card(7, i) for i in range(4)), 7, 4, is_bomb=True)
    assert beats(bomb, s9)
    assert beats(bomb_high, bomb)
    assert not beats(s9, bomb)

    st5_low = Move(MoveType.STRAIGHT, tuple(card(r) for r in range(3, 8)), 7, 5)
    st5_high = Move(MoveType.STRAIGHT, tuple(card(r) for r in range(4, 9)), 8, 5)
    st6 = Move(MoveType.STRAIGHT, tuple(card(r) for r in range(3, 9)), 8, 6)
    assert beats(st5_high, st5_low)
    assert not beats(st6, st5_low)
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
        assert sorted(g.finish_order) == [0, 1, 2], g.finish_order
        last = g.finish_order[-1]
        for seat in range(3):
            if seat != last:
                assert len(g.hands[seat]) == 0, (seat, g.hands[seat])
        assert sum(len(h) for h in g.hands) >= 1
    print("OK  50 random full games: all terminate, valid finish order, last holds leftover")


def test_pass_and_clear():
    hands = [
        [card(3, 0), card(3, 1), card(3, 2)],
        [card(4, 0), card(4, 1)],
        [card(5, 0), card(5, 1)],
    ]
    g = Game(num_players=3, num_decks=1, hands=hands, starting_player=0, seed=0)
    g.apply_move(0, g.legal_moves(0)[1])
    assert g.table_move is not None
    g.apply_move(1, PASS_MOVE)
    assert not g.done
    g.apply_move(2, PASS_MOVE)
    assert g.table_move is None, "table should have cleared"
    assert g.current_player == 0, g.current_player
    print("OK  pass & clear: table cleared after all others pass")


def test_penalty():
    hands = [[card(r, i) for i in range(4)] for r in range(3, 7)]
    hands[2] = [card(15), card(3)]
    hands[3] = [card(17), card(4)]
    hands[0] = [card(14), card(5)]
    hands[1] = [card(13), card(6)]
    m = Match(num_players=4, rng=None, seed=0)
    m.hands = hands
    m._apply_penalty([0, 1, 2, 3])
    transferred = {(src, dst, c.rank) for (src, dst, c) in m.last_transfers}
    assert (3, 0, 17) in transferred, transferred
    assert (2, 1, 15) in transferred, transferred
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
