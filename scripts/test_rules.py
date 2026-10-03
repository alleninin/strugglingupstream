import sys
import os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from game.cards import Card, build_deck
from game.moves import Move, MoveType, generate_moves, beats, PASS_MOVE
from game.rules import Game
from game.match import Match
from env import features
from bots.greedy_bot import GreedyBot


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
        agents = [GreedyBot(num_players=3, num_decks=1, seed=seed * 10 + i)
                  for i in range(3)]
        steps = 0
        while not g.done:
            seat = g.current_player
            legal = g.legal_moves(seat)
            obs = features.state_vector(g, seat)
            move = agents[seat].act(obs, legal)
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


def test_airplane():
    hand = [card(3, 0), card(3, 1), card(3, 2),
            card(7, 0), card(7, 1), card(7, 2),
            card(9, 0), card(9, 1),
            card(11, 0), card(11, 1)]
    airplanes = [m for m in generate_moves(hand) if m.type == MoveType.AIRPLANE]
    assert airplanes, "expected an airplane move"
    assert all(len(m.cards) == 10 for m in airplanes)
    assert all(m.length == 10 for m in airplanes)

    low_air = Move(MoveType.AIRPLANE, tuple(
        [card(3, 0), card(3, 1), card(3, 2),
         card(5, 0), card(5, 1), card(5, 2),
         card(7, 0), card(7, 1),
         card(9, 0), card(9, 1)]), 5, 10)
    high_air = Move(MoveType.AIRPLANE, tuple(
        [card(5, 0), card(5, 1), card(5, 2),
         card(11, 0), card(11, 1), card(11, 2),
         card(3, 0), card(3, 1),
         card(7, 0), card(7, 1)]), 11, 10)
    assert beats(high_air, low_air)
    assert not beats(low_air, high_air)

    bomb = Move(MoveType.BOMB, tuple(card(4, i) for i in range(4)), 4, 4, is_bomb=True)
    assert beats(bomb, high_air)
    assert not beats(high_air, bomb)

    fh = Move(MoveType.FULL_HOUSE,
              tuple(card(6, 0) for _ in range(3)) + tuple(card(8, 0) for _ in range(2)),
              6, 5)
    assert not beats(high_air, fh)
    print("OK  airplane: 2 triples + 2 pairs (10 cards), ranking, bomb > airplane")


def test_greedy_bot():
    from bots.greedy_bot import GreedyBot, M

    # Critical edge case from the spec:
    # hand 6,7,7,7,8,8,9,9,10,10 -> triple 777 + tractor 88991010 + orphan 6.
    hand = [0] * 15
    for r, k in [(6, 1), (7, 3), (8, 2), (9, 2), (10, 2)]:
        hand[r - 3] = k
    assert M(hand) == 3, f"expected M=3, got {M(hand)}"

    groups = GreedyBot().partition_strategy(hand)
    assert any(g.type == "TRIPLE" and g.rank == 7 for g in groups), groups
    assert any(g.type == "CONSEC_PAIRS" and g.ranks == [8, 9, 10]
               for g in groups), groups
    orphans = [g for g in groups if g.type == "SINGLE"]
    assert len(orphans) == 1 and orphans[0].rank == 6, groups
    print("OK  greedy-bot: triple 777 + tractor 88991010 claimed before singles; "
          "6 is the sole orphan (not absorbed into a straight)")


def test_feature_dims_update():
    assert features.TYPE_DIM == 10, features.TYPE_DIM
    s_dim, a_dim = features.feature_dims(4, 2)
    hand = [card(3, 0), card(3, 1), card(3, 2),
            card(7, 0), card(7, 1), card(7, 2),
            card(9, 0), card(9, 1),
            card(11, 0), card(11, 1)]
    ap = [m for m in generate_moves(hand) if m.type == MoveType.AIRPLANE][0]
    mv = features.move_vector(ap)
    assert len(mv) == a_dim, (len(mv), a_dim)
    assert mv[features.NUM_RANKS + MoveType.AIRPLANE] == 1.0
    print(f"OK  feature dims reflect AIRPLANE + new combo types (TYPE_DIM=10, move dim={a_dim})")


def test_consecutive_moves():
    # 3-3-3 / 4-4-4-4 / 5-5-5  -> two consecutive triples (3,4) and a non-consecutive 5
    hand = [card(3, 0), card(3, 1), card(3, 2),
            card(4, 0), card(4, 1), card(4, 2),
            card(5, 0), card(5, 1), card(5, 2)]
    ct = [m for m in generate_moves(hand) if m.type == MoveType.CONSEC_TRIPLES]
    assert ct, "expected a consecutive-triples move"
    two = [m for m in ct if m.length == 2]
    assert two, "expected a 2-in-a-row triple move"
    assert all(len(m.cards) == 6 for m in two)
    assert any(m.rank == 4 for m in two), [m.rank for m in two]

    # 3-3 / 4-4 / 5-5  -> three consecutive pairs
    hand2 = [card(3, 0), card(3, 1),
             card(4, 0), card(4, 1),
             card(5, 0), card(5, 1)]
    cp = [m for m in generate_moves(hand2) if m.type == MoveType.CONSEC_PAIRS]
    assert cp, "expected a consecutive-pairs move"
    three = [m for m in cp if m.length == 3]
    assert three, "expected a 3-in-a-row pair move"
    assert all(len(m.cards) == 6 for m in three)
    assert three[0].rank == 5

    # beating: higher top rank wins, same structure/length required
    low_ct = Move(MoveType.CONSEC_TRIPLES,
                  tuple(card(3, i) for i in range(3)) + tuple(card(4, i) for i in range(3)),
                  4, 2)
    high_ct = Move(MoveType.CONSEC_TRIPLES,
                   tuple(card(4, i) for i in range(3)) + tuple(card(5, i) for i in range(3)),
                   5, 2)
    assert beats(high_ct, low_ct)
    assert not beats(low_ct, high_ct)

    low_cp = Move(MoveType.CONSEC_PAIRS,
                  tuple(card(3, i) for i in range(2)) + tuple(card(4, i) for i in range(2))
                  + tuple(card(5, i) for i in range(2)), 5, 3)
    high_cp = Move(MoveType.CONSEC_PAIRS,
                   tuple(card(4, i) for i in range(2)) + tuple(card(5, i) for i in range(2))
                   + tuple(card(6, i) for i in range(2)), 6, 3)
    assert beats(high_cp, low_cp)

    # a bomb beats a consecutive combo; a combo does not beat a bomb
    bomb = Move(MoveType.BOMB, tuple(card(9, i) for i in range(4)), 9, 4, is_bomb=True)
    assert beats(bomb, high_ct)
    assert not beats(high_ct, bomb)
    # different structure does not beat (pure 2-triple vs winged airplane)
    air = Move(MoveType.AIRPLANE, tuple(
        card(3, i) for i in range(3)) + tuple(card(4, i) for i in range(3))
        + tuple(card(7, i) for i in range(2)) + tuple(card(9, i) for i in range(2)),
        4, 10)
    assert not beats(high_ct, air)
    print("OK  consecutive triples (2-in-a-row) and pairs (3-in-a-row) generate & rank")


def main():
    test_deck()
    test_generate_moves()
    test_beats()
    test_full_game()
    test_pass_and_clear()
    test_penalty()
    test_airplane()
    test_consecutive_moves()
    test_greedy_bot()
    test_feature_dims_update()
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
