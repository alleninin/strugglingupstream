"""Behavioral checks for combination preservation, closing, and blocking."""
import unittest
import numpy as np

from bots.greedy_bot import GreedyBot, _exact_turns
from env import features
from game.cards import build_deck
from game.moves import MoveType, PASS_MOVE, generate_moves
from game.rules import Game
from agents.random_agent import RandomAgent
from evaluate import run_tournament
from training.train import make_opponent_policies
from unittest.mock import patch
from collections import Counter
from test_regressions import hands_for, histogram


def decision(ranks, table_rank=None, opponent_sizes=(20, 20, 20)):
    hands = [hands_for(ranks)[0]] + [build_deck()[:size] for size in opponent_sizes]
    game = Game(num_players=len(hands), hands=hands)
    if table_rank is not None:
        game.table_move = generate_moves(hands_for([table_rank])[0])[0]
        game.table_owner = 1
    legal = game.legal_moves(0)
    return GreedyBot(num_players=len(hands)).act(features.state_vector(game, 0), legal)


class GreedyStrategyTests(unittest.TestCase):
    def test_leads_whole_straight_before_low_orphan(self):
        move = decision([3, 6, 7, 8, 9, 10, 15, 17])
        self.assertEqual(move.type, MoveType.STRAIGHT)
        self.assertEqual(len(move.cards), 5)

    def test_leads_full_house_before_low_orphan(self):
        move = decision([3, 7, 7, 7, 9, 9, 15, 17])
        self.assertEqual(move.type, MoveType.FULL_HOUSE)

    def test_does_not_break_pair_to_follow_a_single(self):
        self.assertEqual(decision([7, 7, 9, 9, 11, 11], 3), PASS_MOVE)

    def test_small_hand_is_not_an_excuse_to_split_pair(self):
        self.assertEqual(decision([7, 7, 9, 9], 3), PASS_MOVE)

    def test_does_not_split_straight_to_follow_single(self):
        self.assertEqual(decision([5, 6, 7, 8, 9, 11, 11, 13, 13], 4), PASS_MOVE)

    def test_winning_bomb_is_played_instead_of_hoarded(self):
        move = decision([3, 3, 3, 3], 17)
        self.assertEqual(move.type, MoveType.BOMB)

    def test_entire_airplane_is_played_to_finish(self):
        move = decision([3, 3, 3, 7, 7, 7, 9, 9, 11, 11])
        self.assertEqual(move.type, MoveType.AIRPLANE)

    def test_threat_uses_high_single_instead_of_cheap_single(self):
        move = decision([5, 8, 9, 9, 11, 11, 17], 3, (1, 20, 20))
        self.assertEqual(move.rank, 17)

    def test_last_card_opponent_is_not_offered_a_low_single(self):
        move = decision([3, 7, 7, 9, 9, 11, 11], opponent_sizes=(1, 20, 20))
        self.assertEqual(move.type, MoveType.PAIR)

    def test_control_card_can_set_up_last_combo(self):
        move = decision([7, 7, 17], 15)
        self.assertEqual(move.rank, 17)

    def test_spare_high_singles_contest_without_endgame_pressure(self):
        for rank in (14, 15, 16, 17):
            with self.subTest(rank=rank):
                move = decision([7, 7, 9, 9, 11, 11, rank], 3)
                self.assertEqual(move.type, MoveType.SINGLE)
                self.assertEqual(move.rank, rank)

    def test_uses_spare_two_without_splitting_straight(self):
        move = decision([5, 6, 7, 8, 9, 15, 17], 4)
        self.assertEqual(move.rank, 15)

    def test_pressure_starts_before_opponent_reaches_two_bombs(self):
        ranks = [3, 3, 5, 5, 7, 7, 9, 9, 11, 11, 13, 13, 15, 15]
        self.assertEqual(decision(ranks, 14), PASS_MOVE)
        move = decision(ranks, 14, (9, 20, 20))
        self.assertEqual(move.rank, 15)

    def test_contests_when_falling_behind_even_with_large_opponent_hands(self):
        ranks = [3] * 3 + [5] * 3 + [7] * 3 + [9] * 3 + [11] * 3 + [13] * 3 + [15] * 2
        self.assertEqual(decision(ranks, 14), PASS_MOVE)
        move = decision(ranks, 14, (15, 20, 20))
        self.assertEqual(move.rank, 15)

    def test_bombs_are_available_under_pressure_but_not_wasted_early(self):
        ranks = [3] * 4 + [5] * 2 + [7] * 2 + [9] * 2
        self.assertEqual(decision(ranks, 17), PASS_MOVE)
        self.assertEqual(decision(ranks, 17, (8, 20, 20)).type, MoveType.BOMB)

    def test_pass_when_no_legal_response_exists(self):
        self.assertEqual(decision([3, 3, 5, 5, 7, 7], 17, (1, 20, 20)), PASS_MOVE)

    def test_bomb_only_response_uses_actual_table_combination_size(self):
        ranks = [3] * 4 + [4] * 4 + [11] * 2 + [13] * 2
        game = Game(hands=[hands_for(ranks)[0]] + [build_deck()[:10] for _ in range(3)])
        airplane = hands_for([5] * 3 + [7] * 3 + [9] * 2 + [12] * 2)[0]
        game.table_move = next(move for move in generate_moves(airplane) if move.type == MoveType.AIRPLANE)
        game.table_owner = 1
        move = GreedyBot().act(features.state_vector(game, 0), game.legal_moves())
        self.assertEqual((move.type, move.rank), (MoveType.BOMB, 4))

    def test_low_single_run_is_contested_on_first_play(self):
        hands = hands_for(list(range(3, 11)) + [13] * 4 + [14] * 4,
                          [11, 11, 12, 12, 15, 16, 17],
                          [11, 11, 12, 12, 15, 16, 17],
                          [9, 9, 10, 10, 15, 15, 14])
        game = Game(hands=hands)
        lead = next(move for move in game.legal_moves() if move.type == MoveType.SINGLE and move.rank == 3)
        game.apply_move(0, lead)
        move = GreedyBot().act(features.state_vector(game, 1), game.legal_moves())
        self.assertFalse(move.is_pass)
        self.assertEqual(move.rank, 15)
        game.apply_move(1, move)
        self.assertEqual(Counter(card.rank for card in game.hands[1]),
                         Counter([11, 11, 12, 12, 16, 17]))

    def test_full_house_uses_wings_that_preserve_straight(self):
        move = decision([3, 3, 3, 5, 5, 6, 7, 8, 9, 11, 11])
        # Playing 333 + JJ leaves a straight plus 5; using 55 destroys the run.
        if move.type == MoveType.FULL_HOUSE:
            self.assertEqual(sorted(c.rank for c in move.cards), [3, 3, 3, 11, 11])
        else:
            self.assertEqual(move.type, MoveType.STRAIGHT)

    def test_exact_partition_can_split_a_bomb_to_make_two_full_houses(self):
        counts = tuple(histogram([3, 3, 3, 3, 3, 4, 4, 4, 4, 4]))
        self.assertEqual(_exact_turns(counts), 2)

    def test_search_agrees_with_exhaustive_partition_on_small_hands(self):
        from functools import lru_cache
        @lru_cache(None)
        def exhaustive(counts):
            ranks = [rank + 3 for rank, count in enumerate(counts) for _ in range(count)]
            if not ranks:
                return 0
            best = len(ranks)
            for move in generate_moves(hands_for(ranks)[0]):
                remaining = list(counts)
                for card in move.cards:
                    remaining[card.rank - 3] -= 1
                best = min(best, 1 + exhaustive(tuple(remaining)))
            return best
        rng = np.random.default_rng(6)
        for _ in range(20):
            ranks = rng.integers(3, 10, size=8).tolist()
            counts = tuple(histogram(ranks))
            self.assertEqual(_exact_turns(counts), exhaustive(counts))


class TournamentTests(unittest.TestCase):
    def test_every_subset_and_every_seat_gets_matched_deals(self):
        pool = {name: RandomAgent() for name in 'ABCDEF'}
        with patch('evaluate.simulate_game', return_value=[0, 1, 2, 3]):
            report = run_tournament(pool, rounds=2, seed=600)
        self.assertEqual(report['num_games'], 120)  # 15 subsets × 4 seats × 2 rounds
        subsets = Counter()
        games = report['games']
        for offset in range(0, len(games), 4):
            block = games[offset:offset + 4]
            self.assertEqual(len({game['seed'] for game in block}), 1)
            names = sorted(block[0]['seats'])
            subsets[tuple(names)] += 1
            for seat in range(4):
                self.assertEqual(sorted(game['seats'][seat] for game in block), names)
        self.assertEqual(len(subsets), 15)
        self.assertEqual(set(subsets.values()), {2})

    def test_evaluation_disables_exploration_and_restores_it(self):
        pool = {name: RandomAgent() for name in 'ABCD'}
        for agent in pool.values():
            agent.epsilon = .4
        def simulate(agents, players, decks, seed):
            self.assertTrue(all(agent.epsilon == 0 for agent in agents))
            return [0, 1, 2, 3]
        with patch('evaluate.simulate_game', side_effect=simulate):
            first = run_tournament(pool, games=5, seed=600)
            second = run_tournament(pool, games=5, seed=600)
        self.assertEqual(first, second)
        self.assertEqual(first['num_games'], 8)
        self.assertTrue(all(agent.epsilon == .4 for agent in pool.values()))

    def test_mixed_opponents_are_seeded_and_vary_across_episodes(self):
        choices = []
        for seed in range(10):
            one = make_opponent_policies(4, seed, opponent='mixed')
            two = make_opponent_policies(4, seed, opponent='mixed')
            kinds = tuple(type(policy.__self__).__name__ for policy in one)
            self.assertEqual(kinds, tuple(type(policy.__self__).__name__ for policy in two))
            choices.append(kinds)
        self.assertGreater(len(set(choices)), 1)
        self.assertEqual({kind for kinds in choices for kind in kinds}, {'GreedyBot', 'RandomAgent'})


if __name__ == '__main__':
    unittest.main()
