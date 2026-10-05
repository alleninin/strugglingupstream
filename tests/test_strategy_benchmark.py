import copy
import unittest
from unittest.mock import patch

import numpy as np
import torch

from agents.ddqn_agent import DDQNAgent
from bots.greedy_bot import GreedyBot
from env import features
from game.cards import build_deck
from scripts.benchmark_strategy import (
    benchmark,
    policy_moves,
    position,
    sample_positions,
    tactical_cases,
)


class StrategyBenchmarkTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.cases = tactical_cases()

    def test_fixtures_conserve_cards_and_expectations_are_legal(self):
        deck = sorted(card.id for card in build_deck(2))
        for case in self.cases:
            with self.subTest(case=case.name):
                game = case.game
                cards = game.played_cards + [
                    card for hand in game.hands for card in hand
                ]
                self.assertEqual(sorted(card.id for card in cards), deck)
                for move in case.accepted:
                    self.assertIn(move, game.legal_moves())
                    copy.deepcopy(game).apply_move(0, move)
        result = benchmark(GreedyBot(), self.cases, [])
        self.assertTrue(all(case["passed"] for case in result["cases"]))

    def test_raw_failures_are_not_hidden_by_finish_safeguard(self):
        agent = DDQNAgent(*features.feature_dims(4), epsilon=1, device="cpu", seed=0)
        case = self.cases[0]
        legal = case.game.legal_moves()
        bad = next(i for i, move in enumerate(legal) if len(move.cards) == 1)
        scores = torch.zeros(len(legal))
        scores[bad] = 1
        rng = copy.deepcopy(agent.rng.bit_generator.state)
        with patch.object(agent, "_scores", return_value=scores):
            report = benchmark(agent, [case], [])
        self.assertTrue(report["cases"][0]["passed"])
        self.assertFalse(report["cases"][0]["raw_passed"])
        self.assertEqual(agent.rng.bit_generator.state, rng)
        self.assertEqual(agent.epsilon, 1)

    def test_policy_does_not_see_opponent_card_assignments(self):
        agent = DDQNAgent(
            *features.feature_dims(4), history_features=True, seed=0, device="cpu"
        )
        game = self.cases[3].game
        swapped = copy.deepcopy(game)
        swapped.hands[1], swapped.hands[2] = swapped.hands[2], swapped.hands[1]
        np.testing.assert_array_equal(
            features.state_for(game, 0, agent), features.state_for(swapped, 0, agent)
        )
        self.assertEqual(policy_moves(agent, game), policy_moves(agent, swapped))

    def test_pass_diagnostics_exclude_leads_and_forced_passes(self):
        lead = self.cases[0].game
        forced = position([3, 5, 7], table_rank=17, opponent_sizes=(1, 20, 20))
        result = benchmark(GreedyBot(), [], [lead, forced])["shared_states"]
        self.assertEqual(result["teacher_contests"], 0)
        self.assertEqual(result["urgent_responses"], 0)
        self.assertEqual(result["urgent_passes"], 0)

    def test_shared_positions_are_seeded_bounded_and_have_real_choices(self):
        first = sample_positions(games=3, seed=73, limit=5)
        second = sample_positions(games=3, seed=73, limit=5)
        self.assertEqual(len(first), 5)
        for one, two in zip(first, second):
            np.testing.assert_array_equal(
                features.state_vector(one, 0), features.state_vector(two, 0)
            )
            self.assertEqual(one.current_player, 0)
            self.assertGreater(len(one.legal_moves()), 1)
            self.assertFalse(one.finish_order)
        result = benchmark(GreedyBot(), [], first)
        self.assertEqual(result["shared_states"]["teacher_agreement"], 5)
        self.assertEqual(result["shared_states"]["missed_finishes"], 0)


if __name__ == "__main__":
    unittest.main()
