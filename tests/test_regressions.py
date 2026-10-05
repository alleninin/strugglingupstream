"""Regression coverage for rules, training targets, rewards and public commands."""

import contextlib
import io
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

import evaluate
from agents.ddqn_agent import DDQNAgent
from agents.prioritized_replay import PrioritizedReplayBuffer, SumTree
from bots.greedy_bot import GreedyBot
from bots.random_bot import RandomAgent
from env import features
from env.env import ZhengShangYouEnv
from game.cards import build_deck
from game.match import Match
from game.moves import PASS_MOVE, Move, MoveType, generate_moves
from game.rules import Game
from training.train import make_opponent_policies, train


def hands_for(*rank_lists):
    deck = build_deck(4)
    hands = []
    for ranks in rank_lists:
        hand = []
        for rank in ranks:
            card = next(c for c in deck if c.rank == rank)
            deck.remove(card)
            hand.append(card)
        hands.append(hand)
    return hands


def game_for(*ranks):
    return Game(num_players=len(ranks), hands=hands_for(*ranks))


def histogram(ranks):
    return np.bincount(np.asarray(ranks) - 3, minlength=15)


class RulesTests(unittest.TestCase):
    def test_every_card_dealt_with_uneven_hands(self):
        for players in (2, 3, 4, 5, 7):
            game = Game(num_players=players, num_decks=1, seed=7)
            ids = [c.id for h in game.hands for c in h]
            self.assertEqual(len(ids), 54)
            self.assertEqual(len(set(ids)), 54)
            self.assertLessEqual(
                max(map(len, game.hands)) - min(map(len, game.hands)), 1
            )

    def test_invalid_setup(self):
        for kwargs in (
            {"num_players": 1},
            {"num_decks": 0},
            {"num_players": 55, "num_decks": 1},
            {"num_players": 2, "hands": [[]]},
        ):
            with self.assertRaises(ValueError):
                Game(**kwargs)

    def test_jokers_are_only_singles_even_with_four_decks(self):
        moves = generate_moves(hands_for([16] * 4 + [17] * 4 + [3] * 3 + [4] * 2)[0])
        for move in moves:
            if any(c.rank >= 16 for c in move.cards):
                self.assertEqual(move.type, MoveType.SINGLE)

    def test_airplane_can_include_twos_but_sequences_cannot(self):
        hand = hands_for([3] * 3 + [15] * 3 + [7] * 2 + [9] * 2)[0]
        self.assertTrue(
            any(
                m.type == MoveType.AIRPLANE and m.rank == 15
                for m in generate_moves(hand)
            )
        )
        for move in generate_moves(build_deck()):
            if move.type in (
                MoveType.STRAIGHT,
                MoveType.CONSEC_PAIRS,
                MoveType.CONSEC_TRIPLES,
            ):
                self.assertTrue(all(c.rank <= 14 for c in move.cards))

    def test_invalid_moves_do_not_change_game(self):
        game = game_for([3, 4, 5], [6, 7])
        card = game.hands[0][0]
        invalid = [
            PASS_MOVE,
            Move(MoveType.SINGLE, (game.hands[1][0],), 6, 1),
            Move(MoveType.PAIR, (card, card), 3, 2),
            Move(MoveType.SINGLE, (card,), 17, 1),
            Move(MoveType.BOMB, (card,), 3, 4, is_bomb=True),
        ]
        original = [h.copy() for h in game.hands]
        for move in invalid:
            with self.assertRaises(ValueError):
                game.apply_move(0, move)
            self.assertEqual(game.hands, original)
            self.assertIsNone(game.table_move)
            self.assertEqual(game.current_player, 0)

    def test_reject_lower_play(self):
        game = game_for([7, 8], [3, 4])
        game.apply_move(0, game.legal_moves()[0])
        with self.assertRaises(ValueError):
            game.apply_move(1, generate_moves(game.hands[1])[0])

    def test_equivalent_deserialized_card_is_removed(self):
        game = game_for([3, 4], [5, 6])
        move = game.legal_moves()[0]
        game.apply_move(0, replace(move, cards=(replace(move.cards[0]),)))
        self.assertEqual([c.rank for c in game.hands[0]], [4])

    def test_finished_owner_requires_all_remaining_players_to_pass(self):
        game = game_for([3], [4, 5], [6, 7])
        game.apply_move(0, game.legal_moves()[0])
        game.apply_move(1, PASS_MOVE)
        self.assertIsNotNone(game.table_move)
        game.apply_move(2, PASS_MOVE)
        self.assertIsNone(game.table_move)
        self.assertEqual(game.current_player, 1)
        self.assertEqual(game.last_trick_winner, 0)
        self.assertEqual(game.legal_moves(0), [])

    def test_no_legal_moves_after_game_end(self):
        game = game_for([3], [4])
        game.apply_move(0, game.legal_moves()[0])
        self.assertTrue(game.done)
        self.assertEqual(game.legal_moves(1), [])

    def test_match_redeals_then_transfers_highest_cards(self):
        match, control = Match(seed=19), Match(seed=19)
        first = [[c.id for c in h] for h in match.next_deal()]
        control.next_deal()
        fresh = [h.copy() for h in control.next_deal()]
        second = match.next_deal([0, 1, 2, 3])
        self.assertNotEqual(first, [[c.id for c in h] for h in second])
        self.assertEqual([len(h) for h in second], [28, 28, 26, 26])
        for loser, winner, card in match.last_transfers:
            self.assertEqual(card.rank, max(c.rank for c in fresh[loser]))
            self.assertIn(card, second[winner])
            self.assertNotIn(card, second[loser])
        self.assertEqual(len({c.id for h in second for c in h}), 108)

    def test_small_matches_have_disjoint_winners_and_losers(self):
        for players in (2, 3, 4, 5):
            match = Match(num_players=players, num_decks=1, seed=3)
            match.next_deal(list(range(players)))
            self.assertEqual(len(match.last_transfers), min(2, players // 2))
            self.assertTrue(all(src != dst for src, dst, _ in match.last_transfers))

    def test_random_games_terminate_and_conserve_cards(self):
        for players, decks in ((2, 1), (3, 1), (4, 1), (4, 2), (5, 2)):
            for seed in range(5):
                game = Game(num_players=players, num_decks=decks, seed=seed)
                bot = RandomAgent(seed=seed)
                played = set()
                steps = 0
                while not game.done:
                    move = bot.act(None, game.legal_moves())
                    self.assertTrue(played.isdisjoint(c.id for c in move.cards))
                    played.update(c.id for c in move.cards)
                    game.apply_move(game.current_player, move)
                    steps += 1
                    self.assertLess(steps, 3000)
                    self.assertEqual(
                        len(played) + sum(map(len, game.hands)), 54 * decks
                    )
                self.assertEqual(sorted(game.finish_order), list(range(players)))


class EnvironmentRewardTests(unittest.TestCase):
    def test_seeded_sequences_vary_but_are_reproducible(self):
        one, two = ZhengShangYouEnv(seed=42), ZhengShangYouEnv(seed=42)
        signatures = []
        for _ in range(3):
            np.testing.assert_array_equal(one.reset(), two.reset())
            a = [[c.id for c in h] for h in one.game.hands]
            b = [[c.id for c in h] for h in two.game.hands]
            self.assertEqual(a, b)
            signatures.append(a)
        self.assertNotEqual(signatures[0], signatures[1])
        np.testing.assert_array_equal(one.reset(seed=6), one.reset(seed=6))

    def test_invalid_env_action_is_not_silently_replaced(self):
        env = ZhengShangYouEnv(num_players=2)
        env.game = game_for([3, 4], [5, 6])
        with self.assertRaises(ValueError):
            env.step(PASS_MOVE)

    def test_invalid_opponent_action_is_not_silently_replaced(self):
        env = ZhengShangYouEnv(num_players=2)
        env.game = game_for([3, 4], [5, 6])
        action = env.get_legal_moves()[0]
        env.opponent_policies = [lambda obs, legal: action]
        with self.assertRaisesRegex(ValueError, "not in the player's hand"):
            env.step(action)

    def test_match_rejects_invalid_agent_action(self):
        class InvalidBot:
            def act(self, obs, legal):
                return PASS_MOVE

        with self.assertRaisesRegex(ValueError, "cannot pass when leading"):
            Match(num_players=2).play_match([InvalidBot(), InvalidBot()], 1)

    def test_agent_finishing_rolls_to_terminal_placement(self):
        env = ZhengShangYouEnv(
            num_players=3, opponent_policies=[lambda obs, legal: legal[0]] * 2
        )
        env.game = game_for([3], [4, 5], [6, 7])
        state, reward, done, info = env.step(env.get_legal_moves()[0])
        self.assertTrue(done)
        self.assertIsNone(state)
        self.assertEqual(reward, 1.1)
        self.assertEqual(info["next_legal_moves"], [])

    def test_finished_opponent_is_not_a_blocking_threat(self):
        bot = GreedyBot()
        card = hands_for([15])[0][0]
        move = Move(MoveType.SINGLE, (card,), 15, 1)
        hand = histogram([3, 3, 5, 5, 7, 7, 15, 15])
        self.assertEqual(
            bot._follow([move, PASS_MOVE], None, [0, 20, 20], hand), PASS_MOVE
        )


class LearningTests(unittest.TestCase):
    def test_ddqn_single_action_advantage_has_zero_gradient(self):
        dims = features.feature_dims(2)
        agent = DDQNAgent(*dims, batch_size=1, seed=2, dueling=True, learning_starts=1)
        game = game_for([3], [4])
        state = features.state_vector(game, 0)
        action = game.legal_moves()[0]
        before = [p.detach().clone() for p in agent.policy_net.adv_stream.parameters()]
        agent.observe((state, action, 1.0, None, True, [], [action]))
        for old, new in zip(before, agent.policy_net.adv_stream.parameters()):
            torch.testing.assert_close(old, new, atol=0, rtol=0)

    def test_replay_does_not_transform_default_priority_twice(self):
        buffer = PrioritizedReplayBuffer(capacity=3, alpha=0.5)
        buffer.push("first", priority=100)
        buffer.push("second")
        self.assertAlmostEqual(buffer.tree.tree[3], buffer.tree.tree[4])
        buffer.update_priorities([0], [400])
        buffer.push("third")
        self.assertAlmostEqual(buffer.tree.tree[3], buffer.tree.tree[5])
        buffer.push("fourth")
        self.assertEqual(len(buffer), 3)
        self.assertNotIn("first", buffer.tree.data)

    def test_sum_tree_zero_boundary_skips_empty_leaf(self):
        tree = SumTree(4)
        tree.add("empty", 0.0)
        tree.add("valid", 1.0)
        self.assertEqual(tree.get(0.0)[1], "valid")

    def test_non_power_of_two_replay_sampling(self):
        buffer = PrioritizedReplayBuffer(capacity=3, rng=np.random.default_rng(4))
        for i in range(3):
            buffer.push(i, priority=i + 1)
        data, indices, weights = buffer.sample(100)
        self.assertEqual(set(data), {0, 1, 2})
        self.assertTrue(np.isfinite(weights).all())
        self.assertTrue(all(0 <= i < 3 for i in indices))

    def test_seed_controls_network_initialization(self):
        for cls in (DDQNAgent,):
            one, two = cls(2, 1, seed=8), cls(2, 1, seed=8)
            for a, b in zip(one.policy_net.parameters(), two.policy_net.parameters()):
                torch.testing.assert_close(a, b)

    def test_opponent_selection(self):
        self.assertIsInstance(
            make_opponent_policies(3, None, opponent="random")[0].__self__, RandomAgent
        )
        self.assertIsInstance(
            make_opponent_policies(3, 0, opponent="greedy")[0].__self__, GreedyBot
        )
        with self.assertRaises(ValueError):
            make_opponent_policies(3, 0, opponent="missing")

    def test_periodic_save_creates_destination_before_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = str(Path(directory) / "new-directory" / "agent.npy")
            with (
                patch("training.train.evaluate_vs_opponent", return_value=0.5),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                train("ddqn", 1, 2, 1, 4, checkpoint, 1, opponent="random")
            self.assertTrue(Path(checkpoint).exists())

    def test_evaluation_skips_missing_and_incompatible_checkpoints(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            checkpoint = str(Path(directory) / "weights.npy")
            self.assertIsNone(evaluate.load_checkpoint(DDQNAgent, checkpoint, 2, 1))
            np.save(checkpoint, np.zeros(99))
            self.assertIsNone(evaluate.load_checkpoint(DDQNAgent, checkpoint, 2, 1))

    def test_tournament_win_rate_uses_actual_appearances(self):
        pool = {name: RandomAgent() for name in ("A", "B", "C", "D", "E")}
        with patch.object(evaluate, "simulate_game", return_value=[0, 1, 2, 3]):
            report = evaluate.run_tournament(pool, rounds=1)
        self.assertEqual(report["num_games"], 20)
        for row in report["results"].values():
            self.assertEqual(row["appearances"], 16)
            self.assertEqual(row["win_rate"], 0.25)


if __name__ == "__main__":
    unittest.main()
