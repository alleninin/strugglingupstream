"""Regression coverage for rules, training targets, rewards and public commands."""
import contextlib
from dataclasses import replace
import io
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch

from agents.ddqn_agent import DDQNAgent
from agents.dqn_agent import DQNAgent
from agents.prioritized_replay import PrioritizedReplayBuffer, SumTree
from agents.qlearning import QLearningAgent
from agents.random_agent import RandomAgent
from bots.greedy_bot import GreedyBot, M
from bots.shaped_reward import ShapedReward, ShapedRewardConfig, moves_to_empty, terminal_reward
from bots.shaped_reward_bot import ShapedRewardBot
from env import features
from env.env import ZhengShangYouEnv
from game.cards import build_deck
from game.match import Match
from game.moves import Move, MoveType, PASS_MOVE, generate_moves
from game.rules import Game
from scripts.watch import load_seat0
from training.train import make_opponent_policies, train
import evaluate


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
            self.assertLessEqual(max(map(len, game.hands)) - min(map(len, game.hands)), 1)

    def test_invalid_setup(self):
        for kwargs in ({"num_players": 1}, {"num_decks": 0},
                       {"num_players": 55, "num_decks": 1},
                       {"num_players": 2, "hands": [[]]}):
            with self.assertRaises(ValueError):
                Game(**kwargs)

    def test_jokers_are_only_singles_even_with_four_decks(self):
        moves = generate_moves(hands_for([16] * 4 + [17] * 4 + [3] * 3 + [4] * 2)[0])
        for move in moves:
            if any(c.rank >= 16 for c in move.cards):
                self.assertEqual(move.type, MoveType.SINGLE)

    def test_airplane_can_include_twos_but_sequences_cannot(self):
        hand = hands_for([3] * 3 + [15] * 3 + [7] * 2 + [9] * 2)[0]
        self.assertTrue(any(m.type == MoveType.AIRPLANE and m.rank == 15
                            for m in generate_moves(hand)))
        for move in generate_moves(build_deck()):
            if move.type in (MoveType.STRAIGHT, MoveType.CONSEC_PAIRS, MoveType.CONSEC_TRIPLES):
                self.assertTrue(all(c.rank <= 14 for c in move.cards))

    def test_invalid_moves_do_not_change_game(self):
        game = game_for([3, 4, 5], [6, 7])
        card = game.hands[0][0]
        invalid = [PASS_MOVE, Move(MoveType.SINGLE, (game.hands[1][0],), 6, 1),
                   Move(MoveType.PAIR, (card, card), 3, 2),
                   Move(MoveType.SINGLE, (card,), 17, 1),
                   Move(MoveType.BOMB, (card,), 3, 4, is_bomb=True)]
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
                    self.assertEqual(len(played) + sum(map(len, game.hands)), 54 * decks)
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

    def test_trick_reward_only_after_actual_resolution_including_leads(self):
        for opponent_passes in (True, False):
            def opponent(obs, legal):
                return PASS_MOVE if opponent_passes else legal[0]
            env = ZhengShangYouEnv(num_players=2, opponent_policies=[opponent])
            env.game = game_for([3, 9], [4, 5])
            state = features.state_vector(env.game, 0)
            action = env.get_legal_moves()[0]
            next_state, _, done, info = env.step(action)
            won = 0 in info["trick_winners"]
            self.assertEqual(won, opponent_passes)
            comp = ShapedReward(ShapedRewardConfig(num_players=2)).components(
                state, action, next_state, done, env.game.finish_order, 0, won_trick=won)
            if opponent_passes:
                self.assertGreater(comp["trick"], 0)
            else:
                self.assertEqual(comp["trick"], 0)

    def test_agent_finishing_rolls_to_terminal_placement(self):
        env = ZhengShangYouEnv(num_players=3,
                              opponent_policies=[lambda obs, legal: legal[0]] * 2)
        env.game = game_for([3], [4, 5], [6, 7])
        state, reward, done, info = env.step(env.get_legal_moves()[0])
        self.assertTrue(done)
        self.assertIsNone(state)
        self.assertEqual(reward, 1.1)
        self.assertEqual(info["next_legal_moves"], [])

    def test_reward_estimate_uses_legal_combinations(self):
        for ranks in ([3, 3, 3, 7, 7], [3, 3, 4, 4, 5, 5],
                      [3, 3, 3, 7, 7, 7, 9, 9, 11, 11]):
            self.assertEqual(moves_to_empty(histogram(ranks)), 1)
        self.assertEqual(moves_to_empty(histogram([16, 16, 17, 17])), 4)
        self.assertEqual(M(histogram([16, 16])), 2)

    def test_terminal_rewards_for_configured_player_count(self):
        self.assertEqual(terminal_reward([0, 1], 1, ShapedRewardConfig(num_players=2)), -1)
        self.assertEqual(terminal_reward([0, 1, 2], 1, ShapedRewardConfig(num_players=3)), 0)
        self.assertEqual(terminal_reward([0, 1, 2, 3], 1, ShapedRewardConfig()), .3)

    def test_reward_log_accumulates_every_step(self):
        dims = features.feature_dims(2)
        bot = ShapedRewardBot(*dims, cfg=ShapedRewardConfig(num_players=2),
                              agent_type="qlearning", log_every=0)
        game = game_for([3, 4], [5, 6])
        state = features.state_vector(game, 0)
        action = game.legal_moves()[0]
        parts = [{"dense": 2., "trick": 3., "terminal": 0., "total": 5.},
                 {"dense": 4., "trick": 0., "terminal": 1., "total": 5.}]
        with patch.object(bot.reward, "components", side_effect=parts):
            bot.observe((state, action, 0, state, False, [action]))
            bot.observe((state, action, 0, None, True, []))
        self.assertEqual((bot._acc_dense, bot._acc_trick, bot._acc_terminal), (6, 3, 1))

    def test_finished_opponent_is_not_a_blocking_threat(self):
        bot = GreedyBot()
        card = hands_for([15])[0][0]
        move = Move(MoveType.SINGLE, (card,), 15, 1)
        hand = histogram([3, 3, 5, 5, 7, 7, 15, 15])
        self.assertEqual(bot._follow([move, PASS_MOVE], None, [0, 20, 20], hand), PASS_MOVE)


class LearningTests(unittest.TestCase):
    def test_linear_learner_can_reverse_action_preferences_by_state(self):
        agent = QLearningAgent(2, features.feature_dims(2)[1], alpha=.3,
                               epsilon=0, min_epsilon=0)
        actions = generate_moves(hands_for([3, 7])[0])
        states = [np.array([1, 0], dtype=np.float32), np.array([0, 1], dtype=np.float32)]
        for _ in range(300):
            for i, state in enumerate(states):
                for j, action in enumerate(actions):
                    agent.observe((state, action, 1. if i == j else -1., None, True, []))
        for i, state in enumerate(states):
            self.assertEqual(agent.act(state, actions), actions[i])

    def test_legacy_linear_checkpoint_keeps_existing_predictions(self):
        agent = QLearningAgent(*features.feature_dims(2))
        old_weights = np.arange(agent.base_dim, dtype=np.float32) / 100
        game = game_for([3, 4], [5, 6])
        state, action = features.state_vector(game, 0), game.legal_moves()[0]
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = str(Path(directory) / "old.npy")
            np.save(checkpoint, old_weights)
            agent.load(checkpoint)
        old_phi = features.combined_vector(state, features.move_vector(action))
        self.assertAlmostEqual(agent.q(agent._phi(state, action)), float(old_weights @ old_phi), places=5)

    def test_negative_dqn_future_values_are_not_clipped_to_zero(self):
        agent = DQNAgent(2, 1, batch_size=1, gamma=.5, seed=1)
        for net in (agent.policy_net, agent.target_net):
            for parameter in net.parameters():
                parameter.data.zero_()
        agent.target_net.net[-1].bias.data.fill_(-2.)
        agent.buffer = [(np.zeros(3, dtype=np.float32), 1.,
                         [np.zeros(3, dtype=np.float32)], False)]
        original = torch.nn.functional.smooth_l1_loss
        targets = []
        def loss(prediction, target):
            targets.append(target.detach().cpu().numpy())
            return original(prediction, target)
        with patch("agents.dqn_agent.nn.functional.smooth_l1_loss", side_effect=loss):
            agent._learn()
        np.testing.assert_allclose(targets[0], [0.])

    def test_ddqn_single_action_advantage_has_zero_gradient(self):
        dims = features.feature_dims(2)
        agent = DDQNAgent(*dims, batch_size=1, seed=2, dueling=True, learning_starts=1)
        game = game_for([3], [4])
        state = features.state_vector(game, 0)
        action = game.legal_moves()[0]
        before = [p.detach().clone() for p in agent.policy_net.adv_stream.parameters()]
        agent.observe((state, action, 1., None, True, [], [action]))
        for old, new in zip(before, agent.policy_net.adv_stream.parameters()):
            torch.testing.assert_close(old, new, atol=0, rtol=0)

    def test_replay_does_not_transform_default_priority_twice(self):
        buffer = PrioritizedReplayBuffer(capacity=3, alpha=.5)
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
        tree.add("empty", 0.)
        tree.add("valid", 1.)
        self.assertEqual(tree.get(0.)[1], "valid")

    def test_non_power_of_two_replay_sampling(self):
        buffer = PrioritizedReplayBuffer(capacity=3, rng=np.random.default_rng(4))
        for i in range(3):
            buffer.push(i, priority=i + 1)
        data, indices, weights = buffer.sample(100)
        self.assertEqual(set(data), {0, 1, 2})
        self.assertTrue(np.isfinite(weights).all())
        self.assertTrue(all(0 <= i < 3 for i in indices))

    def test_seed_controls_network_initialization(self):
        for cls in (DQNAgent, DDQNAgent):
            one, two = cls(2, 1, seed=8), cls(2, 1, seed=8)
            for a, b in zip(one.policy_net.parameters(), two.policy_net.parameters()):
                torch.testing.assert_close(a, b)

    def test_linear_updates_remain_finite_for_large_counts(self):
        agent = QLearningAgent(*features.feature_dims(4), alpha=.05)
        game = Game(seed=5)
        state, action = features.state_vector(game, 0), game.legal_moves()[0]
        for _ in range(500):
            agent.observe((state, action, 1., state, False, [action]))
        self.assertTrue(np.isfinite(agent.w).all())
        self.assertLess(np.linalg.norm(agent.w), 100)

    def test_dqn_replay_overwrites_oldest_without_growing(self):
        agent = DQNAgent(*features.feature_dims(2), buffer_size=2, batch_size=10)
        game = game_for([3], [4])
        state, action = features.state_vector(game, 0), game.legal_moves()[0]
        for reward in (1., 2., 3.):
            agent.observe((state, action, reward, None, True, []))
        self.assertEqual(sorted(item[1] for item in agent.buffer), [2., 3.])

    def test_watch_loads_requested_qlearning_checkpoint(self):
        dims = features.feature_dims(2)
        source = QLearningAgent(*dims)
        source.w[:] = .125
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = str(Path(directory) / "weights.pt")
            source.save(checkpoint)
            self.assertTrue(Path(checkpoint).exists())
            loaded, _ = load_seat0("qlearning", checkpoint, *dims, seed=3, num_players=2)
            np.testing.assert_array_equal(loaded.w, source.w)
            self.assertEqual(loaded.epsilon, 0.)

    def test_opponent_selection(self):
        self.assertIsInstance(make_opponent_policies(3, None, opponent="random")[0].__self__, RandomAgent)
        self.assertIsInstance(make_opponent_policies(3, 0, opponent="greedy")[0].__self__, GreedyBot)
        with self.assertRaises(ValueError):
            make_opponent_policies(3, 0, opponent="missing")

    def test_periodic_save_creates_destination_before_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = str(Path(directory) / "new-directory" / "agent.npy")
            with patch("training.train.evaluate_vs_opponent", return_value=.5), contextlib.redirect_stdout(io.StringIO()):
                train("qlearning", 1, 2, 1, 4, checkpoint, 1, opponent="random", demo_games=0)
            self.assertTrue(Path(checkpoint).exists())

    def test_evaluation_skips_missing_and_incompatible_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            checkpoint = str(Path(directory) / "weights.npy")
            self.assertIsNone(evaluate.load_checkpoint(QLearningAgent, checkpoint, 2, 1))
            np.save(checkpoint, np.zeros(99))
            self.assertIsNone(evaluate.load_checkpoint(QLearningAgent, checkpoint, 2, 1))

    def test_tournament_win_rate_uses_actual_appearances(self):
        pool = {name: RandomAgent() for name in ('A', 'B', 'C', 'D', 'E')}
        with patch.object(evaluate, "simulate_game", return_value=[0, 1, 2, 3]):
            report = evaluate.run_tournament(pool, rounds=1)
        self.assertEqual(report['num_games'], 20)
        for row in report['results'].values():
            self.assertEqual(row['appearances'], 16)
            self.assertEqual(row['win_rate'], .25)


if __name__ == "__main__":
    unittest.main()
