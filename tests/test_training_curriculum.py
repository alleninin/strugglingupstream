import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from agents.ddqn_agent import DDQNAgent
from env import features
from env.env import ZhengShangYouEnv
from game.moves import PASS_MOVE
from game.rules import Game
from training.curriculum import EndgamePool, SelfPlayPool


class TrainingCurriculumTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.game = Game(seed=31)
        self.state = features.state_vector(self.game, 0)
        self.legal = self.game.legal_moves()

    def test_monte_carlo_uses_complete_discounted_returns_without_bootstrap(self):
        agent = DDQNAgent(
            *features.feature_dims(4),
            gamma=0.5,
            target_method="monte-carlo",
            learning_starts=1000,
            seed=1,
            device="cpu",
        )
        for i, reward in enumerate((1.0, 2.0, 4.0)):
            done = i == 2
            agent.observe(
                (
                    self.state,
                    self.legal[0],
                    reward,
                    None if done else self.state,
                    done,
                    [] if done else self.legal,
                    self.legal,
                )
            )
            if not done:
                self.assertEqual(len(agent.buffer), 0)
        data = agent.buffer.tree.data[:3]
        self.assertEqual([item.reward for item in data], [3.0, 4.0, 4.0])
        self.assertTrue(
            all(item.next_moves is None and item.discount == 0 for item in data)
        )
        with patch.object(
            agent.target_net, "forward", side_effect=AssertionError("bootstrapped")
        ):
            torch.testing.assert_close(
                agent._targets(data), torch.tensor([3.0, 4.0, 4.0])
            )
        self.assertFalse(agent.pending)

    def test_abandoned_monte_carlo_episode_is_discarded(self):
        agent = DDQNAgent(*features.feature_dims(4), target_method="monte-carlo")
        agent.observe(
            (self.state, self.legal[0], 5.0, self.state, False, self.legal, self.legal)
        )
        agent.reset_episode()
        agent.observe((self.state, self.legal[0], -1.0, None, True, [], self.legal))
        self.assertEqual(len(agent.buffer), 1)
        self.assertEqual(agent.buffer.tree.data[0].reward, -1.0)

    def test_snapshot_stays_frozen_and_does_not_consume_learner_rng(self):
        agent = DDQNAgent(*features.feature_dims(4), seed=4, epsilon=1, device="cpu")
        pool = SelfPlayPool(4, 2, seed=4, capacity=2)
        pool.capture(agent)
        policy = pool.snapshots[0]
        expected = policy(self.state, self.legal)
        rng = copy.deepcopy(agent.rng.bit_generator.state)
        weights = copy.deepcopy(policy.keywords["network"].state_dict())
        with torch.no_grad():
            for parameter in agent.policy_net.parameters():
                parameter.add_(10)
        self.assertEqual(policy(self.state, self.legal), expected)
        self.assertEqual(agent.rng.bit_generator.state, rng)
        for name, value in policy.keywords["network"].state_dict().items():
            torch.testing.assert_close(value, weights[name])
        self.assertTrue(
            all(not p.requires_grad for p in policy.keywords["network"].parameters())
        )
        pool.capture(agent)
        pool.capture(agent)
        self.assertEqual(len(pool.snapshots), 2)
        self.assertTrue(
            any(p in pool.snapshots for _ in range(30) for p in pool.policies(7))
        )

    def test_anchor_survives_recent_snapshot_eviction_and_learner_changes(self):
        agent = DDQNAgent(*features.feature_dims(4), seed=4, device="cpu")
        pool = SelfPlayPool(4, 2, capacity=1)
        pool.capture(agent, anchor=True)
        anchor = pool.anchor
        expected = anchor(self.state, self.legal)
        for _ in range(3):
            pool.capture(agent)
        self.assertIs(pool.anchor, anchor)
        self.assertEqual(len(pool.snapshots), 1)
        agent.state_dim = 1
        self.assertEqual(anchor(self.state, self.legal), expected)

    def test_practice_keeps_early_pressure_decision(self):
        env = ZhengShangYouEnv(seed=31, reward_scheme="win")
        env.game = self.game
        env.game.hands = [hand[:6] for hand in self.game.hands]
        pool = EndgamePool()
        pool.consider(env, env.get_legal_moves())
        earliest = features.state_vector(pool.candidate, 0).copy()
        env.step(env.get_legal_moves()[0])
        if not env.done:
            pool.consider(env, env.get_legal_moves())
        np.testing.assert_array_equal(
            features.state_vector(pool.candidate, 0), earliest
        )

    def test_history_is_public_padded_bounded_and_reset(self):
        self.assertFalse(features.history_vector(self.game, 0).any())
        self.game.apply_move(0, self.legal[0])
        self.game.apply_move(1, PASS_MOVE)
        rows = features.history_vector(self.game, 0).reshape(4, -1)
        np.testing.assert_array_equal(rows[:2], 0)
        self.assertEqual(rows[-1, 1], 1)
        self.assertEqual(rows[-1, self.game.num_players], 1)
        self.assertEqual(rows[-1, -1], 1)
        original = features.state_vector(self.game, 0, history=True)
        self.game.hands[2][0], self.game.hands[3][0] = (
            self.game.hands[3][0],
            self.game.hands[2][0],
        )
        np.testing.assert_array_equal(
            original, features.state_vector(self.game, 0, history=True)
        )
        for _ in range(6):
            self.game.apply_move(self.game.current_player, self.game.legal_moves()[0])
        self.assertEqual(len(self.game.action_history), 4)
        self.assertFalse(features.history_vector(Game(seed=31), 0).any())

    def test_history_is_invariant_to_seat_rotation(self):
        self.game.apply_move(0, self.legal[0])
        self.game.apply_move(1, PASS_MOVE)
        rotated = copy.deepcopy(self.game)
        rotated.hands = self.game.hands[-1:] + self.game.hands[:-1]
        rotated.table_owner = (self.game.table_owner + 1) % 4
        rotated.action_history = type(rotated.action_history)(
            (((seat + 1) % 4, move) for seat, move in self.game.action_history),
            maxlen=4,
        )
        for seat in range(4):
            np.testing.assert_array_equal(
                features.state_vector(self.game, seat, history=True),
                features.state_vector(rotated, (seat + 1) % 4, history=True),
            )

    def test_history_checkpoints_round_trip_and_switch_back_to_old_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            for history in (True, False):
                for partition in (True, False):
                    source = DDQNAgent(
                        *features.feature_dims(4),
                        history_features=history,
                        partition_features=partition,
                        seed=4,
                        device="cpu",
                    )
                    path = str(Path(directory) / "model.pt")
                    source.save(path)
                    restored = DDQNAgent(
                        *features.feature_dims(4),
                        history_features=not history,
                        device="cpu",
                    )
                    restored.load(path)
                    self.assertEqual(restored.history_features, history)
                    self.assertEqual(restored.partition_features, partition)
                    obs = features.state_for(self.game, 0, restored)
                    x = [source._phi(obs, move) for move in self.legal]
                    torch.testing.assert_close(
                        source._q_batch(x, source.policy_net),
                        restored._q_batch(x, restored.policy_net),
                    )
                    pool = SelfPlayPool(4, 2)
                    pool.capture(restored)
                    np.testing.assert_array_equal(
                        features.state_for(self.game, 0, pool.snapshots[0]), obs
                    )
                    incompatible = DDQNAgent(*features.feature_dims(3), device="cpu")
                    with self.assertRaises(ValueError):
                        incompatible.load(path)

    def test_history_environment_and_replay_train_with_monte_carlo(self):
        env = ZhengShangYouEnv(
            num_players=2,
            num_decks=1,
            seed=5,
            reward_scheme="win",
            history_features=True,
        )
        agent = DDQNAgent(
            *features.feature_dims(2),
            history_features=True,
            target_method="monte-carlo",
            learning_starts=1,
            batch_size=1,
            seed=5,
            device="cpu",
        )
        state = env.reset()
        while not env.done:
            self.assertEqual(len(state), agent.state_dim)
            legal = env.get_legal_moves()
            action = agent.act(state, legal)
            nxt, reward, done, info = env.step(action)
            agent.observe(
                (state, action, reward, nxt, done, info["next_legal_moves"], legal)
            )
            state = nxt
        self.assertGreater(agent.learning_updates, 0)
        self.assertFalse(agent.pending)

    def test_practice_cache_stores_losses_and_restores_independent_games(self):
        env = ZhengShangYouEnv(seed=31, reward_scheme="win", history_features=True)
        env.game = self.game
        env.game.hands = [hand[:6] for hand in self.game.hands]
        pool = EndgamePool(seed=2, capacity=2)
        pool.consider(env, env.get_legal_moves())
        self.assertIsNotNone(pool.candidate)
        pool.finish(False)
        self.assertFalse(pool.positions)
        pool.consider(env, env.get_legal_moves())
        pool.finish(True)
        source = pool.sample(1)
        original = features.state_vector(source, 0, history=True)
        restored = env.reset_from(source)
        np.testing.assert_array_equal(restored, original)
        self.assertEqual(env.game.initial_hand_size, 27)
        env.step(env.get_legal_moves()[0])
        np.testing.assert_array_equal(
            features.state_vector(source, 0, history=True), original
        )
        self.assertIsNone(pool.sample(0))
        bad = copy.deepcopy(source)
        bad.finish_order = [1]
        with self.assertRaises(ValueError):
            env.reset_from(bad)


if __name__ == "__main__":
    unittest.main()
