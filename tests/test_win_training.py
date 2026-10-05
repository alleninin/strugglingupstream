import contextlib
import io
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from agents.ddqn_agent import DDQNAgent
from bots.random_bot import RandomAgent
from env import features
from env.env import ZhengShangYouEnv
from game.cards import build_deck
from game.rules import Game
from training.train import (
    evaluate_vs_opponent,
    exploration_epsilon,
    train,
    training_action,
)


def game(*ranks):
    deck = build_deck(2)
    hands = []
    for values in ranks:
        hand = []
        for rank in values:
            card = next(c for c in deck if c.rank == rank)
            deck.remove(card)
            hand.append(card)
        hands.append(hand)
    return Game(num_players=len(hands), hands=hands)


class WinTrainingTests(unittest.TestCase):
    def test_win_stops_without_rolling_remaining_players(self):
        def must_not_act(*args):
            self.fail("opponents acted after first place was decided")

        env = ZhengShangYouEnv(
            num_players=3, reward_scheme="win", opponent_policies=[must_not_act] * 2
        )
        env.game = game([3], [4, 5], [6, 7])
        state, reward, done, info = env.step(env.get_legal_moves()[0])
        self.assertTrue(done)
        self.assertFalse(env.game.done)
        self.assertEqual(env.game.finish_order, [0])
        self.assertIsNone(state)
        self.assertEqual(info["next_legal_moves"], [])
        self.assertEqual(env.get_legal_moves(), [])
        self.assertEqual(reward, 1.5)
        with self.assertRaises(RuntimeError):
            env.step(None)

    def test_loss_stops_at_first_opponent_finish(self):
        env = ZhengShangYouEnv(
            num_players=3,
            reward_scheme="win",
            opponent_policies=[lambda obs, legal: legal[0]] * 2,
        )
        env.game = game([3, 8], [4], [5, 6])
        state, reward, done, info = env.step(env.get_legal_moves()[0])
        self.assertTrue(done)
        self.assertEqual(env.game.finish_order, [1])
        self.assertEqual(reward, 0.0)
        self.assertIsNone(state)
        self.assertEqual(info["next_legal_moves"], [])

    def test_shaping_return_depends_on_outcome_not_cards_or_duration(self):
        for seed in range(10):
            env = ZhengShangYouEnv(
                num_players=3, num_decks=1, seed=seed, reward_scheme="win"
            )
            state = env.reset()
            initial_potential = env._potential()
            bot = RandomAgent(seed=seed)
            total = 0.0
            while not env.done:
                state, reward, done, _ = env.step(bot.act(state, env.get_legal_moves()))
                total += reward
            outcome = 1.0 if env.game.finish_order[0] == 0 else -1.0
            self.assertAlmostEqual(total, outcome - initial_potential)

    def test_exploration_lasts_through_training(self):
        self.assertEqual(exploration_epsilon(0, 2500), 0.5)
        self.assertGreater(exploration_epsilon(500, 2500), 0.35)
        self.assertGreater(exploration_epsilon(1500, 2500), 0.15)
        self.assertAlmostEqual(exploration_epsilon(2500, 2500), 0.05)

    def test_evaluation_uses_separate_deals_and_restores_exploration(self):
        agent = DDQNAgent(*features.feature_dims(2), epsilon=0.37, seed=0)
        with patch("training.train.ZhengShangYouEnv", wraps=ZhengShangYouEnv) as env:
            evaluate_vs_opponent(agent, 2, 1, seed=0, n=2, opponent="random")
        seeds = [call.kwargs["seed"] for call in env.call_args_list]
        self.assertEqual(len(set(seeds)), 2)
        self.assertTrue(all(seed >= (1 << 62) for seed in seeds))
        self.assertEqual(agent.epsilon, 0.37)

    def test_evaluation_stops_as_soon_as_first_place_is_decided(self):
        environments = []

        def make_env(*args, **kwargs):
            env = ZhengShangYouEnv(*args, **kwargs)
            environments.append(env)
            return env

        with patch("training.train.ZhengShangYouEnv", side_effect=make_env):
            evaluate_vs_opponent(RandomAgent(seed=5), 3, 1, seed=9, n=2)
        for env in environments:
            self.assertEqual(len(env.game.finish_order), 1)
            self.assertFalse(env.game.done)
            self.assertTrue(env.done)

    def test_expert_exploration_preserves_total_epsilon_budget(self):
        rng = np.random.default_rng(91)
        agent = SimpleNamespace(epsilon=0.4, rng=rng)
        agent.act = lambda state, legal: (
            "random" if rng.random() < agent.epsilon else "network"
        )
        teacher = SimpleNamespace(act=lambda state, legal: "teacher")
        counts = Counter(
            training_action(agent, None, None, teacher, 0.5) for _ in range(40000)
        )
        for action, expected in [("teacher", 0.2), ("random", 0.2), ("network", 0.6)]:
            self.assertAlmostEqual(counts[action] / 40000, expected, delta=0.01)
        self.assertEqual(agent.epsilon, 0.4)

    def test_teacher_is_never_used_with_exploration_disabled(self):
        teacher = Mock()
        agent = SimpleNamespace(epsilon=0.0, act=Mock(return_value="network"))
        self.assertEqual(training_action(agent, None, None, teacher, 0.5), "network")
        teacher.act.assert_not_called()

    def test_training_action_restores_epsilon_on_failure(self):
        agent = SimpleNamespace(
            epsilon=0.4,
            rng=SimpleNamespace(random=lambda: 0.9),
            act=Mock(side_effect=RuntimeError("test")),
        )
        with self.assertRaises(RuntimeError):
            training_action(agent, None, None, Mock(), 0.5)
        self.assertEqual(agent.epsilon, 0.4)

    def test_best_checkpoint_survives_later_regression(self):
        saved = []
        original = DDQNAgent.save

        def save(agent, path):
            saved.append(path)
            original(agent, path)

        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "ddqn.pt")
            best = str(Path(directory) / "ddqn.best.pt")
            with (
                patch("training.train.evaluate_vs_opponent", side_effect=[0.8, 0.1]),
                patch.object(DDQNAgent, "save", save),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                train("ddqn", 2, 2, 1, 0, path, 1)
            self.assertTrue(Path(best).exists())
            self.assertEqual(saved.count(best), 1)
            self.assertEqual(saved[-1], path)


if __name__ == "__main__":
    unittest.main()
