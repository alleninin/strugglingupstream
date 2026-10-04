import unittest
from unittest.mock import patch
import contextlib
import io
import tempfile
from pathlib import Path

import torch

from env.env import ZhengShangYouEnv
from game.cards import build_deck
from game.rules import Game
from bots.random_bot import RandomAgent
from archive.legacy.train import exploration_epsilon, evaluate_vs_opponent, train
from archive.legacy.demonstrations import Demonstrations, Example
from agents.ddqn_agent import DDQNAgent
from archive.legacy.dqn_agent import DQNAgent
from archive.legacy.qlearning import QLearningAgent
from env import features


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
        env = ZhengShangYouEnv(num_players=3, reward_scheme="win",
                              opponent_policies=[must_not_act] * 2)
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
        env = ZhengShangYouEnv(num_players=3, reward_scheme="win",
                              opponent_policies=[lambda obs, legal: legal[0]] * 2)
        env.game = game([3, 8], [4], [5, 6])
        state, reward, done, info = env.step(env.get_legal_moves()[0])
        self.assertTrue(done)
        self.assertEqual(env.game.finish_order, [1])
        self.assertEqual(reward, 0.0)  # -1 outcome plus terminal potential correction.
        self.assertIsNone(state)
        self.assertEqual(info["next_legal_moves"], [])

    def test_shaping_return_depends_on_outcome_not_cards_or_duration(self):
        for seed in range(10):
            env = ZhengShangYouEnv(num_players=3, num_decks=1, seed=seed,
                                  reward_scheme="win")
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
        self.assertEqual(exploration_epsilon(0, 2500), .5)
        self.assertGreater(exploration_epsilon(500, 2500), .35)
        self.assertGreater(exploration_epsilon(1500, 2500), .15)
        self.assertAlmostEqual(exploration_epsilon(2500, 2500), .05)

    def test_demonstrations_fit_all_three_agents_and_sync_neural_targets(self):
        torch.set_num_threads(1)
        current = game([3, 8], [4, 9])
        state = features.state_vector(current, 0)
        legal = current.legal_moves()
        # Deliberately teach the last move, avoiding accidental argmax tie success.
        examples = [Example(state, legal, len(legal) - 1)]
        for factory in (DDQNAgent, DQNAgent, QLearningAgent):
            agent = factory(*features.feature_dims(2), seed=42, epsilon=0)
            demos = Demonstrations(examples, seed=1, batch_size=2)
            demos.pretrain(agent, 30)
            self.assertEqual(agent.act(state, legal), legal[-1])
            if hasattr(agent, "target_net"):
                for policy, target in zip(agent.policy_net.parameters(), agent.target_net.parameters()):
                    torch.testing.assert_close(policy, target)

    def test_rehearsal_learns_legal_labels_with_ragged_action_sets(self):
        first = game([3, 8], [4, 9])
        second = game([3, 3, 7], [4, 9])
        examples = [Example(features.state_vector(g, 0), g.legal_moves(), 0)
                    for g in (first, second)]
        for factory in (DDQNAgent, DQNAgent):
            agent = factory(*features.feature_dims(2), seed=3)
            demos = Demonstrations(examples, seed=4)
            loss = demos.loss(agent)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                                for p in agent.policy_net.parameters()))

    def test_empty_demonstrations_rejected(self):
        with self.assertRaises(ValueError):
            Demonstrations([])

    def test_evaluation_uses_separate_deals_and_restores_exploration(self):
        agent = QLearningAgent(*features.feature_dims(2), epsilon=.37, seed=0)
        with patch("archive.legacy.train.ZhengShangYouEnv", wraps=ZhengShangYouEnv) as env:
            evaluate_vs_opponent(agent, 2, 1, seed=0, n=2, opponent="random")
        seeds = [call.kwargs["seed"] for call in env.call_args_list]
        self.assertEqual(len(set(seeds)), 2)
        self.assertTrue(all(seed >= (1 << 62) for seed in seeds))
        self.assertEqual(agent.epsilon, .37)

    def test_best_checkpoint_survives_later_regression(self):
        saved = []
        original = QLearningAgent.save
        def save(agent, path):
            saved.append(path)
            original(agent, path)
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "q.npy")
            best = str(Path(directory) / "q.best.npy")
            with patch("archive.legacy.train.evaluate_vs_opponent", side_effect=[.8, .1]), \
                 patch.object(QLearningAgent, "save", save), contextlib.redirect_stdout(io.StringIO()):
                train("qlearning", 2, 2, 1, 0, path, 1, demo_games=0)
            self.assertTrue(Path(best).exists())
            self.assertEqual(saved.count(best), 1)
            self.assertEqual(saved[-1], path)


if __name__ == "__main__":
    unittest.main()
