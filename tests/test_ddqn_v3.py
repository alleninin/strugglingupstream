import copy
import tempfile
from pathlib import Path
import unittest

import numpy as np
import torch

from agents.ddqn_agent import DDQNAgent
from agents.dqn_agent import DQNAgent
from agents.qlearning import QLearningAgent
from agents.random_agent import RandomAgent
from agents.hand_q_network import HandQNetwork
from env.env import ZhengShangYouEnv
from bots.greedy_bot import GreedyBot
from env import features
from game.rules import Game
from training.train import evaluate_vs_opponent, curriculum_opponent
from unittest.mock import patch


class DDQNV3Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.game = Game(seed=42)
        self.state = features.state_vector(self.game, 0)
        self.legal = self.game.legal_moves()

    def test_public_history_changes_observation_without_revealing_hidden_hands(self):
        original = self.state.copy()
        self.game.played_cards = [self.game.hands[1][0]]
        changed = features.state_vector(self.game, 0)
        old_dim = features.legacy_state_dim(len(original))
        np.testing.assert_array_equal(original[:old_dim], changed[:old_dim])
        self.assertFalse(np.array_equal(original, changed))
        self.game.hands[1], self.game.hands[2] = self.game.hands[2], self.game.hands[1]
        np.testing.assert_array_equal(changed, features.state_vector(self.game, 0))

    def test_played_cards_record_actual_plays_and_reset_on_new_deal(self):
        move = self.legal[0]
        self.game.apply_move(0, move)
        self.assertEqual(self.game.played_cards, list(move.cards))
        self.assertEqual(Game(seed=42).played_cards, [])

    def test_greedy_ignores_appended_history(self):
        bot = GreedyBot()
        old_dim = features.legacy_state_dim(len(self.state))
        full = self.state.copy()
        full[old_dim:] = .01  # Must not be read as tiny opponent hands.
        self.assertEqual(bot.act(full, self.legal), bot.act(full[:old_dim], self.legal))

    def test_three_step_returns_discount_and_flush_last_transition(self):
        agent = DDQNAgent(*features.feature_dims(4), n_step=3, gamma=.5, learning_starts=1000)
        for i, reward in enumerate((1., 2., 4., 8.)):
            done = i == 3
            agent.observe((self.state, self.legal[0], reward,
                           None if done else self.state, done,
                           [] if done else self.legal, self.legal))
        items = agent.buffer.tree.data[:4]
        self.assertEqual([item.reward for item in items], [3., 6., 8., 8.])
        self.assertEqual([item.discount for item in items], [.125, .125, .25, .5])
        self.assertIsNotNone(items[0].next_moves)
        self.assertTrue(all(item.next_moves is None for item in items[1:]))
        self.assertFalse(agent.pending)

    def test_scalar_double_dqn_uses_online_choice_not_target_max(self):
        agent = DDQNAgent(*features.feature_dims(4), n_step=1, learning_starts=1000)
        agent.observe((self.state, self.legal[0], 2., self.state, False,
                       self.legal[:2], self.legal))
        item = agent.buffer.tree.data[0]._replace(discount=.25)
        def scores(net, *args):
            return torch.tensor([3., 1.])
        def target(inputs):
            chosen_first = (inputs[:, agent.state_dim:] == agent._t(item.next_moves[0])).all(dim=1)
            return torch.where(chosen_first, -2., 9.).unsqueeze(-1)
        with patch.object(agent, '_scores', side_effect=scores), \
             patch.object(agent.target_net, 'forward', side_effect=target):
            self.assertAlmostEqual(float(agent._targets([item])[0]), 1.5)

    def test_update_schedule_counts_gradient_steps_and_syncs_target(self):
        agent = DDQNAgent(*features.feature_dims(4), batch_size=1, learning_starts=4,
                          train_every=2, target_update=2, seed=1)
        for _ in range(6):
            agent.observe((self.state, self.legal[0], 1., None, True, [], self.legal))
        self.assertEqual(agent.learning_updates, 2)
        for online, target in zip(agent.policy_net.parameters(), agent.target_net.parameters()):
            torch.testing.assert_close(online, target)

    def test_legacy_checkpoints_ignore_appended_features_and_preserve_predictions(self):
        s_dim, a_dim = features.feature_dims(4)
        old_dim = features.legacy_state_dim(s_dim)
        for factory in (DDQNAgent, DQNAgent, QLearningAgent):
            options = {'dueling': True} if factory is DDQNAgent else {}
            old = factory(old_dim, a_dim, epsilon=0, seed=7, **options)
            expected = old.act(self.state[:old_dim], self.legal)
            with tempfile.TemporaryDirectory() as directory:
                path = str(Path(directory) / 'weights')
                old.save(path)
                restored = factory(s_dim, a_dim, epsilon=0, seed=9)
                restored.load(path)
            self.assertEqual(restored.state_dim, old_dim)
            self.assertEqual(restored.act(self.state, self.legal), expected)

    def test_evaluation_does_not_change_future_exploration(self):
        agent = DDQNAgent(*features.feature_dims(2), seed=7)
        before = copy.deepcopy(agent.rng.bit_generator.state)
        evaluate_vs_opponent(agent, 2, 1, seed=13, n=2, opponent='random')
        self.assertEqual(agent.rng.bit_generator.state, before)

    def test_random_baseline_rng_is_restored_too(self):
        agent = RandomAgent(seed=17)
        before = agent.rng.getstate()
        evaluate_vs_opponent(agent, 2, 1, seed=13, n=2, opponent='random')
        self.assertEqual(agent.rng.getstate(), before)

    def test_curriculum_finishes_against_full_greedy_lineup(self):
        self.assertEqual(curriculum_opponent(0, 100), 'random')
        self.assertEqual(curriculum_opponent(20, 100), 'mixed')
        self.assertEqual(curriculum_opponent(50, 100), 'greedy')
        self.assertEqual(curriculum_opponent(99, 100), 'greedy')

    def test_hand_features_respect_jokers_and_afterstate(self):
        s_dim, a_dim = features.feature_dims(4)
        net = HandQNetwork(s_dim, a_dim)
        x = torch.zeros((1, s_dim + a_dim))
        x[0, 0] = 4  # Four 3s before the play.
        x[0, 13] = 2  # Two black jokers are two singles, never a pair.
        x[0, s_dim] = 1  # Split the four 3s by playing a single.
        encoded = net.encode(x)
        torch.testing.assert_close(encoded[0, -26:-11], torch.tensor([3.] + [0.] * 12 + [2., 0.]) / 4)
        stats = encoded[0, -11:]
        self.assertAlmostEqual(float(stats[2]), 2 / 15)
        self.assertAlmostEqual(float(stats[3]), 1 / 15)
        self.assertAlmostEqual(float(stats[5]), 0.)
        self.assertAlmostEqual(float(stats[10]), 1 / 15)

    def test_planning_checkpoint_restores_network_and_predictions(self):
        agent = DDQNAgent(*features.feature_dims(4), planning_features=True, epsilon=0, seed=2)
        expected = agent.act(self.state, self.legal)
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / 'hand.pt')
            agent.save(path)
            restored = DDQNAgent(*features.feature_dims(4), epsilon=0)
            restored.load(path)
        self.assertTrue(restored.planning_features)
        self.assertEqual(restored.act(self.state, self.legal), expected)

    def test_discounted_potential_matches_discounted_terminal_objective(self):
        gamma = .9
        env = ZhengShangYouEnv(num_players=2, num_decks=1, seed=3,
                              reward_scheme='win', reward_discount=gamma)
        state = env.reset()
        phi0 = env._potential()
        total, discount = 0., 1.
        bot = RandomAgent(seed=3)
        while not env.done:
            state, reward, done, _ = env.step(bot.act(state, env.get_legal_moves()))
            total += discount * reward
            if not done:
                discount *= gamma
        terminal = 1. if env.game.finish_order[0] == 0 else -1.
        self.assertAlmostEqual(total, discount * terminal - phi0)


if __name__ == '__main__':
    unittest.main()
