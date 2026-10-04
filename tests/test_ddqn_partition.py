import tempfile
import unittest
from pathlib import Path

import torch

from agents.ddqn_agent import DDQNAgent
from env import features
from env.env import ZhengShangYouEnv
from env.hand_structure import hand_structure
from game.rules import Game


class PartitionLearningTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_disjoint_straight_is_one_play_not_five_singles(self):
        counts = (1,) * 5 + (0,) * 10
        self.assertEqual(hand_structure(counts), (1, 0))
        self.assertEqual(hand_structure((0,) * 15), (0, 0))
        self.assertEqual(hand_structure((0,) * 13 + (2, 0)), (2, 2))

    def test_features_distinguish_breaking_straight_from_spare_single(self):
        agent = DDQNAgent(*features.feature_dims(4), partition_features=True, device='cpu')
        inputs = torch.zeros((2, agent.input_dim))
        inputs[:, :5] = 1  # 34567 straight, plus an isolated 2.
        inputs[:, 12] = 1
        inputs[0, agent.state_dim] = 1  # Break the straight by playing 3.
        inputs[1, agent.state_dim + 12] = 1  # Play the spare 2 instead.
        encoded = agent.policy_net.encode(inputs)
        self.assertAlmostEqual(float(encoded[0, -4]), 5 / 15)
        self.assertAlmostEqual(float(encoded[1, -4]), 1 / 15)
        # Engineered observation features must still allow weight gradients.
        agent.policy_net(inputs).sum().backward()
        self.assertTrue(all(p.grad is not None for p in agent.policy_net.parameters()))

    def test_new_and_old_hand_checkpoints_load_without_changing_scores(self):
        game = Game(seed=13)
        state = features.state_vector(game, 0)
        moves = game.legal_moves()
        for enriched in (False, True):
            source = DDQNAgent(*features.feature_dims(4), partition_features=enriched,
                               epsilon=0, seed=4, device='cpu')
            inputs = [source._phi(state, move) for move in moves]
            expected = source._q_batch(inputs, source.policy_net).detach()
            with tempfile.TemporaryDirectory() as directory:
                path = str(Path(directory) / 'agent.pt')
                source.save(path)
                restored = DDQNAgent(*features.feature_dims(4), partition_features=not enriched,
                                     epsilon=0, device='cpu')
                restored.load(path)
            self.assertEqual(restored.partition_features, enriched)
            torch.testing.assert_close(expected, restored._q_batch(inputs, restored.policy_net))
            self.assertEqual(source.act(state, moves), restored.act(state, moves))

    def test_partition_network_updates_through_replay(self):
        env = ZhengShangYouEnv(num_players=2, num_decks=1, seed=8, reward_scheme='win')
        state = env.reset()
        agent = DDQNAgent(*features.feature_dims(2), partition_features=True, seed=3,
                          batch_size=1, learning_starts=1, n_step=3, device='cpu')
        while not env.done:
            legal = env.get_legal_moves()
            action = agent.act(state, legal)
            next_state, reward, done, info = env.step(action)
            agent.observe((state, action, reward, next_state, done, info['next_legal_moves'], legal))
            state = next_state
        self.assertGreater(agent.learning_updates, 0)
        self.assertFalse(agent.pending)
        self.assertTrue(all(torch.isfinite(p).all() for p in agent.policy_net.parameters()))


if __name__ == '__main__':
    unittest.main()
