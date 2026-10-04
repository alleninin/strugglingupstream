import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from archive.legacy.dqn_agent import DQNAgent
from archive.legacy.qlearning import QLearningAgent
from agents.hand_q_network import HandQNetwork
from env import features
from game.rules import Game
from archive.legacy.demonstrations import Demonstrations, Example


class SharedPlanningTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.dims = features.feature_dims(4)
        self.game = Game(seed=21)
        self.state = features.state_vector(self.game, 0)
        self.moves = self.game.legal_moves()

    def test_linear_partition_features_match_neural_features(self):
        agent = QLearningAgent(*self.dims, partition_features=True)
        net = HandQNetwork(*self.dims, partition_features=True)
        for move in self.moves:
            raw = features.combined_vector(self.state, features.move_vector(move))
            expected = net.encode(torch.from_numpy(raw[None]))[0, -5:].numpy()
            np.testing.assert_allclose(agent._phi(self.state, move)[-5:], expected, atol=1e-7)

    def test_dqn_and_linear_round_trip_new_and_old_models(self):
        for cls in (DQNAgent, QLearningAgent):
            for enabled in (False, True):
                agent = cls(*self.dims, partition_features=enabled, seed=3, epsilon=0)
                if cls is QLearningAgent:
                    agent.w = agent.rng.normal(size=agent.dim).astype(np.float32)
                expected = agent.act(self.state, self.moves)
                with tempfile.TemporaryDirectory() as directory:
                    path = str(Path(directory) / 'model')
                    agent.save(path)
                    restored = cls(*self.dims, partition_features=not enabled, epsilon=0)
                    restored.load(path)
                self.assertEqual(restored.partition_features, enabled)
                self.assertEqual(expected, restored.act(self.state, self.moves))
                if cls is QLearningAgent:
                    np.testing.assert_array_equal(agent.w, restored.w)
                else:
                    phis = [agent._phi(self.state, m) for m in self.moves]
                    torch.testing.assert_close(agent._q_batch(phis, agent.policy_net),
                                               restored._q_batch(phis, restored.policy_net))

    def test_old_additive_linear_checkpoint_loads_into_partition_constructor(self):
        for state_dim in (self.dims[0], features.legacy_state_dim(self.dims[0])):
            weights = np.arange(state_dim + self.dims[1], dtype=np.float32) / 100
            with tempfile.TemporaryDirectory() as directory:
                path = str(Path(directory) / 'old.npy')
                np.save(path, weights)
                agent = QLearningAgent(*self.dims, partition_features=True)
                agent.load(path)
            self.assertFalse(agent.partition_features)
            move = self.moves[0]
            raw = features.combined_vector(self.state[:state_dim], features.move_vector(move))
            self.assertAlmostEqual(agent.q(agent._phi(self.state, move)), float(weights @ raw), places=5)

    def test_imitation_and_rl_updates_work_with_both_representations(self):
        for cls in (DQNAgent, QLearningAgent):
            kwargs = {'batch_size': 1, 'device': 'cpu'} if cls is DQNAgent else {}
            agent = cls(*self.dims, partition_features=True, seed=7, **kwargs)
            demos = Demonstrations([Example(self.state, self.moves, 0)], batch_size=1)
            demos.pretrain(agent, 2)
            agent.demonstrations = demos
            before = (torch.cat([p.detach().flatten().clone() for p in agent.policy_net.parameters()])
                      if cls is DQNAgent else agent.w.copy())
            agent.observe((self.state, self.moves[0], 1., None, True, []))
            after = (torch.cat([p.detach().flatten() for p in agent.policy_net.parameters()]).numpy()
                     if cls is DQNAgent else agent.w)
            self.assertTrue(np.isfinite(after).all())
            self.assertFalse(np.array_equal(before, after))


if __name__ == '__main__':
    unittest.main()
