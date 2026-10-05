import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from agents.ddqn_agent import DDQNAgent
from bots.greedy_bot import GreedyBot
from env import features
from game.rules import Game
from training.supervision import Example, GreedySupervision
from training.train import train


class SupervisionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.agent = DDQNAgent(*features.feature_dims(4), seed=42, device="cpu")
        self.game = Game(seed=24)
        self.state = features.state_vector(self.game, 0)
        self.legal = self.game.legal_moves()
        self.teacher = GreedyBot()

    def test_margin_learns_teacher_preference_without_changing_target_or_rng(self):
        supervision = GreedySupervision(seed=42, batch_size=1)
        supervision.add(self.agent, self.state, self.legal, self.teacher)
        target = copy.deepcopy(self.agent.target_net.state_dict())
        rng = copy.deepcopy(self.agent.rng.bit_generator.state)
        initial = float(supervision.loss(self.agent).detach())
        for _ in range(50):
            self.agent.optimizer.zero_grad()
            loss = supervision.loss(self.agent)
            loss.backward()
            self.agent.optimizer.step()
        self.assertLess(float(supervision.loss(self.agent).detach()), initial)
        chosen = self.agent.act(self.state, self.legal, network=self.agent.policy_net)
        self.assertEqual(chosen, self.teacher.act(self.state, self.legal))
        self.assertEqual(self.agent.rng.bit_generator.state, rng)
        for name, value in target.items():
            torch.testing.assert_close(value, self.agent.target_net.state_dict()[name])

    def test_margin_compares_only_moves_from_the_same_state(self):
        supervision = GreedySupervision(batch_size=2)
        supervision.examples.extend(
            [
                Example(self.state.copy(), self.agent._move_vectors(self.legal[:2]), 0),
                Example(self.state.copy(), self.agent._move_vectors(self.legal[:3]), 2),
            ]
        )
        scores = torch.tensor([0.0, 0.3, 0.7, 0.1, 0.0], requires_grad=True)
        with (
            patch.object(supervision, "rng") as rng,
            patch.object(self.agent, "_scores", return_value=scores),
        ):
            rng.integers.return_value = [0, 1]
            loss = supervision.loss(self.agent)
        self.assertAlmostEqual(loss.item(), 0.6)
        loss.backward()
        torch.testing.assert_close(
            scores.grad, torch.tensor([-0.5, 0.5, 0.5, 0.0, -0.5])
        )

    def test_fade_clears_examples_and_disabled_collection_never_calls_teacher(self):
        supervision = GreedySupervision(weight=0.5, fraction=0.6)
        supervision.add(self.agent, self.state, self.legal, self.teacher)
        supervision.schedule(30, 101)
        self.assertAlmostEqual(supervision.weight, 0.25)
        supervision.schedule(60, 101)
        self.assertEqual(supervision.weight, 0)
        self.assertFalse(supervision.examples)
        supervision.add(self.agent, self.state, self.legal, None)
        self.assertFalse(supervision.examples)
        for episodes in (1, 2, 5, 100):
            supervision = GreedySupervision(fraction=1)
            supervision.schedule(episodes - 1, episodes)
            self.assertEqual(supervision.weight, 0)

    def test_bounded_examples_own_arrays_and_skip_forced_decisions(self):
        supervision = GreedySupervision(capacity=2)
        supervision.add(self.agent, self.state, self.legal[:1], None)
        self.assertFalse(supervision.examples)
        for _ in range(3):
            supervision.add(self.agent, self.state, self.legal, self.teacher)
        self.assertEqual(len(supervision.examples), 2)
        saved = supervision.examples[0][0].copy()
        self.state[:] = 0
        np.testing.assert_array_equal(supervision.examples[0][0], saved)

    def test_supervised_ddqn_updates_support_history_and_dueling(self):
        for options in ({"history_features": True}, {"dueling": True}):
            with self.subTest(options=options):
                agent = DDQNAgent(
                    *features.feature_dims(4),
                    batch_size=1,
                    learning_starts=1,
                    device="cpu",
                    seed=42,
                    **options,
                )
                state = features.state_for(self.game, 0, agent)
                agent.demonstrations = GreedySupervision(batch_size=1, seed=0)
                agent.demonstrations.add(agent, state, self.legal, self.teacher)
                agent.observe((state, self.legal[0], 1.0, None, True, [], self.legal))
                self.assertEqual(agent.learning_updates, 1)
                self.assertTrue(
                    all(torch.isfinite(p).all() for p in agent.policy_net.parameters())
                )
                agent.demonstrations.schedule(99, 100)
                with patch.object(
                    agent.demonstrations,
                    "loss",
                    side_effect=AssertionError("supervision after fade"),
                ):
                    agent._learn()
                self.assertEqual(agent.learning_updates, 2)

    def test_invalid_configuration_is_rejected(self):
        for options in (
            {"weight": -1},
            {"weight": float("nan")},
            {"weight": float("inf")},
            {"fraction": 0},
            {"fraction": 1.1},
            {"capacity": 0},
            {"batch_size": 0},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                GreedySupervision(**options)

    def test_training_fades_and_saved_policy_needs_no_teacher(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "agent.pt")
            agent = train(
                "ddqn",
                10,
                4,
                2,
                0,
                path,
                0,
                supervision_weight=0.5,
                learning_starts=1,
                train_every=1,
            )
            self.assertEqual(agent.demonstrations.weight, 0)
            self.assertFalse(agent.demonstrations.examples)
            self.assertGreater(agent.learning_updates, 0)
            restored = DDQNAgent(*features.feature_dims(4), epsilon=0, device="cpu")
            restored.load(path)
            self.assertIsNone(restored.demonstrations)
            self.assertEqual(
                restored.act(self.state, self.legal),
                agent.act(self.state, self.legal, network=agent.policy_net),
            )


if __name__ == "__main__":
    unittest.main()
