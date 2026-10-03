"""Numerical equivalence of grouped DDQN values, targets, and gradients."""
from copy import deepcopy
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch

from agents.ddqn_agent import DDQNAgent, ReplayTransition
from agents.dueling_dqn import DuelingQNetwork
from env import features
from game.rules import Game


def replay_items():
    rng = np.random.default_rng(11)
    state_dim, move_dim = features.feature_dims(4)
    items = []
    for i, count in enumerate((1, 7, 2, 13, 4)):
        terminal = i in (0, 3)
        items.append(ReplayTransition(
            rng.normal(size=state_dim).astype(np.float32),
            rng.normal(size=(count, move_dim)).astype(np.float32),
            count - 1, float(i - 2),
            None if terminal else rng.normal(size=state_dim).astype(np.float32),
            None if terminal else rng.normal(size=(i + 1, move_dim)).astype(np.float32)))
    return items


def repeated_features(state, moves):
    return np.concatenate([np.repeat(state[None, :], len(moves), axis=0), moves], axis=1)


def reference_predictions(net, data):
    joined = np.concatenate([repeated_features(item.state, item.moves) for item in data])
    values, advantages = net(torch.from_numpy(joined))
    offset = 0
    result = []
    for item in data:
        group = advantages[offset:offset + len(item.moves), 0]
        result.append(values[offset + item.chosen, 0] + group[item.chosen] - group.mean())
        offset += len(item.moves)
    return torch.stack(result)


class DDQNBatchingTests(unittest.TestCase):
    def setUp(self):
        self.agent = DDQNAgent(*features.feature_dims(4), seed=7, device='cpu', dueling=True)
        self.data = replay_items()

    def test_grouped_values_and_gradients_match_repeated_state_reference(self):
        reference = deepcopy(self.agent.policy_net)
        expected = reference_predictions(reference, self.data)
        actual = self.agent._predictions(self.data)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
        targets = torch.tensor([1., -2., .3, .9, -1.])
        weights = torch.tensor([.4, .6, 1., .9, .5])
        (weights * (actual - targets).square()).mean().backward()
        (weights * (expected - targets).square()).mean().backward()
        for (name, parameter), (_, old_parameter) in zip(
                self.agent.policy_net.named_parameters(), reference.named_parameters()):
            self.assertIsNotNone(parameter.grad, name)
            torch.testing.assert_close(parameter.grad, old_parameter.grad, atol=2e-6, rtol=1e-4,
                                       msg=lambda message: f'{name}: {message}')

    def test_double_dqn_targets_match_reference_and_use_online_selection(self):
        torch.manual_seed(55)
        independent_target = DuelingQNetwork(*features.feature_dims(4))
        self.agent.target_net.load_state_dict(independent_target.state_dict())
        expected = []
        differs_from_target_max = False
        with torch.no_grad():
            for item in self.data:
                if item.next_moves is None:
                    expected.append(item.reward)
                    continue
                x = torch.from_numpy(repeated_features(item.next_state, item.next_moves))
                pv, pa = self.agent.policy_net(x)
                tv, ta = self.agent.target_net(x)
                online_q = (pv + pa - pa.mean()).flatten()
                target_q = (tv + ta - ta.mean()).flatten()
                chosen = int(online_q.argmax())
                differs_from_target_max |= chosen != int(target_q.argmax())
                expected.append(item.reward + self.agent.gamma * float(target_q[chosen]))
        actual = self.agent._targets(self.data)
        self.assertTrue(differs_from_target_max)
        self.assertFalse(actual.requires_grad)
        torch.testing.assert_close(actual, torch.tensor(expected), atol=1e-6, rtol=1e-5)

    def test_all_terminal_batch_does_not_evaluate_next_states(self):
        terminal = [item for item in self.data if item.next_moves is None]
        with patch.object(self.agent.policy_net, 'forward_grouped', side_effect=AssertionError('unexpected forward')):
            actual = self.agent._targets(terminal)
        torch.testing.assert_close(actual, torch.tensor([item.reward for item in terminal]))

    def test_grouped_argmax_masks_padding_and_keeps_first_tie(self):
        lengths = np.array([1, 3, 2], dtype=np.int64)
        offsets = np.array([0, 1, 4], dtype=np.int64)
        # Negative first group must not select a later group's positive score.
        values = torch.tensor([-5., 4., 4., 2., -8., -3.])
        torch.testing.assert_close(self.agent._best_indices(values, lengths, offsets),
                                   torch.tensor([0, 1, 5]))

    def test_state_encoder_runs_once_per_decision_not_per_move(self):
        seen = []
        hook = self.agent.policy_net.common.register_forward_pre_hook(
            lambda module, args: seen.append(args[0].shape[0]))
        try:
            self.agent._predictions(self.data)
        finally:
            hook.remove()
        self.assertEqual(seen, [len(self.data)])
        self.assertGreater(sum(len(item.moves) for item in self.data), len(self.data))

    def test_act_agrees_with_original_dueling_formula(self):
        self.agent.epsilon = 0.
        for seed in range(5):
            game = Game(seed=seed)
            state, legal = features.state_vector(game, 0), game.legal_moves()
            x = np.stack([self.agent._phi(state, move) for move in legal])
            with torch.no_grad():
                values, advantages = self.agent.policy_net(torch.from_numpy(x))
                expected = int((values + advantages - advantages.mean()).argmax())
            self.assertEqual(self.agent.act(state, legal), legal[expected])

    def test_replay_records_chosen_index_and_snapshots_state(self):
        game = Game(seed=42)
        state, legal = features.state_vector(game, 0), game.legal_moves()
        self.agent.observe((state, legal[-1], 1., state, False, legal, legal))
        item = self.agent.buffer.tree.data[0]
        self.assertEqual(item.chosen, len(legal) - 1)
        self.assertEqual(item.moves.shape, (len(legal), features.feature_dims(4)[1]))
        saved = state.copy()
        state[:] = 0
        np.testing.assert_array_equal(item.state, saved)
        np.testing.assert_array_equal(item.next_state, saved)

    def test_old_network_state_dict_loads_without_parameter_changes(self):
        game = Game(seed=42)
        state, legal = features.state_vector(game, 0), game.legal_moves()
        x = torch.from_numpy(np.stack([self.agent._phi(state, move) for move in legal]))
        with torch.no_grad():
            expected = self.agent.policy_net(x)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = str(Path(directory) / 'weights.pt')
            torch.save(self.agent.policy_net.state_dict(), checkpoint)
            restored = DDQNAgent(*features.feature_dims(4), seed=99, device='cpu')
            restored.load(checkpoint)
        with torch.no_grad():
            actual = restored.policy_net(x)
        for original, loaded in zip(expected, actual):
            torch.testing.assert_close(original, loaded)


if __name__ == '__main__':
    unittest.main()
