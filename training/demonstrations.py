"""Optional expert warm-up and rehearsal; no expert is consulted at inference."""
from dataclasses import dataclass, field

import numpy as np
import torch
from env import features
from env.env import ZhengShangYouEnv
from bots.greedy_bot import GreedyBot


@dataclass
class Example:
    state: np.ndarray
    moves: list
    chosen: int
    move_features: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        self.move_features = np.stack([features.move_vector(move) for move in self.moves])


class Demonstrations:
    def __init__(self, examples, seed=0, weight=0.1, batch_size=32):
        if not examples:
            raise ValueError("demonstrations require at least one non-forced decision")
        self.examples = examples
        self.rng = np.random.default_rng(seed)
        self.weight = weight
        self.batch_size = batch_size

    def sample(self):
        return [self.examples[i] for i in self.rng.integers(
            len(self.examples), size=self.batch_size)]

    def loss(self, agent):
        data = self.sample()
        lengths = np.array([len(item.moves) for item in data])
        offsets = np.cumsum(lengths) - lengths
        if hasattr(agent.policy_net, "forward_grouped"):
            moves = np.concatenate([item.move_features for item in data])
            owners = torch.as_tensor(np.repeat(np.arange(len(data)), lengths),
                                     dtype=torch.long, device=agent.device)
            _, scores = agent.policy_net.forward_grouped(
                torch.as_tensor(np.stack([item.state[:agent.state_dim] for item in data]), device=agent.device),
                torch.as_tensor(moves, device=agent.device), owners)
        else:
            phis = np.concatenate([np.concatenate((
                np.broadcast_to(item.state[:agent.state_dim], (len(item.moves), agent.state_dim)),
                item.move_features), axis=1) for item in data])
            scores = agent._q_batch(phis, agent.policy_net)
        columns = np.arange(max(lengths))
        indices = np.minimum(offsets[:, None] + columns, len(scores) - 1)
        logits = scores[torch.as_tensor(indices, device=agent.device)]
        mask = torch.as_tensor(columns >= lengths[:, None], device=agent.device)
        labels = torch.as_tensor([item.chosen for item in data], device=agent.device)
        return torch.nn.functional.cross_entropy(logits.masked_fill(mask, -torch.inf), labels)

    def update_linear(self, agent, weight=1.0):
        # A normalized multiclass perceptron warm-up fits legal move preferences.
        # The linear model has no optimizer/autograd; a margin update also avoids
        # an unbounded logit scale competing with its bounded RL returns.
        item = self.examples[int(self.rng.integers(len(self.examples)))]
        phis = np.stack([agent._phi(item.state, move) for move in item.moves])
        scores = phis @ agent.w
        scores[item.chosen] -= 0.2
        rival = int(scores.argmax())
        if rival != item.chosen:
            direction = phis[item.chosen] - phis[rival]
            margin = .2 - float(agent.w @ direction)
            agent.w += (weight * margin / max(1.0, float(direction @ direction))) * direction

    def pretrain(self, agent, updates):
        if hasattr(agent, "policy_net"):
            for _ in range(updates):
                loss = self.loss(agent)
                agent.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.policy_net.parameters(), 10.0)
                agent.optimizer.step()
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
        else:
            for _ in range(updates * self.batch_size):
                self.update_linear(agent)


def collect_demonstrations(games, num_players, num_decks, seed):
    teacher = GreedyBot(num_players=num_players, num_decks=num_decks, seed=seed)
    env = ZhengShangYouEnv(num_players=num_players, num_decks=num_decks,
                          reward_scheme="win")
    examples = []
    for episode in range(games):
        # Disjoint from both training and evaluation deals.
        deal_seed = None if seed is None else (1 << 61) + seed * 1000003 + episode
        state = env.reset(seed=deal_seed)
        while not env.done:
            legal = env.get_legal_moves()
            action = teacher.act(state, legal)
            if len(legal) > 1:
                examples.append(Example(state.copy(), legal, legal.index(action)))
            state, _, _, _ = env.step(action)
    return examples
