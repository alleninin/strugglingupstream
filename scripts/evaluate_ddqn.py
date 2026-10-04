"""Compare saved DDQN policies on identical held-out deals without retraining."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from agents.ddqn_agent import DDQNAgent
from bots.random_bot import RandomAgent
from bots.greedy_bot import GreedyBot
from env.features import feature_dims
from training.train import evaluate_vs_opponent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoints', nargs='*', type=Path)
    parser.add_argument('--games', type=int, default=500)
    parser.add_argument('--seed', type=int, default=54321)
    parser.add_argument('--include-baselines', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.games < 1:
        parser.error('--games must be positive')
    torch.set_num_threads(1)
    results = dict(games=args.games, seed=args.seed, num_players=4, num_decks=2, results={})
    policies = []
    for path in args.checkpoints:
        agent = DDQNAgent(*feature_dims(4, 2), seed=0, device='cpu')
        agent.load(str(path))
        policies.append((str(path), agent))
    if args.include_baselines:
        policies.extend([('greedy', GreedyBot(seed=0)), ('random', RandomAgent(seed=0))])
    for name, agent in policies:
        rates = {opponent: evaluate_vs_opponent(agent, 4, 2, args.seed,
                 n=args.games, opponent=opponent) for opponent in ('greedy', 'random')}
        results['results'][name] = rates
        print(name, json.dumps(rates), flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2) + '\n')


if __name__ == '__main__':
    main()
