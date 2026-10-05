"""Compare saved DDQN policies on identical held-out deals without retraining."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from agents.runtime import load_agent
from bots.greedy_bot import GreedyBot
from bots.random_bot import RandomAgent
from training.train import evaluate_vs_opponent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", nargs="*", type=Path)
    parser.add_argument("--games", type=int, default=500)
    parser.add_argument("--seed", type=int, default=54321)
    parser.add_argument("--include-baselines", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.games < 1:
        parser.error("--games must be positive")
    if not args.checkpoints and not args.include_baselines:
        parser.error("provide a checkpoint or --include-baselines")
    torch.set_num_threads(1)
    results = dict(
        games=args.games, seed=args.seed, num_players=4, num_decks=2, results={}
    )
    policies = []
    for path in args.checkpoints:
        agent = load_agent("ddqn", path, seed=0)
        policies.append((str(path), agent))
    if args.include_baselines:
        policies.extend(
            [("greedy", GreedyBot(seed=0)), ("random", RandomAgent(seed=0))]
        )
    for name, agent in policies:
        rates = {
            opponent: evaluate_vs_opponent(
                agent, 4, 2, args.seed, n=args.games, opponent=opponent
            )
            for opponent in ("greedy", "random")
        }
        results["results"][name] = rates
        print(name, json.dumps(rates), flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
