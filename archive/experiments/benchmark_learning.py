"""Reproduce learning comparisons, with an independent final evaluation set."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--agent', choices=['ddqn', 'dqn', 'qlearning'], required=True)
    parser.add_argument('--episodes', type=int, default=2500)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--demo-games', type=int, default=200)
    parser.add_argument('--demo-updates', type=int, default=1000)
    parser.add_argument('--eval-games', type=int, default=300)
    parser.add_argument('--baseline', type=Path, help='directory of pre-change source snapshots')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    from archive.legacy.train import evaluate_vs_opponent
    if args.baseline:
        # Load the exact saved environment, agent and training loop together.
        load_module('env.env', args.baseline / 'env_before.py')
        module_name = 'qlearning' if args.agent == 'qlearning' else args.agent + '_agent'
        load_module('agents.' + module_name, args.baseline / (args.agent + '_before.py'))
        train = load_module('training._reference', args.baseline / 'train_before.py').train
        options = {}
    else:
        from archive.legacy.train import train
        options = dict(demo_games=args.demo_games, demo_updates=args.demo_updates)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    suffix = '.npy' if args.agent == 'qlearning' else '.pt'
    checkpoint = args.output.with_suffix(suffix)
    agent = train(args.agent, args.episodes, 4, 2, args.seed, str(checkpoint),
                  eval_every=500, progress_every=500, eval_games=100, **options)
    # The independent evaluation helper was imported before any baseline module
    # replacement and uses the same held-out deals for every configuration.
    results = {opponent: evaluate_vs_opponent(agent, 4, 2, seed=913,
                n=args.eval_games, opponent=opponent) for opponent in ('greedy', 'random')}
    result = dict(agent=args.agent, episodes=args.episodes, seed=args.seed,
                  demo_games=0 if args.baseline else args.demo_games,
                  demo_updates=0 if args.baseline else args.demo_updates,
                  eval_games=args.eval_games, win_rates=results, checkpoint=str(checkpoint))
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
