"""Retrain five learners and evaluate them on held-out, seat-balanced deals.

Example:
python3.12 -B scripts/benchmark_bots.py --episodes 1000 --rounds 3 \
  --output reports/greedy-v2-20261002 --checkpoints checkpoints/greedy-v2-20261002 \
  --legacy-source reports/greedy-v2-20261002/greedy_before.py
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from agents.dqn_agent import DQNAgent
from agents.ddqn_agent import DDQNAgent
from agents.qlearning import QLearningAgent
from agents.random_agent import RandomAgent
from bots.greedy_bot import GreedyBot
from bots.shaped_reward_bot import ShapedRewardBot
from env.features import feature_dims
from evaluate import load_checkpoint, print_results, run_tournament, simulate_game
from training.train import train

KINDS = {'qlearning': 'QLearning', 'dqn': 'DQN', 'ddqn': 'DDQN',
         'shaped': 'Shaped', 'shaped-ql': 'ShapedQL'}


def checkpoint_factory(kind):
    if kind == 'qlearning':
        return QLearningAgent, {}
    if kind == 'dqn':
        return DQNAgent, {}
    if kind == 'ddqn':
        return DDQNAgent, {}
    return ShapedRewardBot, {'agent_type': 'qlearning' if kind == 'shaped-ql' else 'ddqn'}


def fixed_opponents(agent, factory, deals, seed):
    """Play one focal agent against three fresh copies of the fixed opponent."""
    records = []
    for deal in range(deals):
        for seat in range(4):
            opponents = [factory(seed=seed + deal * 4 + i) for i in range(4)]
            opponents[seat] = agent
            finish = simulate_game(opponents, 4, 2, seed + deal)
            records.append({'seed': seed + deal, 'seat': seat, 'place': finish.index(seat) + 1})
    placements = [record['place'] for record in records]
    return {'appearances': len(records), 'wins': placements.count(1),
            'win_rate': placements.count(1) / len(records),
            'mean_place': sum(placements) / len(records), 'games': records}


def write_report(output, config, training, tournament, baselines):
    rows = sorted(tournament['results'].items(), key=lambda item: item[1]['mean_place'])
    lines = ['# Greedy strategy and retraining experiment', '',
             f"Five learners trained from scratch for **{config['episodes']} episodes each**, "
             'against a seeded mixture of greedy and random opponents (four players, two decks).', '',
             'Greedy and Random are fixed policies and need no training. GreedyBefore is the saved '
             'pre-change strategy. All evaluated learners were reloaded from the new checkpoints '
             'with exploration disabled.', '',
             '## Tournament', '',
             f"{tournament['num_games']} games; all four-player subsets, {config['rounds']} rounds, "
             'every lineup rotated through all four seats on the same deal. Starting seat is 0 '
             'and every participant takes that seat in each block. Shuffle seeds are held out '
             'from training. Results are descriptive; seat rotations share deals and are not '
             'independent trials.', '',
             '| Bot | Appearances | First place | Mean place |', '|---|---:|---:|---:|']
    for name, row in rows:
        lines.append(f"| {name} | {row['appearances']} | {row['win_rate']:.1%} | {row['mean_place']:.3f} |")
    lines += ['', '## Fixed-opponent comparisons', '',
              f"Each row uses {config['baseline_deals']} new deals × four focal seats against three "
              'copies of the specified opponent. All focal bots receive the same deal seeds.', '',
              '| Bot | vs 3 Greedy: first | vs 3 Greedy: mean | vs 3 Random: first | vs 3 Random: mean |',
              '|---|---:|---:|---:|---:|']
    for name in tournament['results']:
        greedy, random = baselines[name]['greedy'], baselines[name]['random']
        lines.append(f"| {name} | {greedy['win_rate']:.1%} | {greedy['mean_place']:.3f} | "
                     f"{random['win_rate']:.1%} | {random['mean_place']:.3f} |")
    if 'GreedyBefore' in baselines:
        comparison = baselines['Greedy']['previous_greedy']
        lines += ['', f"New Greedy against three previous Greedy bots: **{comparison['win_rate']:.1%}** "
                  f"first place, mean placement **{comparison['mean_place']:.3f}** "
                  f"over {comparison['appearances']} games."]
    lines += ['', '## Training and limits', '',
              '| Agent | Episodes | Seconds | Checkpoint |', '|---|---:|---:|---|']
    for name, row in training.items():
        lines.append(f"| {name} | {row['episodes']} | {row['seconds']:.1f} | `{row['checkpoint']}` |")
    lines += ['', 'This is one training seed and a finite training budget, not evidence of convergence '
              'or strength against human players. The small-hand solver minimizes the number of '
              'combinations without modeling opponent cards. Larger-hand estimates remain heuristic. '
              'Old checkpoints are preserved. No policy was selected using this held-out tournament.', '',
              'Machine-readable configurations, source hashes, training times, and every game result '
              'are in `experiment.json`, `tournament.json`, and `baselines.json`.', '']
    (output / 'REPORT.md').write_text('\n'.join(lines))


def train_one(kind, name, episodes, seed, checkpoint):
    torch.set_num_threads(1)
    print(f'\n=== Training {name}: {episodes} episodes ===', flush=True)
    started = time.perf_counter()
    agent = train(kind, episodes, 4, 2, seed, str(checkpoint), 0,
                  opponent='mixed', progress_every=100)
    inner = getattr(agent, 'inner', agent)
    finite = (all(torch.isfinite(p).all().item() for p in inner.policy_net.parameters())
              if hasattr(inner, 'policy_net') else np.isfinite(inner.w).all())
    if not finite:
        raise RuntimeError(f'{name} produced non-finite weights')
    return {'episodes': episodes, 'seconds': time.perf_counter() - started,
            'checkpoint': str(checkpoint), 'epsilon_at_end': agent.epsilon,
            'device': str(getattr(inner, 'device', 'cpu'))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episodes', type=int, default=1000)
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--baseline-deals', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--eval-seed', type=int, default=200000)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoints', type=Path, required=True)
    parser.add_argument('--legacy-source', type=Path)
    parser.add_argument('--skip-training', action='store_true')
    parser.add_argument('--workers', type=int, default=3, help='independent training processes, one CPU thread each')
    args = parser.parse_args()
    if min(args.episodes, args.rounds, args.baseline_deals, args.workers) < 1:
        parser.error('episodes, rounds, and baseline-deals must be positive')
    # Small MLP batches run more efficiently without eight-way CPU thread overhead.
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    args.checkpoints.mkdir(parents=True, exist_ok=True)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update({'num_players': 4, 'num_decks': 2, 'opponent': 'mixed',
                   'torch_threads': 1, 'torch_version': torch.__version__,
                   'source_sha256': {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                     for directory in ('agents', 'bots', 'env', 'game', 'training')
                                     for path in sorted((ROOT / directory).glob('*.py'))}})
    training = {}
    if args.skip_training and (args.output / 'experiment.json').exists():
        previous = json.loads((args.output / 'experiment.json').read_text())
        training = previous.get('training', {})
    checkpoints = {kind: args.checkpoints / f"{kind}_agent.{'npy' if kind in ('qlearning', 'shaped-ql') else 'pt'}"
                   for kind in KINDS}
    if not args.skip_training:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            jobs = {executor.submit(train_one, kind, name, args.episodes, args.seed, checkpoints[kind]): name
                    for kind, name in KINDS.items()}
            for job in as_completed(jobs):
                name = jobs[job]
                training[name] = job.result()
                (args.output / 'experiment.json').write_text(json.dumps({'config': config, 'training': training}, indent=2)+'\n')
    pool = {}
    dimensions = feature_dims(4, 2)
    for kind, name in KINDS.items():
        factory, kwargs = checkpoint_factory(kind)
        pool[name] = load_checkpoint(factory, str(checkpoints[kind]), *dimensions, seed=args.seed, **kwargs)
        if pool[name] is None:
            raise RuntimeError(f'missing required checkpoint: {checkpoints[kind]}')
    pool['Greedy'] = GreedyBot()
    pool['Random'] = RandomAgent(seed=args.seed)
    legacy = None
    if args.legacy_source:
        spec = importlib.util.spec_from_file_location('greedy_before', args.legacy_source)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        legacy = module.GreedyBot
        pool['GreedyBefore'] = legacy()
    print('\n=== Held-out mixed tournament ===', flush=True)
    tournament = run_tournament(pool, seed=args.eval_seed, rounds=args.rounds, progress_every=100)
    print_results(tournament)
    (args.output / 'tournament.json').write_text(json.dumps(tournament, indent=2)+'\n')
    baselines = {}
    for name, agent in pool.items():
        agent.epsilon = 0.
        baselines[name] = {}
        for label, factory in (('greedy', GreedyBot), ('random', RandomAgent)):
            result = fixed_opponents(agent, factory, args.baseline_deals, args.eval_seed + 10000)
            baselines[name][label] = result
            print(f"{name} vs 3 {label}: first={result['win_rate']:.3f}, "
                  f"place={result['mean_place']:.3f}, n={result['appearances']}", flush=True)
        (args.output / 'baselines.json').write_text(json.dumps(baselines, indent=2)+'\n')
    if legacy:
        result = fixed_opponents(pool['Greedy'], legacy, args.baseline_deals, args.eval_seed + 10000)
        baselines['Greedy']['previous_greedy'] = result
        print(f"Greedy vs 3 GreedyBefore: first={result['win_rate']:.3f}, "
              f"place={result['mean_place']:.3f}, n={result['appearances']}", flush=True)
        (args.output / 'baselines.json').write_text(json.dumps(baselines, indent=2)+'\n')
    write_report(args.output, config, training, tournament, baselines)
    print(f"\nReport: {args.output / 'REPORT.md'}", flush=True)


if __name__ == '__main__':
    main()
