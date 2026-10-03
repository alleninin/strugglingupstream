"""Seeded tournaments with matched deals and complete seat rotations."""
import argparse
from itertools import combinations
import json
import math
import os
from pathlib import Path
import random

from game.rules import Game
from env import features
from agents.qlearning import QLearningAgent
from agents.dqn_agent import DQNAgent
from agents.ddqn_agent import DDQNAgent
from agents.random_agent import RandomAgent
from bots.greedy_bot import GreedyBot


def simulate_game(agents, num_players, num_decks, seed):
    game = Game(num_players=num_players, num_decks=num_decks, seed=seed)
    for seat, agent in enumerate(agents):
        agent.reset_episode()
        if isinstance(agent, RandomAgent):
            agent.rng.seed(seed * num_players + seat)
    while not game.done:
        seat = game.current_player
        legal = game.legal_moves(seat)
        move = agents[seat].act(features.state_vector(game, seat), legal)
        # Invalid actions must fail visibly; replacing them would hide bot bugs.
        game.apply_move(seat, move)
    return game.finish_order


def load_checkpoint(factory, path, *args, **kwargs):
    if not os.path.exists(path):
        print(f"note: {path} not found; skipping checkpoint")
        return None
    try:
        agent = factory(*args, **kwargs)
        agent.load(path)
    except Exception as error:
        print(f"warning: skipping {path}: {error}")
        return None
    agent.epsilon = 0.0
    return agent


def summarize(records, names):
    placements = {name: [] for name in names}
    for record in records:
        for place, seat in enumerate(record['finish_order'], 1):
            placements[record['seats'][seat]].append(place)
    return {name: {
        'appearances': len(values), 'wins': values.count(1),
        'win_rate': values.count(1) / len(values) if values else None,
        'mean_place': sum(values) / len(values) if values else None,
    } for name, values in placements.items()}


def run_tournament(pool, num_players=4, num_decks=2, seed=100000,
                   games=300, rounds=None, progress_every=0):
    """Each lineup plays the same shuffled deal once in every seat rotation.

    With rounds set, cover every player subset in each round. Otherwise use a
    shuffled subset schedule and round games up to a full seat-rotation block.
    Only legal observations are passed to agents; no hidden-hand lookahead.
    """
    names = list(pool)
    if not names or num_players < 2 or games < 1 or (rounds is not None and rounds < 1):
        raise ValueError('need agents, at least two players, and a positive game/round count')
    if len(names) >= num_players:
        lineups = list(combinations(names, num_players))
    else:
        lineups = [tuple(names[i % len(names)] for i in range(num_players))]
    blocks = len(lineups) * rounds if rounds is not None else math.ceil(games / num_players)
    rng = random.Random(seed)
    schedule = []
    while len(schedule) < blocks:
        cycle = lineups.copy()
        rng.shuffle(cycle)
        schedule.extend(cycle)
    old_eps = {name: getattr(agent, 'epsilon', 0.) for name, agent in pool.items()}
    records = []
    try:
        for agent in pool.values():
            agent.epsilon = 0.
        for block, lineup in enumerate(schedule[:blocks]):
            order = list(lineup)
            rng.shuffle(order)
            deal_seed = seed + block
            for rotation in range(num_players):
                seats = order[rotation:] + order[:rotation]
                finish = simulate_game([pool[name] for name in seats], num_players, num_decks, deal_seed)
                records.append({'seed': deal_seed, 'seats': seats, 'finish_order': finish})
                if progress_every and len(records) % progress_every == 0:
                    print(f'[tournament {len(records)}/{blocks * num_players}]', flush=True)
    finally:
        for name, agent in pool.items():
            agent.epsilon = old_eps[name]
    return {'num_games': len(records), 'num_players': num_players, 'num_decks': num_decks,
            'seed': seed, 'rounds': rounds, 'results': summarize(records, names), 'games': records}


def print_results(report):
    print(f"Tournament: {report['num_games']} deals, {report['num_players']} players, "
          f"{report['num_decks']} deck(s); matched deals with rotated seats")
    print(f"{'Agent':<14}{'Games':>8}{'WinRate(1st)':>14}{'AvgPlace':>12}")
    rows = sorted(report['results'].items(), key=lambda item: item[1]['mean_place'] or float('inf'))
    for name, row in rows:
        win = row['win_rate'] if row['win_rate'] is not None else float('nan')
        place = row['mean_place'] if row['mean_place'] is not None else float('nan')
        print(f"{name:<14}{row['appearances']:>8}{win:>14.3f}{place:>12.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--num-players', type=int, default=4)
    ap.add_argument('--num-decks', type=int, default=2)
    ap.add_argument('--games', type=int, default=300, help='rounded up to a complete seat rotation')
    ap.add_argument('--rounds', type=int, help='complete rounds over every player subset; overrides --games')
    ap.add_argument('--seed', type=int, default=100000)
    ap.add_argument('--dqn-path', default='checkpoints/dqn_agent.pt')
    ap.add_argument('--ddqn-path', default='checkpoints/ddqn_agent.pt')
    ap.add_argument('--q-path', default='checkpoints/qlearning_agent.npy')
    ap.add_argument('--shaped-path', default='checkpoints/shaped_agent.pt')
    ap.add_argument('--shaped-arch', choices=['ddqn', 'qlearning'], default='ddqn')
    ap.add_argument('--shaped-ql-path', default='checkpoints/shaped-ql_agent.npy')
    ap.add_argument('--json', help='save configuration, aggregate results, and every game result')
    args = ap.parse_args()
    s_dim, a_dim = features.feature_dims(args.num_players, args.num_decks)
    from bots.shaped_reward_bot import ShapedRewardBot
    specs = [('QLearning', QLearningAgent, args.q_path, {}),
             ('DQN', DQNAgent, args.dqn_path, {}),
             ('DDQN', DDQNAgent, args.ddqn_path, {}),
             ('Shaped', ShapedRewardBot, args.shaped_path, {'agent_type': args.shaped_arch}),
             ('ShapedQL', ShapedRewardBot, args.shaped_ql_path, {'agent_type': 'qlearning'})]
    pool = {}
    for name, factory, path, kwargs in specs:
        agent = load_checkpoint(factory, path, s_dim, a_dim, seed=args.seed, **kwargs)
        if agent is not None:
            pool[name] = agent
    pool['Greedy'] = GreedyBot(num_players=args.num_players, num_decks=args.num_decks)
    pool['Random'] = RandomAgent(seed=args.seed)
    report = run_tournament(pool, args.num_players, args.num_decks, args.seed, args.games, args.rounds)
    print_results(report)
    if args.json:
        destination = Path(args.json)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
