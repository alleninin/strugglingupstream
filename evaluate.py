import argparse
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from game.rules import Game
from env import features
from agents.random_agent import RandomAgent
from agents.qlearning import QLearningAgent
from agents.dqn_agent import DQNAgent


def simulate_game(agents, num_players, num_decks, seed):
    g = Game(num_players=num_players, num_decks=num_decks, seed=seed)
    while not g.done:
        seat = g.current_player
        legal = g.legal_moves(seat)
        obs = features.state_vector(g, seat)
        move = agents[seat].act(obs, legal)
        if move not in legal:
            move = legal[0]
        g.apply_move(seat, move)
    return g.finish_order


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-players", type=int, default=3)
    ap.add_argument("--num-decks", type=int, default=1)
    ap.add_argument("--games", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dqn-path", default="checkpoints/dqn_agent.pt")
    ap.add_argument("--q-path", default="checkpoints/q_agent.npy")
    args = ap.parse_args()

    s_dim, a_dim = features.feature_dims(args.num_players, args.num_decks)

    random_agent = RandomAgent(seed=args.seed)
    q_agent = QLearningAgent(s_dim, a_dim, seed=args.seed)
    if os.path.exists(args.q_path):
        q_agent.load(args.q_path)
    dqn_agent = DQNAgent(s_dim, a_dim, seed=args.seed)
    if os.path.exists(args.dqn_path):
        dqn_agent.load(args.dqn_path)

    for ag in (q_agent, dqn_agent):
        ag.epsilon = 0.0

    pool = {"Random": random_agent, "QLearning": q_agent, "DQN": dqn_agent}
    names = list(pool.keys())
    placements = {n: [] for n in names}
    wins = {n: 0 for n in names}

    for g_i in range(args.games):
        order_names = [names[(i + g_i) % len(names)] for i in range(args.num_players)]
        agents = [pool[nm] for nm in order_names]
        finish = simulate_game(agents, args.num_players, args.num_decks,
                               seed=args.seed * 100 + g_i)
        for rank, seat in enumerate(finish):
            nm = order_names[seat]
            placements[nm].append(rank + 1)
            if rank == 0:
                wins[nm] += 1

    import numpy as np
    print(f"Tournament: {args.games} deals, {args.num_players} players, "
          f"{args.num_decks} deck(s)")
    print(f"{'Agent':<12}{'WinRate(1st)':>14}{'AvgPlace':>12}")
    for nm in names:
        wp = wins[nm] / args.games
        avg = float(np.mean(placements[nm])) if placements[nm] else float("nan")
        print(f"{nm:<12}{wp:>14.3f}{avg:>12.3f}")


if __name__ == "__main__":
    main()
