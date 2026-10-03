"""Watch a trained agent (seat 0) play against 3 greedy bots, move by move.

Prints the full play-by-play for a single game (or the first game when running
several), then a win-rate / placement summary across all games.

Examples
--------
  python3 scripts/watch.py --seed 42
  python3 scripts/watch.py --agent ddqn --checkpoint checkpoints/ddqn_agent.pt --seed 7
  python3 scripts/watch.py --agent dqn --games 50        # summary over 50 games
  python3 scripts/watch.py --agent greedy               # greedy (seat 0) vs 3 greedy
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from game.rules import Game
from env import features
from bots.greedy_bot import GreedyBot


def load_seat0(kind, path, s_dim, a_dim, seed, num_players=4, num_decks=2):
    """Build the seat-0 agent and load its requested checkpoint."""
    if kind == "greedy":
        return GreedyBot(num_players=num_players, num_decks=num_decks, seed=seed), "Greedy"
    if kind == "random":
        from agents.random_agent import RandomAgent
        return RandomAgent(seed=seed), "Random"
    if kind == "qlearning":
        from agents.qlearning import QLearningAgent
        ag = QLearningAgent(s_dim, a_dim, seed=seed)
    elif kind == "dqn":
        from agents.dqn_agent import DQNAgent
        ag = DQNAgent(s_dim, a_dim, epsilon=0.0)
    elif kind == "ddqn":
        from agents.ddqn_agent import DDQNAgent
        ag = DDQNAgent(s_dim, a_dim, epsilon=0.0)
    elif kind == "shaped":
        from bots.shaped_reward_bot import ShapedRewardBot
        ag = ShapedRewardBot(s_dim, a_dim, agent_type="ddqn")
    else:
        raise SystemExit(f"unknown --agent {kind!r}")
    if not os.path.exists(path):
        raise SystemExit(f"checkpoint not found: {path}")
    ag.load(path)
    ag.epsilon = 0.0
    return ag, kind.upper()


def play_one(seat0_agent, num_players, num_decks, seed, verbose=True):
    g = Game(num_players=num_players, num_decks=num_decks, seed=seed)
    # seats 1..n-1 are greedy
    greedy = [GreedyBot(num_players=num_players, num_decks=num_decks,
                        seed=seed * 10 + s) for s in range(1, num_players)]
    agents = [seat0_agent] + greedy

    if verbose:
        print(f"\n=== Deal seed={seed} :: P0={type(seat0_agent).__name__}, "
              f"P1..P{num_players - 1}=Greedy ===")
        print(f"P0 hand: {' '.join(c.label for c in sorted(g.hands[0], key=lambda c: c.rank))}")

    turn = 0
    while not g.done:
        seat = g.current_player
        legal = g.legal_moves(seat)
        obs = features.state_vector(g, seat)
        move = agents[seat].act(obs, legal)
        lead = g.table_move is None
        tag = "leads " if lead else "follows"
        if verbose:
            tlabel = (f"  [table: {g.table_move} by P{g.table_owner}]"
                      if g.table_move is not None else "")
            print(f"  P{seat} {tag}: {move}  (hand {len(g.hands[seat]) - len(move.cards)}){tlabel}")
        g.apply_move(seat, move)
        turn += 1
        if turn > 5000:
            print("SAFETY STOP")
            break

    place = g.finish_order.index(0) + 1
    if verbose:
        print(f"  Finish order: {g.finish_order}  -> P0 placed {place}{['st','nd','rd','th'][min(place,4)-1]}")
    return place


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", choices=["ddqn", "dqn", "qlearning", "shaped", "greedy", "random"],
                    default="ddqn")
    ap.add_argument("--checkpoint", default=None,
                    help="trained agent checkpoint (defaults to checkpoints/<agent>_agent.pt/.npy)")
    ap.add_argument("--num-players", type=int, default=4)
    ap.add_argument("--num-decks", type=int, default=2)
    ap.add_argument("--games", type=int, default=1,
                    help="number of games to play (play-by-play only printed for the first)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.checkpoint is None:
        ext = "npy" if args.agent == "qlearning" else "pt"
        args.checkpoint = os.path.join("checkpoints", f"{args.agent}_agent.{ext}")

    s_dim, a_dim = features.feature_dims(args.num_players, args.num_decks)
    seat0, label = load_seat0(args.agent, args.checkpoint, s_dim, a_dim, args.seed,
                             args.num_players, args.num_decks)

    places = []
    for i in range(args.games):
        verbose = (args.games == 1) or (i == 0)
        p = play_one(seat0, args.num_players, args.num_decks,
                     args.seed * 1000 + i, verbose=verbose)
        places.append(p)

    if args.games > 1:
        dist = {p: places.count(p) for p in range(1, args.num_players + 1)}
        dist_str = "  ".join(f"place {p}={dist[p]}" for p in dist)
        print(f"\n{args.games} games (P0={label} vs {args.num_players-1} Greedy): "
              f"win%={places.count(1)/args.games:.3f}   {dist_str}")


if __name__ == "__main__":
    main()
