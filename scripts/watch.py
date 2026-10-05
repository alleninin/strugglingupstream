"""Watch DDQN, Greedy or Random play and summarize their placements."""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from agents.runtime import load_agent
from bots.greedy_bot import GreedyBot
from env import features
from game.rules import Game


def play_one(seat0_agent, num_players, num_decks, seed, verbose=True):
    g = Game(num_players=num_players, num_decks=num_decks, seed=seed)

    greedy = [
        GreedyBot(num_players=num_players, num_decks=num_decks, seed=seed * 10 + s)
        for s in range(1, num_players)
    ]
    agents = [seat0_agent] + greedy

    if verbose:
        print(
            f"\n=== Deal seed={seed} :: P0={type(seat0_agent).__name__}, "
            f"P1..P{num_players - 1}=Greedy ==="
        )
        print(
            f"P0 hand: {' '.join(c.label for c in sorted(g.hands[0], key=lambda c: c.rank))}"
        )

    while not g.done:
        seat = g.current_player
        legal = g.legal_moves(seat)
        obs = features.state_for(g, seat, agents[seat])
        move = agents[seat].act(obs, legal)
        lead = g.table_move is None
        tag = "leads " if lead else "follows"
        if verbose:
            tlabel = (
                f"  [table: {g.table_move} by P{g.table_owner}]"
                if g.table_move is not None
                else ""
            )
            print(
                f"  P{seat} {tag}: {move}  (hand {len(g.hands[seat]) - len(move.cards)}){tlabel}"
            )
        g.apply_move(seat, move)

    place = g.finish_order.index(0) + 1
    if verbose:
        print(
            f"  Finish order: {g.finish_order}  -> P0 placed {place}{['st', 'nd', 'rd', 'th'][min(place, 4) - 1]}"
        )
    return place


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", choices=["ddqn", "greedy", "random"], default="ddqn")
    ap.add_argument(
        "--checkpoint",
        default=None,
        help="DDQN checkpoint (defaults to checkpoints/ddqn_agent.best.pt)",
    )
    ap.add_argument("--num-players", type=int, default=4)
    ap.add_argument("--num-decks", type=int, default=2)
    ap.add_argument(
        "--games",
        type=int,
        default=1,
        help="number of games to play (play-by-play only printed for the first)",
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.checkpoint is None:
        args.checkpoint = "checkpoints/ddqn_agent.best.pt"

    if args.games < 1:
        ap.error("--games must be positive")
    seat0 = load_agent(
        args.agent, args.checkpoint, args.num_players, args.num_decks, args.seed
    )
    label = args.agent.upper() if args.agent == "ddqn" else args.agent.title()

    places = []
    for i in range(args.games):
        verbose = (args.games == 1) or (i == 0)
        p = play_one(
            seat0,
            args.num_players,
            args.num_decks,
            args.seed * 1000 + i,
            verbose=verbose,
        )
        places.append(p)

    if args.games > 1:
        dist = {p: places.count(p) for p in range(1, args.num_players + 1)}
        dist_str = "  ".join(f"place {p}={dist[p]}" for p in dist)
        print(
            f"\n{args.games} games (P0={label} vs {args.num_players - 1} Greedy): "
            f"win%={places.count(1) / args.games:.3f}   {dist_str}"
        )


if __name__ == "__main__":
    main()
