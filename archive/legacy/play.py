import argparse
import os
import sys

from pathlib import Path
ROOT = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, ROOT)

from game.rules import Game
from env import features
from game.moves import PASS_MOVE
from archive.legacy.dqn_agent import DQNAgent
from archive.legacy.qlearning import QLearningAgent
from agents.ddqn_agent import DDQNAgent
from bots.greedy_bot import GreedyBot
from bots.random_bot import RandomAgent


def show_hand(hand):
    return " ".join(c.label for c in sorted(hand, key=lambda c: c.rank))


def human_choose(game: Game, legal):
    print("\n--- Your turn (seat 0) ---")
    print("Your hand:", show_hand(game.hands[0]))
    if game.table_move is not None:
        print("Table:", game.table_move, "(led by P%d)" % game.table_owner)
    else:
        print("Table: (you are leading)")
    print("Legal moves:")
    for i, m in enumerate(legal):
        print(f"  [{i}] {m}")
    while True:
        raw = input("Choose # (or 'p' to pass): ").strip()
        if raw.lower() == "p" and any(m.is_pass for m in legal):
            return PASS_MOVE
        try:
            idx = int(raw)
            if 0 <= idx < len(legal):
                return legal[idx]
        except ValueError:
            pass
        print("Invalid choice, try again.")


def load_agent(kind, path, s_dim, a_dim, seed, num_players, num_decks):
    if kind == "random":
        return RandomAgent(seed=seed)
    if kind == "q":
        ag = QLearningAgent(s_dim, a_dim, seed=seed)
    elif kind == "ddqn":
        ag = DDQNAgent(s_dim, a_dim, seed=seed)
    elif kind.startswith("shaped"):
        arch = kind.split(":", 1)[1] if ":" in kind else "ddqn"
        from archive.legacy.shaped_reward_bot import ShapedRewardBot
        ag = ShapedRewardBot(s_dim, a_dim, agent_type=arch)
    elif kind == "greedy":
        from bots.greedy_bot import GreedyBot
        ag = GreedyBot(num_players=num_players, num_decks=num_decks, seed=seed)
    else:
        ag = DQNAgent(s_dim, a_dim, seed=seed)
    if kind == "greedy":
        return ag
    try:
        ag.load(path)
    except Exception as e:
        print(f"warning: skipping {path}: {e}")
        return GreedyBot(num_players=num_players, num_decks=num_decks, seed=seed)
    ag.epsilon = 0.0
    return ag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-players", type=int, default=4)
    ap.add_argument("--num-decks", type=int, default=2)
    ap.add_argument("--checkpoint", default=None,
                    help="trained agent (.pt for DQN/DDQN, .npy for Q-learning); "
                         "used for every opponent seat (legacy single-agent mode)")
    ap.add_argument("--agent", choices=["dqn", "ddqn"], default="dqn",
                    help="which .pt agent architecture to load with --checkpoint")
    ap.add_argument("--q-checkpoint", default=None,
                    help="Q-learning opponent (seat 1, then wraps if more seats)")
    ap.add_argument("--dqn-checkpoint", default=None,
                    help="DQN opponent (seat 2, then wraps if more seats)")
    ap.add_argument("--ddqn-checkpoint", default=None,
                    help="DDQN opponent (seat 3, then wraps if more seats)")
    ap.add_argument("--shaped-checkpoint", default=None,
                    help="Shaped-reward (DDQN/QL) opponent; per-seat like the others")
    ap.add_argument("--shaped-arch", choices=["ddqn", "qlearning"], default="ddqn",
                    help="inner architecture behind --shaped-checkpoint")
    ap.add_argument("--greedy", action="store_true",
                    help="add a greedy rule-based opponent (per-seat like the others)")
    ap.add_argument("--random", action="store_true", help="add a random opponent")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    s_dim, a_dim = features.feature_dims(args.num_players, args.num_decks)

    specs = []
    if args.q_checkpoint:
        specs.append(("q", args.q_checkpoint, "QL"))
    if args.dqn_checkpoint:
        specs.append(("dqn", args.dqn_checkpoint, "DQN"))
    if args.ddqn_checkpoint:
        specs.append(("ddqn", args.ddqn_checkpoint, "DDQN"))
    if args.shaped_checkpoint:
        specs.append((f"shaped:{args.shaped_arch}", args.shaped_checkpoint, "SHAPED"))
    if args.greedy:
        specs.append(("greedy", None, "GREEDY"))
    if args.random:
        specs.append(("random", None, "RANDOM"))

    opps, names = [], []
    for s in range(1, args.num_players):
        if args.checkpoint:
            kind = "q" if args.checkpoint.endswith(".npy") else args.agent
            opps.append(load_agent(kind, args.checkpoint, s_dim, a_dim, args.seed,
                                   args.num_players, args.num_decks))
            names.append(args.agent.upper() if kind != "q" else "QL")
        elif specs:
            idx = (s - 1) % len(specs)
            kind, path, label = specs[idx]
            opps.append(load_agent(kind, path, s_dim, a_dim, args.seed,
                                   args.num_players, args.num_decks))
            names.append(label)
        else:
            opps.append(GreedyBot(num_players=args.num_players,
                                  num_decks=args.num_decks, seed=args.seed))
            names.append("GREEDY")

    g = Game(num_players=args.num_players, num_decks=args.num_decks, seed=args.seed)
    print("Deal dealt. You are P0. Opponents:",
          ", ".join(f"P{s + 1}={n}" for s, n in enumerate(names)))

    while not g.done:
        seat = g.current_player
        legal = g.legal_moves(seat)
        if seat == 0:
            move = human_choose(g, legal)
        else:
            obs = features.state_for(g, seat, opps[seat - 1])
            move = opps[seat - 1].act(obs, legal)
            print(f"P{seat} plays {move}  (hand: {len(g.hands[seat]) - len(move.cards)})")
        g.apply_move(seat, move)

    print("\n=== Game over ===")
    print("Finish order:", g.finish_order)
    place = g.finish_order.index(0) + 1
    print(f"You finished {place}{['st','nd','rd','th'][min(place,4)-1]} of {args.num_players}.")


if __name__ == "__main__":
    main()
