import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from game.rules import Game
from env import features
from game.moves import PASS_MOVE
from agents.random_agent import RandomAgent
from agents.dqn_agent import DQNAgent
from agents.qlearning import QLearningAgent


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-players", type=int, default=3)
    ap.add_argument("--num-decks", type=int, default=1)
    ap.add_argument("--checkpoint", default=None,
                    help="trained agent (.pt for DQN, .npy for Q-learning)")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    opps = []
    for s in range(1, args.num_players):
        if args.checkpoint:
            if args.checkpoint.endswith(".npy"):
                s_dim, a_dim = features.feature_dims(args.num_players, args.num_decks)
                ag = QLearningAgent(s_dim, a_dim, seed=args.seed)
                ag.load(args.checkpoint)
            else:
                s_dim, a_dim = features.feature_dims(args.num_players, args.num_decks)
                ag = DQNAgent(s_dim, a_dim, seed=args.seed)
                ag.load(args.checkpoint)
            ag.epsilon = 0.0
        else:
            ag = RandomAgent(seed=args.seed)
        opps.append(ag)

    g = Game(num_players=args.num_players, num_decks=args.num_decks, seed=args.seed)
    print("Deal dealt. You are P0. Opponents:",
          "trained agent" if args.checkpoint else "random")

    while not g.done:
        seat = g.current_player
        legal = g.legal_moves(seat)
        if seat == 0:
            move = human_choose(g, legal)
        else:
            obs = features.state_vector(g, seat)
            move = opps[seat - 1].act(obs, legal)
            print(f"P{seat} plays {move}  (hand: {len(g.hands[seat])})")
        g.apply_move(seat, move)

    print("\n=== Game over ===")
    print("Finish order:", g.finish_order)
    place = g.finish_order.index(0) + 1
    print(f"You finished {place}{['st','nd','rd','th'][min(place,4)-1]} of {args.num_players}.")


if __name__ == "__main__":
    main()
