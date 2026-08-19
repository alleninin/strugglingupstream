import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from game.rules import Game
from agents.random_agent import RandomAgent


def main():
    num_players = 4
    g = Game(num_players=num_players, num_decks=2, seed=42)
    agents = [RandomAgent(seed=42 + i) for i in range(num_players)]

    print("Initial hands:")
    for i, h in enumerate(g.hands):
        labels = " ".join(c.label for c in sorted(h, key=lambda c: c.rank))
        print(f"  P{i}: {labels}")

    turn = 0
    while not g.done:
        seat = g.current_player
        legal = g.legal_moves(seat)
        move = agents[seat].act(None, legal)
        lead = g.table_move is None
        tag = "leads" if lead else "follows"
        print(f"  P{seat} {tag}: {move}  (hand: {len(g.hands[seat])})")
        g.apply_move(seat, move)
        turn += 1
        if turn > 1000:
            print("SAFETY STOP")
            break

    print("\nFinish order:", g.finish_order)
    print("Placements:", [f"P{p}" for p in g.finish_order])


if __name__ == "__main__":
    main()
