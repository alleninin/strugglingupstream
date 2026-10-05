import argparse
import os
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from agents.runtime import load_agent
from env import features
from game.cards import RANK_LABELS
from game.moves import PASS_MOVE, generate_moves
from game.rules import Game


def show_hand(hand):
    return " ".join(c.label for c in sorted(hand, key=lambda c: c.rank))


def parse_play(raw, hand, legal):
    """Resolve typed rank counts to a legal move using the actual cards in hand."""
    if raw.strip().lower() in ("p", "pass"):
        if PASS_MOVE in legal:
            return PASS_MOVE
        raise ValueError("You must play cards when leading.")
    labels = {label: rank for rank, label in RANK_LABELS.items()}
    tokens = raw.upper().replace(",", " ").split()
    if not tokens:
        raise ValueError("Enter cards, such as 3 3 4 4 5 5, or p to pass.")
    unknown = next((token for token in tokens if token not in labels), None)
    if unknown is not None:
        raise ValueError(
            f"Unknown card '{unknown}'. Use 3–10, J, Q, K, A, 2, BJ or RJ."
        )
    wanted = Counter(labels[token] for token in tokens)
    if wanted - Counter(card.rank for card in hand):
        raise ValueError("You don't have those cards.")
    for move in legal:
        if not move.is_pass and Counter(card.rank for card in move.cards) == wanted:
            return move
    if any(
        Counter(card.rank for card in move.cards) == wanted
        for move in generate_moves(hand)
    ):
        raise ValueError(
            "That play doesn't beat the table. Try another play or p to pass."
        )
    raise ValueError("Those cards don't form a valid combination.")


def human_choose(game: Game, legal):
    while True:
        print(f"\nYour hand ({len(game.hands[0])}): {show_hand(game.hands[0])}")
        print(
            "Opponents:",
            " | ".join(
                f"P{seat}: {len(hand)} left"
                for seat, hand in enumerate(game.hands[1:], 1)
            ),
        )
        if game.table_move is None:
            print("Table: clear — you lead")
        else:
            print(f"Table: {show_hand(game.table_move.cards)} (P{game.table_owner})")
        raw = input("Cards (p=pass, ?=help): ").strip()
        command = raw.lower()
        if command in ("?", "help"):
            print("Type cards: 3 3 4 4 5 5. Ranks: 3–10 J Q K A 2 BJ RJ.")
            print("p: pass | moves: show legal plays | quit: end game")
        elif command == "moves":
            print("Legal plays:")
            for move in legal:
                print("  " + ("p" if move.is_pass else show_hand(move.cards)))
        elif command == "quit":
            print("Game ended.")
            raise SystemExit(0)
        else:
            try:
                return parse_play(raw, game.hands[0], legal)
            except ValueError as error:
                print(error)


def main():
    ap = argparse.ArgumentParser(
        description="Play against DDQN, Greedy or Random bots."
    )
    ap.add_argument("--num-players", type=int, default=4)
    ap.add_argument("--num-decks", type=int, default=2)
    ap.add_argument("--ddqn", action="store_true", help="use the current saved DDQN")
    ap.add_argument(
        "--ddqn-checkpoint", "--checkpoint", dest="checkpoint", default=None
    )
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--random", action="store_true")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    kinds = ["ddqn"] if args.ddqn or args.checkpoint else []
    kinds += (["greedy"] if args.greedy else []) + (["random"] if args.random else [])
    kinds = kinds or ["greedy"]
    checkpoint = args.checkpoint or "checkpoints/ddqn_agent.best.pt"
    names = [kinds[(seat - 1) % len(kinds)] for seat in range(1, args.num_players)]
    opps = [
        load_agent(kind, checkpoint, args.num_players, args.num_decks, args.seed)
        for kind in names
    ]

    g = Game(num_players=args.num_players, num_decks=args.num_decks, seed=args.seed)
    print(
        "You are P0. Opponents:",
        ", ".join(f"P{s + 1}={n}" for s, n in enumerate(names)),
    )

    while not g.done:
        seat = g.current_player
        legal = g.legal_moves(seat)
        if seat == 0:
            move = human_choose(g, legal)
            print("You:", "pass" if move.is_pass else show_hand(move.cards))
        else:
            obs = features.state_for(g, seat, opps[seat - 1])
            move = opps[seat - 1].act(obs, legal)
            print(
                f"P{seat}: {'pass' if move.is_pass else show_hand(move.cards)} "
                f"({len(g.hands[seat]) - len(move.cards)} left)"
            )
        g.apply_move(seat, move)

    print("\n=== Game over ===")
    print("Finish order:", g.finish_order)
    place = g.finish_order.index(0) + 1
    print(
        f"You finished {place}{['st', 'nd', 'rd', 'th'][min(place, 4) - 1]} of {args.num_players}."
    )


if __name__ == "__main__":
    try:
        main()
    except (EOFError, KeyboardInterrupt):
        print("\nGame ended.")
