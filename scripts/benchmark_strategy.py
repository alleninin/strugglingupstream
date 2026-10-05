"""Compare policies on shared tactical fixtures and Greedy-generated game states."""

import argparse
import copy
import json
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from agents.ddqn_agent import DDQNAgent
from agents.runtime import load_agent
from bots.greedy_bot import GreedyBot
from env import features
from env.env import ZhengShangYouEnv
from game.cards import build_deck
from game.moves import MoveType, generate_moves
from game.rules import Game


@dataclass
class TacticalCase:
    name: str
    category: str
    game: Game
    accepted: tuple


def position(ranks, table_rank=None, opponent_sizes=(20, 20, 20)):
    """Construct a card-conserving four-player diagnostic position, not a replay."""
    available = build_deck(2)

    def take(rank):
        card = next(card for card in available if card.rank == rank)
        available.remove(card)
        return card

    own = [take(rank) for rank in ranks]
    table = None if table_rank is None else generate_moves([take(table_rank)])[0]
    random.Random(1701).shuffle(available)
    hands = [own]
    for size in opponent_sizes:
        hands.append(available[:size])
        del available[:size]
    game = Game(hands=hands, starting_player=0)
    game.initial_hand_size = 27
    game.table_move = table
    game.table_owner = -1 if table is None else 3
    game.played_cards = available + ([] if table is None else list(table.cards))
    if table is not None:
        game.action_history.append((3, table))
    return game


def tactical_cases():
    """Explicit behavioral expectations; non-finishing checks are heuristic advice."""

    def case(name, category, ranks, rule, value=None, table=None, sizes=(20, 20, 20)):
        game = position(ranks, table, sizes)
        accepted = tuple(
            move
            for move in game.legal_moves()
            if (rule == "finish" and len(move.cards) == len(ranks))
            or (rule == "pass" and move.is_pass)
            or (rule == "type" and move.type == value)
            or (rule == "rank" and not move.is_pass and move.rank == value)
        )
        if not accepted:
            raise ValueError(f"fixture has no acceptable legal move: {name}")
        return TacticalCase(name, category, game, accepted)

    return [
        case("finish full house", "finish", [9, 9, 9, 15, 15], "finish"),
        case("finish with bomb", "finish", [3] * 4, "finish", table=17),
        case(
            "finish airplane",
            "finish",
            [3] * 3 + [7] * 3 + [9] * 2 + [11] * 2,
            "finish",
        ),
        case(
            "lead straight",
            "combination",
            [3, 6, 7, 8, 9, 10, 15, 17],
            "type",
            MoveType.STRAIGHT,
        ),
        case(
            "lead full house",
            "combination",
            [3, 7, 7, 7, 9, 9, 15, 17],
            "type",
            MoveType.FULL_HOUSE,
        ),
        case(
            "preserve three pairs", "combination", [7, 7, 9, 9, 11, 11], "pass", table=3
        ),
        case("preserve two pairs", "combination", [7, 7, 9, 9], "pass", table=3),
        case(
            "preserve straight",
            "combination",
            [5, 6, 7, 8, 9, 11, 11, 13, 13],
            "pass",
            table=4,
        ),
        case(
            "block last-card opponent",
            "blocking",
            [5, 8, 9, 9, 11, 11, 17],
            "rank",
            17,
            table=3,
            sizes=(1, 20, 20),
        ),
        case(
            "avoid single-card gift",
            "blocking",
            [3, 7, 7, 9, 9, 11, 11],
            "type",
            MoveType.PAIR,
            sizes=(1, 20, 20),
        ),
        case(
            "use control to set up pair", "contesting", [7, 7, 17], "rank", 17, table=15
        ),
        case(
            "contest with spare two",
            "contesting",
            [5, 6, 7, 8, 9, 15, 17],
            "rank",
            15,
            table=4,
        ),
        case(
            "save bomb without pressure",
            "bomb",
            [3] * 4 + [5] * 2 + [7] * 2 + [9] * 2,
            "pass",
            table=17,
        ),
        case(
            "spend bomb under pressure",
            "bomb",
            [3] * 4 + [5] * 2 + [7] * 2 + [9] * 2,
            "type",
            MoveType.BOMB,
            table=17,
            sizes=(8, 20, 20),
        ),
    ]


def sample_positions(games=100, seed=920041, limit=1000):
    """Reservoir-sample non-forced states from independent Greedy-vs-Greedy deals."""
    if games < 1 or limit < 1:
        raise ValueError("games and position limit must be positive")
    teacher = GreedyBot()
    env = ZhengShangYouEnv(reward_scheme="win")
    rng = random.Random(seed)
    positions, seen = [], 0
    for episode in range(games):
        state = env.reset(seed=(1 << 60) + seed * 1000003 + episode)
        while not env.done:
            legal = env.get_legal_moves()
            if len(legal) > 1:
                seen += 1
                index = (
                    len(positions) if len(positions) < limit else rng.randrange(seen)
                )
                if index < limit:
                    snapshot = copy.deepcopy(env.game)
                    if index == len(positions):
                        positions.append(snapshot)
                    else:
                        positions[index] = snapshot
            state, _, _, _ = env.step(teacher.act(state, legal))
    return positions


def policy_moves(agent, game):
    """Return deployed and raw-network moves without exploration or RNG mutation."""
    legal = game.legal_moves()
    obs = features.state_for(game, game.current_player, agent)
    if not isinstance(agent, DDQNAgent):
        move = agent.act(obs, legal)
        return move, move
    with torch.no_grad():
        scores = agent._scores(
            agent.policy_net,
            agent._t(obs[None, : agent.state_dim]),
            agent._t(agent._move_vectors(legal)),
            torch.zeros(len(legal), dtype=torch.long, device=agent.device),
            [len(legal)],
        )
    raw = legal[int(scores.argmax().item())]
    deployed = agent.act(obs, legal, network=agent.policy_net)
    return deployed, raw


def describe(move):
    return {"type": move.type.name, "ranks": [card.rank for card in move.cards]}


def benchmark(agent, cases, positions):
    categories = defaultdict(lambda: {"cases": 0, "passed": 0, "raw_passed": 0})
    details = []
    for case in cases:
        deployed, raw = policy_moves(agent, case.game)
        passed, raw_passed = deployed in case.accepted, raw in case.accepted
        totals = categories[case.category]
        totals["cases"] += 1
        totals["passed"] += passed
        totals["raw_passed"] += raw_passed
        details.append(
            dict(
                name=case.name,
                category=case.category,
                passed=passed,
                raw_passed=raw_passed,
                move=describe(deployed),
                raw_move=describe(raw),
                expected=[describe(move) for move in case.accepted],
            )
        )
    teacher = GreedyBot()
    counts = dict(
        decisions=len(positions),
        teacher_agreement=0,
        teacher_contests=0,
        passes_when_teacher_contests=0,
        urgent_responses=0,
        urgent_passes=0,
        finishes_available=0,
        missed_finishes=0,
        raw_missed_finishes=0,
    )
    for game in positions:
        legal = game.legal_moves()
        move, raw = policy_moves(agent, game)
        expert = teacher.act(features.state_vector(game, 0), legal)
        counts["teacher_agreement"] += move == expert
        if game.table_move is not None and not expert.is_pass:
            counts["teacher_contests"] += 1
            counts["passes_when_teacher_contests"] += move.is_pass
        minimum = min(len(hand) for hand in game.hands[1:] if hand)
        if (
            game.table_move is not None
            and minimum <= len(game.table_move.cards)
            and any(not candidate.is_pass for candidate in legal)
        ):
            counts["urgent_responses"] += 1
            counts["urgent_passes"] += move.is_pass
        hand_size = len(game.hands[0])
        if any(len(candidate.cards) == hand_size for candidate in legal):
            counts["finishes_available"] += 1
            counts["missed_finishes"] += len(move.cards) != hand_size
            counts["raw_missed_finishes"] += len(raw.cards) != hand_size
    return dict(tactical=dict(categories), cases=details, shared_states=counts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", nargs="*", type=Path)
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--positions", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=920041)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.games < 1 or args.positions < 1:
        parser.error("--games and --positions must be positive")
    torch.set_num_threads(1)
    cases = tactical_cases()
    positions = sample_positions(args.games, args.seed, args.positions)
    policies = [("greedy", GreedyBot())]
    policies.extend(
        (str(path), load_agent("ddqn", path, seed=0)) for path in args.checkpoints
    )
    results = dict(
        seed=args.seed,
        source_games=args.games,
        position_limit=args.positions,
        notes="Constructed tactical checks are heuristic expectations, not optimality proofs. "
        "Shared-state agreement/pass counts are diagnostics, not win rates. "
        "Raw scores exclude the deployed immediate-finish safeguard.",
        results={},
    )
    for name, agent in policies:
        result = benchmark(agent, cases, positions)
        results["results"][name] = result
        print(
            name,
            json.dumps(
                dict(tactical=result["tactical"], shared_states=result["shared_states"])
            ),
            flush=True,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
