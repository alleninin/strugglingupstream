"""Before/after probes; run from any directory with Python 3.12.

These deterministic policies are stress tests, not substitutes for human play.
Each policy plays 50 held-out deals in all four seats against three Greedy bots.
"""
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from bots.greedy_bot import GreedyBot
from env.features import state_vector
from game.moves import MoveType
from game.rules import Game

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('greedy_before_passing_fix', HERE / 'greedy_before.py')
old = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = old
spec.loader.exec_module(old)


def probe_move(legal, policy):
    non_pass = [move for move in legal if not move.is_pass]
    if not non_pass:
        return legal[0]
    if policy == 'low_singles':
        singles = [move for move in non_pass if move.type == MoveType.SINGLE]
        if singles:
            return min(singles, key=lambda move: move.rank)
    ordinary = [move for move in non_pass if not move.is_bomb]
    return min(ordinary or non_pass, key=lambda move: (-len(move.cards), move.rank))


def run(bot_type, policy):
    rows = []
    for seed in range(310000, 310050):
        for focal in range(4):
            game = Game(seed=seed)
            bots = [bot_type() for _ in range(4)]
            eligible, passed, early_eligible, early_passed = 0, 0, 0, 0
            streak, longest = 0, 0
            while not game.done:
                seat = game.current_player
                legal = game.legal_moves()
                if seat == focal:
                    move = probe_move(legal, policy)
                    streak += len(move.cards)
                    longest = max(longest, streak)
                else:
                    move = bots[seat].act(state_vector(game, seat), legal)
                    if game.table_move is not None and game.hands[focal] and any(not candidate.is_pass for candidate in legal):
                        eligible += 1
                        passed += move.is_pass
                        if len(game.hands[focal]) > 8:
                            early_eligible += 1
                            early_passed += move.is_pass
                    if not move.is_pass:
                        streak = 0
                game.apply_move(seat, move)
            rows.append({'seed': seed, 'focal': focal, 'probe_place': game.finish_order.index(focal) + 1,
                         'eligible': eligible, 'voluntary_passes': passed,
                         'early_eligible': early_eligible, 'early_passes': early_passed,
                         'longest_uncontested_cards_shed': longest})
    return {'games': len(rows), 'probe_win_rate': sum(row['probe_place'] == 1 for row in rows) / len(rows),
            'voluntary_pass_rate': sum(row['voluntary_passes'] for row in rows) / sum(row['eligible'] for row in rows),
            'early_pass_rate': sum(row['early_passes'] for row in rows) / sum(row['early_eligible'] for row in rows),
            'mean_longest_uncontested_cards': sum(row['longest_uncontested_cards_shed'] for row in rows) / len(rows),
            'records': rows}


results = {}
for policy in ('low_singles', 'large_combos'):
    results[policy] = {}
    for label, bot_type in (('before', old.GreedyBot), ('after', GreedyBot)):
        report = run(bot_type, policy)
        results[policy][label] = report
        print(policy, label, {key: value for key, value in report.items() if key != 'records'}, flush=True)
(HERE / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
