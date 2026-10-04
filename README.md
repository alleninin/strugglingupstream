# Zheng Shang You (争上游)

A Python card-game engine with Greedy, Random, Q-learning, DQN and Double DQN bots.
Play against them, train your own, or run tournaments. Defaults: **4 players, 2 decks**.

## Setup

Run commands from the repository root, using the same Python interpreter throughout.
These examples use Python 3.12.

```bash
python3.12 -m pip install -r requirements.txt
```

## Play

```bash
# You vs three greedy bots (no training needed)
python3.12 scripts/play.py --greedy

# You vs three copies of a trained DDQN bot
python3.12 scripts/play.py --ddqn-checkpoint checkpoints/ddqn_v4_agent.best.pt

# You vs DDQN, Greedy and Random
python3.12 scripts/play.py --ddqn-checkpoint checkpoints/ddqn_v4_agent.best.pt --greedy --random
```

You are **P0**. Enter a displayed move number, or `p` to pass when allowed.
Selected opponents repeat to fill the seats. For other trained bots, use
`--dqn-checkpoint FILE`, `--q-checkpoint FILE`, or `--shaped-checkpoint FILE`
(add `--shaped-arch qlearning` for shaped Q-learning). Add `--seed 42` to repeat a deal.
If a checkpoint fails to load, play prints a warning and substitutes Greedy.

## Train

Choose a bot; **each command starts fresh**, not from an existing checkpoint.

```bash
python3.12 -m training.train --agent ddqn --episodes 2500 --save-path checkpoints/ddqn_v4_agent.pt
python3.12 -m training.train --agent dqn --episodes 5000 --save-path checkpoints/dqn_win_agent.pt
python3.12 -m training.train --agent qlearning --episodes 5000 --save-path checkpoints/qlearning_win_agent.npy
```

- DDQN learns without demonstrations, using remaining-hand partition features and a random → mixed → greedy opponent curriculum.
- DQN and Q-learning default to greedy opponents and 200 greedy demonstration games; disable demonstrations with `--demo-games 0`.
- Evaluation runs against greedy every 500 episodes. The ordinary checkpoint holds the latest weights; **`.best.pt` / `.best.npy`** holds the best validation model. Prefer the latter for play and evaluation.
- Useful options: `--seed 42`, `--opponent greedy`, `--eval-every 500`, `--eval-games 100`. CPU with one Torch thread is the default.

For alternative placement/trick rewards, use `--agent shaped` (`.pt`) or
`--agent shaped-ql` (`.npy`). All training options: `python3.12 -m training.train --help`.
Old checkpoints retain their old architecture; retrain to use the new DDQN features.
Partition features estimate how many plays each move leaves and add CPU work.
They are enabled by default for DDQN (`--no-partition-features` disables them).
Other learners can opt in; their win-rate benefit has not yet been established:

```bash
python3.12 -m training.train --agent dqn --partition-features --episodes 5000 --save-path checkpoints/dqn_partition_agent.pt
python3.12 -m training.train --agent qlearning --partition-features --episodes 5000 --save-path checkpoints/qlearning_partition_agent.npy
```

The flag also works with `shaped` and `shaped-ql`. Add `--opponent curriculum` to
use DDQN's opponent schedule; add `--demo-games 0` to disable demonstration warm-up.
Each agent keeps its own learning algorithm and reward.

## Evaluate

**Tournament:** available trained bots plus Greedy and Random, with seats rotated on
matched deals. Missing or incompatible checkpoints are skipped.

```bash
python3.12 evaluate.py --games 400 \
  --ddqn-path checkpoints/ddqn_v4_agent.best.pt \
  --dqn-path checkpoints/dqn_win_agent.best.pt \
  --q-path checkpoints/qlearning_win_agent.best.npy \
  --json reports/tournament.json
```

Results include win rates and placements. `--rounds 3` instead of `--games` runs
three complete rounds across every player subset. Without path flags, the script
looks for `checkpoints/<agent>_agent.pt` / `.npy`.

**DDQN vs fixed opponents:** test against three greedy bots and, separately, three
random bots. Pass multiple checkpoint paths to compare models on identical deals.
This evaluator uses four players and two decks.

```bash
python3.12 scripts/evaluate_ddqn.py checkpoints/ddqn_v4_agent.best.pt \
  --games 500 --include-baselines --output reports/ddqn_eval.json
```

Evaluation disables exploration. Tournament scores and scores against three greedy
bots measure different matchups; training win rates also include exploration.
Use a separate evaluation seed (`--seed`) when checking a validation-selected model.

## Watch bots

Watch a chosen bot in P0 against greedy opponents. With multiple games, only the
first prints every move; the final summary shows win rate and placements.

```bash
python3.12 scripts/watch.py --agent ddqn --checkpoint checkpoints/ddqn_v4_agent.best.pt --games 20
python3.12 scripts/watch.py --agent random --games 20
python3.12 scripts/demo.py  # one full greedy-bot game
```

`watch.py` also supports `dqn`, `qlearning`, `shaped` and `greedy`.
Play, watch, training and tournaments accept `--num-players` and `--num-decks`;
use settings matching your trained checkpoint. Each command has `--help`.

## Rules in this variant

- Empty your hand first. Rank order: `3 … 10 J Q K A 2 Black Joker Red Joker`.
- Play singles, pairs, triples, full houses, straights (5+), consecutive pairs (3+ ranks), consecutive triples (2+ ranks), or four-of-a-kind bombs. Runs exclude 2s and jokers; jokers are singles only.
- An airplane is exactly two distinct normal-rank triples plus two other distinct pairs (10 cards), ranked by its higher triple; the triples need not be consecutive.
- Beat the same combination type and length with a higher rank, or use a bomb. Only a higher bomb beats a bomb. After all other active players pass, the last player to play leads again.
- Multi-deal tribute is implemented in `game/match.py`; the commands above run individual deals.

## Tests and code

```bash
python3.12 -B -m unittest discover -s tests -v
python3.12 -B scripts/test_rules.py
```

`game/` contains the rules, `env/` the RL environment/features, `agents/` the
learners, `bots/` the heuristic/reward bots, and `training/` the training loop.
See the [DDQN experiment report](reports/ddqn-fix-20261003/REPORT.md) for architecture,
measured results and limitations.

Historical benchmark scripts and the redundant shaped-training CLI are in
[archive/experiments](archive/experiments/README.md). Active training uses `training.train`.
