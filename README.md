# Zheng Shang You (争上游)

A card-game engine with one trainable **DDQN** bot, **Greedy**, and a small **Random**
baseline used by the training curriculum. Defaults: four players, two decks.

## Setup

Run from the repository root:

```bash
python3.12 -m pip install -r requirements.txt
```

## Play

```bash
python3.12 scripts/play.py --ddqn                  # current trained bot
python3.12 scripts/play.py --greedy                # no checkpoint needed
python3.12 scripts/play.py --ddqn --greedy --random # mixed opponents
```

You are P0. Type the cards you want to play, such as `3 3 4 4 5 5` or `J J`.
Use `3`–`10`, `J`, `Q`, `K`, `A`, `2`, `BJ` (black joker), `RJ` (red joker).
Input is case-insensitive; spaces or commas separate cards. Illegal plays explain
what went wrong and let you retry. Use `p` to pass, `?` for help, `moves` to list
legal plays, or `quit` to leave. Your hand and table are repeated before each prompt. Opponents repeat to fill the seats. Use `--checkpoint FILE` for another DDQN model or `--seed 42` to
repeat a deal. A missing checkpoint stops play rather than silently changing bots.

## Train

```bash
python3.12 -m training.train --episodes 2500
```

Starts a fresh model with hand-partition features and random → mixed → greedy
opponents. Half of epsilon exploration follows Greedy; half chooses random moves.
Play/evaluation use the learned policy and always take an immediate finish.

To try direct Greedy supervision that fades away during training:

```bash
python3.12 -m training.train --episodes 2500 --supervision-weight 0.5 \
  --save-path checkpoints/supervised.pt
```

This adds a move-ranking loss on ordinary training decisions, fading to zero over
`--supervision-fraction 0.6` of the run. It adds no demonstration games and does not
consult Greedy during inference. Existing guided exploration remains separate;
`--supervision-weight 0` is the default and disables this experiment.

Three matched 2,500-episode seeds improved mean wins against Greedy from **30.2%
to 37.0%**, but combination-preservation checks worsened, so this stays opt-in.
See the [results and saved bot](archive/reports/ddqn-supervision-20261004/REPORT.md).

For mixed self-play against Greedy, Random and frozen DDQNs (at most 5,000 episodes):

```bash
python3.12 -m training.train --episodes 5000 --opponent selfplay \
  --selfplay-checkpoint checkpoints/ddqn_agent.best.pt \
  --save-path checkpoints/selfplay.pt
```

The supplied checkpoint stays as a frozen opponent alongside three rolling
snapshots. Without `--selfplay-checkpoint`, self-play starts after 500 curriculum
episodes. Each opponent seat is sampled as 60% Greedy, 30% frozen DDQN, 10% Random.

Optional experiments: `--endgame-rate 0.15` replays difficult training positions;
`--history-features` adds four recent public actions; `--target-method monte-carlo`
uses complete-game returns. These options are experimental. The capped comparison
did not establish a self-play advantage, so the defaults and supplied checkpoint
are unchanged.
Practice episodes count toward `--episodes`; evaluation always uses fresh full deals.

Evaluation against Greedy runs every 500 episodes. Saves latest weights to
`checkpoints/ddqn_agent.pt` and best validation weights to `checkpoints/ddqn_agent.best.pt`
(or your `--save-path` and its `.best.pt` sibling). Use a separate save path to
preserve the supplied model. History checkpoints load automatically in play/evaluation.

Useful flags: `--eval-games 500`, `--seed 1`, `--opponent greedy`,
`--expert-exploration 0`. CPU with one Torch thread is the default. Training starts
fresh; it does not resume checkpoints. Use `--help` for all options.

## Evaluate and watch

```bash
# DDQN against three greedy bots and, separately, three random bots
python3.12 scripts/evaluate_ddqn.py checkpoints/ddqn_agent.best.pt --games 500 --include-baselines --output reports/evaluation.json

# Mixed tournament: two DDQN seats, one Greedy, one Random; seats rotate
python3.12 evaluate.py --games 400 --json reports/tournament.json

# After training supervised.best.pt, compare tactical behavior on shared positions
python3.12 scripts/benchmark_strategy.py checkpoints/ddqn_agent.best.pt checkpoints/supervised.best.pt --output reports/strategy.json

# Watch DDQN against greedy; summarize 20 games, printing the first
python3.12 scripts/watch.py --games 20
```

The old `--dqn-path` / `--q-path` options moved to `archive.legacy.evaluate`.
The current model is `checkpoints/ddqn_agent.best.pt`; older weights remain archived.
See [archived evaluation](archive/README.md) to compare the older learners.

The strategy benchmark uses four players and two decks. It reports 14 tactical
checks plus shared positions from Greedy games, separating raw network choices from the immediate-finish safeguard.
Its heuristic checks and Greedy-agreement counts are diagnostics, not win rates.

Evaluation disables exploration. Fixed-opponent and mixed-tournament win rates
measure different matchups. Play, training, watch and tournaments also support
`--num-players` / `--num-decks`; match the checkpoint's training settings.

## Rules and tests

Empty your hand first. Ranks: `3 … A 2 Black Joker Red Joker`. Play singles, pairs,
triples, full houses, runs, airplanes or four-card bombs. Beat the same type/length
with a higher rank, or use a bomb. Jokers are singles only; runs exclude 2/jokers.
An airplane is two distinct normal-rank triples plus two other distinct pairs.
When every other active player passes, the last player to play leads again.
Multi-deal tribute is implemented in `game/match.py`.

```bash
python3.12 -B -m unittest discover -s tests -v
```

`agents/`: DDQN and its networks/replay. `bots/`: Greedy and Random. `game/`: rules.
`env/`: observations and rewards. `training/`: current trainer. `scripts/`: play/evaluate/watch.

Older learners, checkpoints and experiment reports are preserved in
[archive/](archive/README.md), including the [self-play comparison](archive/reports/ddqn-selfplay-20261004/REPORT.md).
