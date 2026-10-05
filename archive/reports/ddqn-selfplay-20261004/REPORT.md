# Capped self-play experiments — 2026-10-04

Implemented the requested training experiments inside the existing DDQN trainer.
No LSTM, new bot family, training-resume system, or additional dependency was added.
Every newly trained learner ran for at most 5,000 episodes: four 2,000-episode
screens followed by three 5,000-episode comparisons. The supplied checkpoint and
training defaults remain unchanged because self-play did not demonstrate an
advantage over the simpler trainer at the same learner episode budget.

## Available options

- `--opponent selfplay`: independently sample each opposing seat as 60% Greedy,
  30% frozen DDQN and 10% Random. Capture the learner every 500 episodes and retain
  three recent snapshots. Frozen inference copies cannot change with learner
  weights/configuration and never consume its exploration RNG.
- `--selfplay-checkpoint FILE`: retain one additional fixed pretrained opponent.
  With an anchor, the mixture starts immediately. Without one, the existing
  curriculum runs until the first snapshot at episode 500.
- `--endgame-rate 0.15`: replace about 15% of episodes with valid positions from
  lost training games. Cache at most 256 positions, taking the first non-forced
  decision where the learner holds at most 12 cards and an opponent at most eight.
  Copy the game, preserving the table, turn, public history and initial deal scale.
  Practice episodes count toward the episode limit; practice losses are not
  recursively reinserted. No evaluation position enters this cache.
- `--history-features`: append four public actions, oldest first, including actor
  relative to the learner, move type/rank/length and explicit padding. Opponents'
  hidden cards are never included. Checkpoint schema selects the appropriate
  observation automatically in play, watch, tournaments and evaluation.
- `--target-method monte-carlo`: regress to the complete discounted return from
  each decision. These targets do not bootstrap. Incomplete episodes are discarded.
  This is an alternative to the default n-step Double-DQN target, using the same
  network and replay implementation; it is not presented as Double-DQN learning.

## Results

All runs use training seed 0, the existing guided exploration, CPU/one Torch
thread, and best-checkpoint selection every 500 episodes on 100 validation deals.
The final test uses 500 fresh matched deals per opponent, evaluation seed 8100427,
with epsilon zero, against three Greedy bots and separately three Random bots.
The prior supplied model is evaluated on exactly those deals too.

| Model | Learner episodes | vs Greedy | vs Random |
| --- | ---: | ---: | ---: |
| Previously supplied model | 2,500 (earlier run) | 30.8% | 92.6% |
| Existing trainer | 5,000 | 36.8% | 91.0% |
| Anchored self-play | 5,000 | 35.0% | 89.6% |
| Anchored self-play + history + endgame practice | 5,000 | 29.4% | 91.2% |

Greedy control: 25.4% vs Greedy, 86.2% vs Random. Random control: 1.6%, 25.4%.
These are fresh deals, so the earlier report's 36.2% for the supplied model is not
a contradictory result or a code regression. One training seed and 500 test games
cannot establish a reliable small difference. The 1.8-point difference between
self-play and the same-budget baseline is not treated as significant. Self-play
also has access to a pretrained 2,500-episode anchor; equal learner episode counts
do not mean identical total historical training data or compute.

The full history/practice run selected its best checkpoint at episode 2,000 and
used 711 practice episodes by the 5,000-episode cap. Its later checkpoints did not
improve validation. Adding all options did not produce the hoped-for improvement.
No 40–50% claim is supported by this experiment. The current checkpoint is retained
rather than choosing a replacement based on small, single-seed held-out differences.

The initial 2,000-episode screens used 200 other held-out deals (seed 741021):

| Screen | vs Greedy | vs Random |
| --- | ---: | ---: |
| Existing trainer | 36.5% | 92.0% |
| Unanchored self-play + late endgame practice | 31.0% | 85.5% |
| Monte Carlo targets | 14.5% | 77.5% |
| History only | 33.5% | 89.5% |

That initial practice screen cached the latest pressure decision. The final
implementation instead captures the first such decision, to avoid concentrating
on positions already lost after earlier mistakes. The final self-play runs also
retain the previously trained anchor. Thus the screen is a historical prototype,
not a claim that the final recipe reproduces its precise numbers.

Final train/evaluation times while runs overlapped on this machine:
- Existing trainer: 436.7s training + 67.6s validation.
- Self-play: 539.1s + 68.2s.
- History/endgame: 484.9s + 69.2s.

These are observed concurrent-run timings, not isolated hardware benchmarks.
All aggregate results, validation histories and checkpoint hashes are in `results.json`.

## Reproduce or try the saved policies

Run from the project root. `anchor.pt` preserves the prior supplied checkpoint;
`baseline.best.pt`, `selfplay.best.pt` and `history-endgame.best.pt` are the selected
experimental weights. They are archived together to keep the active bot/checkpoint
folders small.

```bash
python3.12 -m training.train --episodes 5000 --seed 0 \
  --opponent selfplay \
  --selfplay-checkpoint archive/reports/ddqn-selfplay-20261004/anchor.pt \
  --save-path checkpoints/selfplay.pt

# Add these flags for the history/endgame experiment:
# --history-features --endgame-rate 0.15

python3.12 scripts/evaluate_ddqn.py \
  archive/reports/ddqn-selfplay-20261004/selfplay.best.pt \
  --games 500 --seed 8100427 --output reports/selfplay-evaluation.json

python3.12 scripts/play.py --ddqn \
  --checkpoint archive/reports/ddqn-selfplay-20261004/selfplay.best.pt
```

The simple baseline uses the same training command without self-play/anchor flags.
Optional modes remain opt-in. Existing legacy, hand-feature and partition-feature
checkpoints still load. Archived runnable CLIs received only the observation
adapter needed to accept history checkpoints; historical experiment snapshots
were not rewritten.

Validation: 99 active tests and 97 archived tests pass, including frozen opponent
isolation, public history/seat rotation, old/new checkpoint round trips, copied
practice positions, and exact discounted Monte Carlo returns. History-aware play,
watch, both tournament CLIs and the combined training options passed smoke checks.
Ruff F/E9/I checks pass. Full training logs and pilot scratch files are in
`/tmp/zheng-selfplay`, outside the active project.
