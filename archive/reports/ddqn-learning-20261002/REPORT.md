# Learning investigation — October 2–3, 2026

The training pipeline had objective and exploration problems, but correcting
those alone did **not** make the learners strong against greedy. Experiments also
showed forgetting during reinforcement learning. Do not interpret the code changes
as a guarantee that longer training will improve win rate.

## Findings and changes

1. The old reward paid 0.1 per card and a placement bonus, even after another
   player had secured first place. This optimizes a different objective from the
   reported first-place win rate. The new `win` reward ends the episode at the
   first finisher, awards +1/-1, and uses undiscounted potential shaping. The
   shaping return telescopes to a constant, rather than rewarding a losing
   strategy independently of the outcome. Legacy rewards remain selectable.
2. Exploration decayed on every transition, including forced passes. At 500
   episodes, old DDQN/DQN epsilon was already about 0.06 and Q-learning had
   reached its 0.05 minimum. The schedule now runs by episode, from 0.5 to 0.05
   across 80% of the requested run (minimum duration 1,000 episodes).
3. Evaluation seeds overlapped training seeds, particularly with seed 0.
   Evaluation and demonstration deals now occupy separate seed ranges.
4. Random play almost never beats three greedy opponents. A new expert-assisted
   option provides greedy demonstrations, warm-up, and rehearsal during RL.
   Default plain learners use 200 demonstration games and 1,000 warm-up batches;
   `--demo-games 0` preserves pure RL. No teacher or greedy fallback is used by
   a saved model during play. Larger warm-ups were also explored below.
5. Neural TD updates now use Huber loss, gradient clipping, and a smaller
   training learning rate. Network shapes/checkpoint formats remain compatible.
6. Evaluation can get worse with further training. The trainer now keeps a
   separate `.best.pt`/`.best.npy` checkpoint, including warm-up validation,
   rather than leaving users only the last weights. The ordinary path still
   holds the latest weights. This guards validation regressions, not unseen-test
   regressions. Progress also reports the exploratory training win rate.

The underlying DQN maximum-target and DDQN online-selection/target-evaluation
calculations passed their regression tests. Q-learning is a linear approximation
with state–action interaction features, not tabular learning over all hands;
its representational limitations are not fixed merely by increasing iterations.

## Controlled comparisons

Four players, two decks, training seed 0, 2,500 RL episodes against three greedy
opponents. Each final checkpoint was tested on the same **300 separate deals per
opponent type**, using evaluation seed 913 and exploration disabled. No checkpoint
was selected using these final test scores. One training seed was used; these are
diagnostics, not multi-seed statistical claims.

| Model | Before vs greedy | Revised vs greedy | Before vs random | Revised vs random |
| --- | ---: | ---: | ---: | ---: |
| DDQN | 0.7% | 5.3% | 7.7% | 56.0% |
| DQN | 10.0% | 7.3% | 70.0% | 73.3% |
| Q-learning | 6.7% | 7.7% | 72.3% | 77.0% |

Revised runs include **200 additional demonstration games**, 1,000 warm-up
batches, and rehearsal weight 0.1. They therefore do not have the same total
data/compute budget as the baseline. These are the **last** checkpoints, not
the best intermediate checkpoints. Automatic best-checkpoint retention was added
after these runs exposed regression; it does not retrospectively change the table.
Feature caching and additional logging were also added without changing the
learning calculation. Runs were concurrent, so their elapsed times should not
be used as controlled speed measurements.

DDQN improves substantially from a very weak baseline, but remains below greedy.
DQN's final greedy score falls; there is no evidence here of a consistent DQN
improvement from this combined recipe. Q-learning's one-point gain against greedy
is too small to draw a strong conclusion from 300 games. For context, an identical
greedy policy won 26.7% against three greedy opponents; random won 0.3%. Against
three random opponents, greedy won 88.7% and random won 22.3%.

An initial DDQN ablation changed reward, schedule and optimizer stabilization
without demonstrations: after 2,500 episodes it scored **0% vs greedy and 0.3% vs
random** on the same final evaluation. Those changes alone are insufficient.

## Demonstration-only diagnostics

An exploratory larger warm-up used 1,000 greedy demonstration games (6,921
non-forced decisions) and 3,000 warm-up batches, with **zero RL episodes**.
Testing used the same 300 deals per opponent as above:

| Model | Vs greedy | Vs random |
| --- | ---: | ---: |
| DDQN | 14.3% | 80.3% |
| DQN | 12.0% | 85.7% |
| Q-learning | 7.3% | 70.3% |

These are imitation-learning results, not evidence that reinforcement learning
has learned to outperform its teacher. They show why retaining the warm-up model
matters. They use more expert data than the main comparison and were exploratory
trials on a reused test set; validate on new deals before making strong comparisons.
The smaller 200-game warm-up diagnostics used only 100 test deals and are recorded
separately in `demo-only-*.json`; do not directly compare those estimates with the
300-game table as if they used identical samples.

## Validation and artifacts

- 76 unit tests pass, covering reward telescoping, stopping at either winner,
  legal ragged-action demonstration losses, warm-up for all three learners,
  target-network synchronization, exploration, separate evaluation seeds,
  best-checkpoint retention, and the existing rules and TD-target regressions.
- All three training CLIs complete short warm-up/training/evaluation/save runs.
- No previous user checkpoint was overwritten. Main final checkpoints and logs
  are alongside this report as `after-{ddqn,dqn,qlearning}.*`; larger demonstration
  checkpoints use `larger-demo-*`. Baseline DDQN and win-only checkpoints are in
  `checkpoints/ddqn-learning-20261002/`. `controls.json` records their final tests.
- `*_before.py` files preserve pre-change training/environment/agent code.

From the repository root, reproduce a main comparison:

```sh
python3.12 -B archive/experiments/benchmark_learning.py --agent ddqn --episodes 2500 --baseline reports/ddqn-learning-20261002 --output /tmp/before-ddqn.json
python3.12 -B archive/experiments/benchmark_learning.py --agent ddqn --episodes 2500 --demo-games 200 --demo-updates 1000 --output /tmp/after-ddqn.json
```

Replace `ddqn` with `dqn` or `qlearning` for the other learners. The reproduction
script saves its checkpoint beside the JSON output. The larger warm-up uses
`--episodes 0 --demo-games 1000 --demo-updates 3000`.

For a new expert-assisted training run with stronger warm-up, use a new output
path and play the `.best` checkpoint if later training regresses:

```sh
python3.12 -m training.train --agent ddqn --episodes 5000 --demo-games 1000 --demo-updates 3000 --save-path checkpoints/ddqn_win_agent.pt
python3.12 scripts/play.py --ddqn-checkpoint checkpoints/ddqn_win_agent.best.pt
```

This is a measured improvement to parts of the pipeline, not a completed solution
to strong autonomous card-game learning. Pure RL remains weak in these tests;
broader training distributions, better planning representations, and multi-seed
experiments remain open work.
