# Learning audit — October 3, 2026

## Measured performance

Each policy played the same 300 held-out deals per opponent type, with exploration
disabled, four players and two decks, evaluation seed 731926. Opponents were either
three greedy bots or three random bots. Checkpoints were snapshotted before testing;
source paths and SHA-256 hashes are in `results.json`. Original checkpoints were not
modified. Saved weights contain no episode counter, so the exact training episode
of each snapshot cannot be recovered from the file.

| Policy | Wins vs greedy | Rate vs greedy | Rate vs random |
| --- | ---: | ---: | ---: |
| DQN demonstration-only warm-up | 34/300 | 11.3% | 80.0% |
| DQN best saved checkpoint | 35/300 | 11.7% | 73.7% |
| DQN latest saved checkpoint | 25/300 | 8.3% | 73.7% |
| DDQN best saved checkpoint | 51/300 | 17.0% | 82.3% |
| Greedy control | 77/300 | 25.7% | 87.3% |
| Random control | 1/300 | 0.3% | 26.0% |

The warm-up was reconstructed with CLI defaults: seed 0, 200 demonstration games,
1,000 minibatches, learning rate 0.0003. It is the relevant pre-RL comparison if the
user trained with default seed/demo settings. This is one run, not a multi-seed
ablation. A one-win difference does not establish learning improvement; the latest
checkpoint's lower score is suggestive of regression, not definitive on 300 games.
At an 11.7% rate, a rough binomial 95% interval is about 8–15%.

DQN is substantially better than random, but RL has not demonstrated an improvement
over its warm-up on this test. DDQN is stronger here but still below the greedy
control. Four evenly matched players would each average about 25%, not 50%.

## What the implementation explains

1. **The recent DDQN representation improvements were not applied to DQN.**
   DQN concatenates raw state and move features into a small MLP. DDQN explicitly
   supplies the remaining hand and combination counts through `HandQNetwork`.
   DQN must infer combination preservation from raw counts and game outcomes.
   The prior DDQN report contains representation ablations supporting this as an
   important limitation; this audit does not isolate its effect on DQN.
2. **Demonstration warm-up produces action rankings, not calibrated returns.**
   `Demonstrations.loss` applies cross-entropy directly to the same Q outputs used
   for Bellman targets. Cross-entropy can reward larger gaps even when those scores
   no longer represent attainable rewards. Warm-up copies these outputs to the
   target network. Rehearsal continues this pressure with weight 0.1 during RL.
   This is a concrete objective mismatch, though its causal impact on win rate
   needs an ablation before selecting a replacement.
3. **DQN has a more aggressive, shorter-horizon update schedule.** It starts after
   64 replay transitions and updates every transition using one-step target-network
   maxima. Default DDQN waits for 1,000 transitions, updates every four transitions,
   uses three-step returns and online selection followed by target evaluation.
   DQN starts fitting a small, mostly losing experience pool; its max bootstrap
   also propagates optimistic value estimates. These are algorithm/configuration
   differences, not proof that the Bellman update itself is coded incorrectly.
4. **Training progress is a noisy, exploratory metric.** Epsilon is the probability
   of a random action at each decision, not of a random whole game. One bad choice
   can ruin a combination. The posted 100-game training batches therefore understate
   exploitation performance, but the independent evaluation confirms weakness even
   with epsilon zero. More episodes alone are not a demonstrated remedy.

## Decision/value probe

A separate common set of 651 non-forced decisions was collected from 100 greedy
teacher games using seed 982711. These are teacher-visited states, not the models'
own game trajectories. `probe.json` records all measurements.

| Policy | Teacher agreement | Pass when pass and another move available | Maximum selected Q |
| --- | ---: | ---: | ---: |
| DQN warm-up | 56.8% | 9.9% | 10.37 |
| DQN best | 47.6% | 27.8% | 7.43 |
| DQN latest | 45.5% | 33.9% | 13.58 |
| DDQN best | 37.2% | 30.9% | Not measured |

With the default undiscounted win reward, remaining return is terminal +/-1 plus
current hand size / initial hand size; its upper bound is 2. DQN's values are far
outside that bound, including before RL. This directly demonstrates value-scale
miscalibration; it does not prove that every move with an inflated value is bad.
RL also shifts DQN toward more passing on these states. Disagreement with greedy
or passing is not itself an error: DDQN agrees less yet wins more.

## Next implementation work supported by this audit

- Test the remaining-hand representation in DQN while retaining the DQN target
  rule, so architecture and algorithm are not conflated.
- Separate imitation scores from the value head, or test bounded-margin imitation
  and value calibration before bootstrapping. Compare against no demonstrations.
- Ablate replay warm-up/update frequency and multi-step returns individually.
- Track held-out wins and value calibration, retain best checkpoints, and validate
  any selected change across multiple training seeds and a fresh final test set.

No training algorithm or defaults were changed in this diagnostic task. There is
no new Q-learning win checkpoint in the current directory, so this audit makes no
fresh claim about its current learning curve. Its existing linear interaction model
also lacks DDQN's explicit combination features.

## Reproduce

From the repository root (rerunning replaces this audit's snapshots/results):

```bash
python3.12 -B reports/learning-audit-20261003/evaluate.py
python3.12 -B reports/learning-audit-20261003/probe.py
```

Evaluation includes complete games and leaves model exploration/RNG state intact.
The evaluation and probe completed successfully. No game or learner code changed.
