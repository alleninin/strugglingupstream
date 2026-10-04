# DDQN hand-partition fix — October 3, 2026

## Held-out results

Each policy played the same 500 held-out deals against three greedy opponents,
and another matched 500 against three random opponents. Exploration was disabled;
evaluation seed was 920261003, separate from training, validation and the earlier
audit. The feature definition was selected before these tests. The third training
seed was added after seeing mixed first/second-seed results; no feature or optimizer
change was made in response to the held-out scores.

| Training seed | Baseline vs greedy | Revised vs greedy | Baseline vs random | Revised vs random |
| --- | ---: | ---: | ---: | ---: |
| 0 | 21.4% | 31.0% | 79.2% | 88.6% |
| 1 | 22.6% | 21.2% | 88.8% | 80.0% |
| 2 | 18.4% | 24.8% | 79.0% | 88.8% |
| Mean across seeds | 20.8% | 25.7% | 82.3% | 85.8% |

Greedy control: **25.8% vs greedy, 86.4% vs random**. Random control: **0.2% vs
greedy, 22.4% vs random**. Raw rates are in `comparison.json` and individual
`final-*.json` files; multiply by 500 to recover win counts.

The representation improved two of three training seeds. The mean greedy-opponent
score rose from **20.8% to 25.7%**, but seed 1 fell from 22.6% to 21.2%. This is an
improvement in the tested recipe, not a guarantee for every run, nor evidence of
consistent superiority over greedy. These are only three training seeds, sharing
500 evaluation deals; the mean is not based on 1,500 independent deal types.
Training seeds change initialization/exploration; the existing deal-seed ranges
also overlap between training runs. More independent runs would better estimate
reliability. No blanket claim that longer training fixes the remaining variance
is made.

The supplied model uses the normal CLI **seed 0**, selected as the shipping seed
before final evaluation. It won **155/500 (31.0%)**, versus **107/500 (21.4%)** for
the same-seed baseline. Seed 1's higher validation score did not translate to the
best held-out performance, illustrating why a separate test set matters.

## What changed

The retained fix adds five features describing disjoint partitions of the player's
own hand, before and after each candidate move: estimated plays remaining, change
in that estimate, remaining singles in the heuristic partition, and change in
singles. Previously the network counted pairs/triples/runs that could overlap,
without directly seeing how many separate plays those cards actually require.
For example, playing a spare 2 leaves a 34567 straight as one play; playing the 3
instead leaves five singles. The new representation makes that distinction explicit.

For larger hands, the estimate chooses the better of two legal greedy partitions.
At eight cards or fewer, the play-count feature uses exact minimum-partition search.
The single count remains a heuristic statistic even for small hands. These estimates
ignore opponents and cannot predict trick control or guarantee the best action.
The network still learns action values and chooses moves; there is no greedy action
fallback and no hidden-card access. No demonstrations were used in these experiments.

**The win reward was left unchanged.** The learner already receives +1 for winning
and -1 for losing, with card-count potential shaping and terminal potential zero.
At discount 1, total shaped return differs from terminal outcome by a constant for
each starting state. Arbitrary pass penalties or card-shedding bonuses would change
that objective. The experiments below support improving the representation before
replacing the objective.

Hand-partition features are enabled by default in the DDQN training CLI, along with
existing planning features. `--no-partition-features` reproduces the prior input
representation; `--no-planning-features` disables both. Other learners' defaults are
unchanged. The feature schema is saved in checkpoints, so old and new models load
with their original representation. A new checkpoint cannot upgrade old weights.

## Controlled training

Four players, two decks; seeds 0, 1 and 2; 2,500 episodes each. Both conditions used
identical curriculum, reward, learning rate, three-step returns, prioritized replay,
exploration schedule and evaluation cadence. Validation used 100 games against
greedy every 500 episodes; models were selected by validation, not final-test scores.
Baseline seeds 0 and 1 selected episode 2,500 at 23% validation wins. The new
seeds 0 and 1 selected episode 2,500 at 27% and 31%, respectively. Seed 2 was
added after the first two held-out comparisons showed mixed results; it uses
the same unchanged feature definition and training settings. Its baseline selected
episode 2,500 at 22%; its revised model selected episode 2,000 at 30%.

CPU, one Torch thread. Observed training-only times were 150.9/145.0/144.7 seconds for the
baselines and 240.5/215.0/213.0 seconds for partition features. Runs overlapped, so these
are not isolated speed benchmarks. The new estimator adds CPU work; partition
results are cached with a bounded cache. GPU speed was not measured.

## Alternatives tested but not retained

All exploratory alternatives below used training seed 0 and 2,500 episodes, with
unchanged validation deals. They were compared before the final test. They are
not additional production flags or promised improvements.

| Alternative | Best validation vs greedy |
| --- | ---: |
| Baseline | 23% |
| Partition features (retained) | 27% |
| Combo-count potential / initial combo estimate | 13% |
| Combo-count potential / (initial hand size / 3) | 12% |
| Bound Q to hand fraction plus a tanh outcome | 24% |
| Cut multi-step traces before exploratory continuations | 22% |

Combo potentials preserved the terminal objective through potential differences,
but made learning worse in these runs. The initial-combo normalization additionally
introduced a deal-specific scale absent from the observation; the fixed-scale run
removed that issue and still failed to improve. Bounded values were too close to
the baseline on a small validation set to justify an extra change. Trace truncation
addresses exploratory rewards entering multi-step targets, but did not help this
experiment. None of these comparisons proves the alternatives can never work.

Logs and checkpoints retain these exploratory results. `exploratory.patch` records
the prototype implementation relative to the pre-fix source (fixed-scale reward
variant); the initial-normalization variant is specified above. The final loader
intentionally does not support the discarded bounded-value prototype.

## Validation and use

- 93 unit tests pass, including gradient/replay updates with partition features,
  straight preservation, joker handling, empty hands, and old/new checkpoint
  prediction compatibility. Existing reward and Double-DQN target tests still pass.
- The standalone rule checks pass, including 50 terminating random games.
- The training CLI completed a two-player smoke run with the new default, including
  evaluation and checkpoint saves.
- `play.py` completed a full four-player game loading the new checkpoint without
  warning or greedy fallback; scripted human inputs chose the first legal option.
- The ordinary and best seed-0 models are copied to `checkpoints/ddqn_v4_agent.pt`
  and `checkpoints/ddqn_v4_agent.best.pt`. The source files and hashes are recorded
  in `published-checkpoints.json`. Existing user checkpoints were not overwritten.
- The concise README now uses the new paths and documents the opt-out flag.

Train a new model (choose a different path to preserve the supplied checkpoint):

```bash
python3.12 -m training.train --agent ddqn --episodes 2500 --seed 0 \
  --save-path checkpoints/my_ddqn.pt
```

Add `--no-partition-features` to reproduce the baseline. Repeat with `--seed 1` or `--seed 2`
and distinct save paths for the additional training runs.

Play the supplied model:

```bash
python3.12 scripts/play.py --ddqn-checkpoint checkpoints/ddqn_v4_agent.best.pt
```

Reproduce held-out evaluation, without retraining:

```bash
python3.12 -B scripts/evaluate_ddqn.py \
  reports/ddqn-fix-20261003/baseline-0.best.pt \
  reports/ddqn-fix-20261003/baseline-1.best.pt \
  reports/ddqn-fix-20261003/baseline-2.best.pt \
  reports/ddqn-fix-20261003/partition-0.best.pt \
  reports/ddqn-fix-20261003/partition-1.best.pt \
  reports/ddqn-fix-20261003/partition-2.best.pt \
  --games 500 --seed 920261003 --include-baselines \
  --output /tmp/ddqn-partition-evaluation.json
```
