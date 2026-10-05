# Fading Greedy supervision — 2026-10-04

## Plan and implementation

Test whether directly teaching Greedy's move preferences improves sample efficiency,
while allowing DDQN to override those preferences later. Keep the existing network,
rewards, guided exploration, replay, target calculation and opponent curriculum.

`training/supervision.py` labels non-forced decisions from the learner's ordinary
training games with the existing public-information Greedy bot. A bounded buffer
retains up to 4,096 examples. Each DDQN update samples 16 examples using a separate
RNG and adds a large-margin ranking loss:

```
L = mean(max_a [Q(s,a) + 0.1 * (a != expert)] - Q(s,expert))
total_loss = existing_DDQN_loss + supervision_weight * L
```

The margin stops pushing once the expert move leads all alternatives by 0.1 Q
units. It does not replace Q targets with classification labels. The auxiliary
weight starts at 0.5 and fades linearly to zero over the first 60% of episodes;
afterward the example buffer is cleared and no further ranking supervision occurs.
Existing epsilon-guided exploration remains enabled in both experimental arms.
No separate demonstration games, pretraining phase, new bot class, dependency,
network architecture or checkpoint format is introduced. Inference never consults
the teacher. The default weight is zero, preserving existing training behavior.

## Strategy benchmark

`scripts/benchmark_strategy.py` compares every checkpoint on identical positions:

- 14 constructed, card-conserving fixtures covering finishing, combination
  preservation, blocking, contesting and bomb use. Accepted moves are explicit and
  results include both the selected and accepted moves.
- A bounded reservoir of non-forced decisions from 100 independent Greedy-vs-Greedy
  games (seed 920041, capacity 1,000; this seed produces 691 decisions).
- Both actual deployed choices and raw network choices, so the immediate-finish
  safeguard cannot conceal failures in the learned policy.

The fixture expectations express heuristic advice, not proofs of optimal play.
Greedy agreement and pressure-pass counts are diagnostics, not correctness labels
or win rates. Pressure means an opponent has no more cards than the current table
combination, with a legal response available. Every tested agent receives only its
own hand and public information. These positions never enter training or checkpoint
selection. This is a behavioral benchmark, not a counterfactual rollout solver.

## Experiment protocol

Train both arms for 2,500 episodes on each of seeds 0, 1 and 2. Both use the same
curriculum, guided exploration, CPU/one Torch thread and best-checkpoint selection
every 500 episodes on 100 validation games. Supervised runs add only
`--supervision-weight 0.5 --supervision-fraction 0.6`.

Test the selected checkpoints on 500 matched fresh deals against three Greedy
bots and another 500 against three Random bots, evaluation seed 920042. All models
share these test deals; averaging three training seeds does not make them 1,500
independent evaluation deals. Do not tune on this test set. Timings are recorded
while independent runs overlap, not as isolated performance benchmarks.

The seed-0 baseline reproduces every tensor in the previously supplied checkpoint
exactly, confirming the disabled path preserves that training result.

## Results and decision

| Training seed | Baseline vs Greedy | Supervised vs Greedy | Baseline vs Random | Supervised vs Random |
| --- | ---: | ---: | ---: | ---: |
| 0 | 30.8% | 40.4% | 91.8% | 91.4% |
| 1 | 29.0% | 36.8% | 89.4% | 93.0% |
| 2 | 30.8% | 33.8% | 90.2% | 92.2% |
| Mean | **30.2%** | **37.0%** | **90.5%** | **92.2%** |

Greedy control: 25.4% vs Greedy, 87.0% vs Random. Random control: 0.2%, 22.2%.
Supervision improves all three training seeds against Greedy, averaging +6.8
percentage points on this held-out set. These are encouraging results from three
training seeds and one shared set of 500 deals per opponent, not a guarantee of
performance against humans or unseen opponents. No hyperparameters were tuned on
these held-out results.

Best validation checkpoints occur at episode 2,500 for all baseline seeds, and
2,000 / 2,000 / 2,500 for supervised seeds 0 / 1 / 2. All selected supervised models
have already completed the fade; all six runs continued to 2,500 episodes.

On the same 691 sampled positions for each policy:

| Seed | Baseline Greedy agreement | Supervised agreement | Baseline pressure passes / 78 | Supervised pressure passes / 78 |
| --- | ---: | ---: | ---: | ---: |
| 0 | 322 / 691 | 379 / 691 | 14 | 11 |
| 1 | 331 / 691 | 398 / 691 | 18 | 10 |
| 2 | 329 / 691 | 344 / 691 | 14 | 9 |

Passes where Greedy contests decline from 138 / 100 / 110 to 112 / 95 / 105,
with 402 eligible following decisions per policy. Leading decisions and forced
passes are excluded from that denominator. Agreement and pressure-pass changes
are descriptive proxies; they do not prove that each changed decision is better.

The constructed fixtures expose an important weakness. Baseline models pass
9 / 8 / 7 of the 14 checks; supervised models pass 7 / 7 / 9. Across the three
models, combination checks decline from 10 / 15 to 3 / 15, while contesting checks
improve from 1 / 6 to 6 / 6. These are repeated evaluations of the same fixtures,
not independent cases. The supervised seed-0 raw network also misses one of the
three constructed finishes; the deployed safeguard correctly takes it. No model
misses any of the 10 finishing opportunities in the shared natural positions.

Keep the experiment **opt-in** and leave the supplied checkpoint/defaults unchanged.
The stronger Greedy win rate is useful, but this is not a complete solution to
combination preservation or blocking. The present comparison does not isolate
whether the remaining weaknesses come from the fade schedule, training-state
coverage or other learning limitations.

All six selected checkpoints, training logs, aggregate metrics (`results.json`)
and detailed behavioral results (`strategy.json`) are archived here. The seed-0
supervised checkpoint is the reproducible example to try, not a seed chosen after
seeing the held-out scores:

```bash
python3.12 scripts/play.py --ddqn \
  --checkpoint archive/reports/ddqn-supervision-20261004/supervised-0.best.pt
```

Observed training / validation seconds while runs overlapped:

| Run | Training seconds | Validation seconds |
| --- | ---: | ---: |
| baseline-0 | 200.9 | 36.7 |
| baseline-1 | 209.0 | 36.9 |
| baseline-2 | 214.6 | 36.8 |
| supervised-0 | 249.2 | 34.0 |
| supervised-1 | 250.6 | 35.7 |
| supervised-2 | 238.6 | 34.2 |

These timings include supervision work and exclude held-out testing. They are not
isolated benchmarks. Scratch checkpoints and smoke outputs remain in
`/tmp/zheng-supervision`; the retained experiment is self-contained here.

## Verification

111 active tests and 97 archived tests pass, along with the standalone rules
suite, Ruff F/E9/I checks and formatting checks. New coverage checks exact margin
values/gradients across variable legal sets, teacher learning, isolated RNG use,
bounded owned examples, fade/disable behavior, checkpoint round trips, history and
dueling updates, physical fixture validity, hidden-card isolation, raw-versus-
deployed finishing behavior, forced-pass exclusions and deterministic shared states.

CLI smoke checks passed for play, watch, mixed tournaments and the strategy
benchmark using supervised checkpoints. A combined 10-episode smoke run also
exercised supervision with history, Monte Carlo targets, anchored self-play and
endgame practice enabled. No feature needs additional inference flags.

## Reproduce

```bash
python3.12 -m training.train --episodes 2500 --seed 0 \
  --supervision-weight 0.5 --supervision-fraction 0.6 \
  --save-path checkpoints/supervised.pt

python3.12 -m training.train --episodes 2500 --seed 0 \
  --save-path checkpoints/baseline.pt

python3.12 scripts/evaluate_ddqn.py \
  checkpoints/baseline.best.pt checkpoints/supervised.best.pt \
  --games 500 --seed 920042 --include-baselines --output reports/supervision-evaluation.json

python3.12 scripts/benchmark_strategy.py \
  checkpoints/baseline.best.pt checkpoints/supervised.best.pt \
  --games 100 --seed 920041 --output reports/supervision-strategy.json

python3.12 scripts/play.py --ddqn --checkpoint checkpoints/supervised.best.pt
```

Repeat training with seeds 1 and 2 and distinct save paths for the full comparison.
