# DDQN strategy learning — 2026-10-04

The retained change improves the training examples: half of epsilon exploration
uses the existing Greedy bot, half remains uniform random exploration. Network
exploitation keeps the same probability. DDQN still receives actual game rewards
and learns its own action values. No Greedy action policy is consulted by DDQN during play or
evaluation. A separate exact safeguard always takes a legal hand-emptying move.
No new bot, network architecture, or runtime dependency was added.

## Diagnosis

A 200-deal audit of the previous supplied model found 882 passes where Greedy would
contest, among 2,894 decisions. Disagreement alone does not establish an error.
More concretely, it once played triple 999 instead of immediately finishing with
full house 99922. Its approximated action values had no exact-finish safeguard.

The existing partition representation exposes combo preservation; the learner
still has to discover useful multi-turn sequences through exploration and delayed
rewards. Pure uniform exploration disrupts those sequences. The guided-versus-
unguided training comparison below supports training experience as a contributing
bottleneck, without proving it is the only cause. Rewards already telescope to
the intended first-place objective; they were preserved.

## Matched evaluation

All learners trained for 2,500 episodes, seeds 0/1/2, same random→mixed→Greedy
curriculum, epsilon schedule, optimizer, replay and partition features. Validation
selected the best checkpoint every 500 episodes on 100 separate deals. Evaluation
uses 500 held-out deals per opponent, seed 8100426, epsilon zero, versus three
Greedy bots or three Random bots. These same 500 deals are shared across training
seeds, not 1,500 independent deals. No hyperparameters were tuned on these results.

| Variant | Seed 0 vs Greedy | Seed 1 | Seed 2 | Mean vs Greedy | Mean vs Random |
| --- | ---: | ---: | ---: | ---: | ---: |
| Previous partition DDQN | 27.4% | 21.8% | 27.2% | 25.5% | 84.7% |
| Finish safeguard only, retrained | 28.0% | 27.2% | 22.8% | 26.0% | 88.8% |
| Guided exploration + safeguard | 36.2% | 26.8% | 31.6% | 31.5% | 89.5% |
| Extra control features (rejected) | 28.0% | 19.4% | 16.0% | 21.1% | 86.3% |

Greedy control: 26.2% vs Greedy, 86.4% vs Random. Random: 0.2%, 26.2%.
Previous-checkpoint evaluation also enables the finish safeguard, making the
weight comparison conservative. The finish-only row retrains with that safeguard
and pure random exploration, isolating the additional effect of guidance. Control-
feature experiments trained without the safeguard but used it during evaluation;
they exposed urgency and public-card overcall possibilities and were discarded.
Their lower performance does not show that public-card reasoning is useless.

The published model remains training seed 0 (the existing default), selected by
validation, not by choosing the best held-out seed. It scores **36.2% vs Greedy
and 91.6% vs Random**, versus the previous supplied model's **27.4% / 88.8%**.
Guidance improved two of three seeds over the finish-only retraining; seed 1
was essentially unchanged. Seed variation remains substantial. These experiments do not establish performance
against humans or all unseen opponents.

## Behavior and mixed tournament

On diagnostic deals (seed 78103, 200 games), using the safeguard in both models:
- Passes when Greedy contests: 882 → 518.
- Urgent-contest passes: 150/324 (46.3%) → 77/238 (32.4%). Here "urgent" means a
  legal non-pass exists and an opponent holds at most the table move's card count.
  This is a pressure proxy, not a proof that every such pass is wrong.
- Wins: 58 → 70. Policies visit different states, so decision counts differ.
- No immediate finishes were missed once the safeguard was enabled.

The normal `evaluate.py --games 400` matchup contains two DDQN seats, one Greedy
and one Random; 100 deals rotate through four seats. The new model scores 35.375%
per DDQN appearance (283/800), Greedy 29.0% (116/400), Random 0.25% (1/400).
Mean places: 1.990, 2.190, 3.830. This is a different matchup from the fixed-opponent
benchmark and its rotated games are correlated.

## Use and reproduce

The supplied checkpoint is `checkpoints/ddqn_agent.best.pt`. The previous supplied
weights are preserved at `archive/checkpoints/ddqn_v4_agent.best.pt` (SHA-256
`56a9588f40d79d2c40856a70edec57927cf8cd359556f7d4385126c02da23ae0`).

```bash
python3.12 scripts/play.py --ddqn
python3.12 -m training.train --episodes 2500 --seed 0 --save-path checkpoints/my_run.pt
# Ablation: same trainer, pure random exploration
python3.12 -m training.train --episodes 2500 --seed 0 --expert-exploration 0 --save-path checkpoints/unguided.pt
python3.12 scripts/evaluate_ddqn.py checkpoints/ddqn_agent.best.pt --games 500 --seed 8100426 --include-baselines --output reports/evaluation.json
```

Repeat these commands with seeds 1 and 2 for the guided and finish-only rows. The experimental exploration
wrapper and final production helper produced exactly identical weights over a
100-episode reproduction check including periodic evaluation. 86 active tests and
97 archived tests pass; play completed a scripted game, watch loaded and played
three games, and the full 400-game tournament completed. Experiment checkpoints
and scratch logs stay in `/tmp/zheng-strategy-fix` to avoid adding unused bots and
weights to the project. `results.json` preserves the aggregate measurements.

## Subsequent code review and cleanup

Reviewed active agents, bots, game engine, environment, trainer, CLIs and tests.
Historical archived implementations were preserved. The cleanup:

- Shares inference bot loading between play, watch and fixed-opponent evaluation.
- Builds feature vectors directly, avoiding many temporary scalar arrays.
- Stops win-rate evaluation when first place is decided.
- Removes unused Greedy group metadata, its duplicate embedded self-test, unused
  imports, redundant directory creation, and standalone/inline Python comments.
- Keeps concise docstrings for non-obvious algorithms and applies consistent formatting.
- Rejects invalid bot actions instead of silently substituting another move.
- Rejects empty evaluations and nonpositive watch-game counts; limits checkpoint
  exception handling to expected loading failures.

Verification: 89 active tests, 97 archived tests, the standalone rules suite,
Ruff F/E9/I checks and formatting checks pass. State vectors and 6,492 move vectors
matched the pre-cleanup implementation exactly over 12 complete games with
2–5 players and one/two decks. A seeded 100-episode training run produced identical
weights before and after cleanup. Play, watch, evaluation and tournament CLIs pass
smoke checks, including their invalid-input paths.

On 100 matched held-out games, both evaluators returned 47% for the supplied model;
full-playout evaluation took 7.60 seconds and first-finish evaluation took 5.06
seconds on this machine (about 33% faster, a single local timing sample).
DDQN outcomes are unchanged. The earlier Random-control figure is a historical
measurement: early termination changes that control's cumulative RNG consumption
between deals, so its exact percentage need not reproduce under the faster evaluator.
