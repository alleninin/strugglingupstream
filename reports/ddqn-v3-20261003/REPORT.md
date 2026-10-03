# DDQN v3 investigation and validation

## Outcome

The final models were trained **from scratch, without demonstrations**, for 2,500
games each. On a fresh 500-deal test set, they won **21.0% and 19.8%** against three
greedy opponents. The user's existing `ddqn_win_agent.best.pt` won **10.6%** on the
same deals. This is a substantial improvement, but remains below the greedy
control's **25.4%**. Two training seeds are encouraging evidence, not a guarantee
of identical results on every run.

The main improvement came from exposing the hand remaining after a candidate
move, including its combination structure. Simplifying the network or training
the plain network longer was not sufficient by itself.

## Reference repository

Inspected [zheng-zero](https://github.com/tcmmichaelb139/zheng-zero) at commit
`8328a1e3ea85d9fd1e9e2881e9d464e2864fffb3`. Its
[network and replay implementation](https://github.com/tcmmichaelb139/zheng-zero/blob/8328a1e3ea85d9fd1e9e2881e9d464e2864fffb3/ZhengShangYou/zhengzero/zhengzero.py)
uses a scalar action-value network, public played-card history, and five
512-unit hidden layers. The useful ideas here were richer public observations
and straightforward action scoring. Its target uses a target-network maximum,
not online selection followed by target evaluation; its replay construction also
omits the last transition. Those details were not copied.

Its reported 84% result uses different rules and a different opponent:
[random exploration](https://github.com/tcmmichaelb139/zheng-zero/blob/8328a1e3ea85d9fd1e9e2881e9d464e2864fffb3/ZhengShangYou/players/base_player.py)
heavily favors longer plays. The actual
[reward code](https://github.com/tcmmichaelb139/zheng-zero/blob/8328a1e3ea85d9fd1e9e2881e9d464e2864fffb3/ZhengShangYou/env/zhengshangyou.py)
pays 0.25 per non-pass action and a 0.75 winner bonus, rather than a bonus scaled
by cards shed. Its published number is not a benchmark against this project's
combination-preserving greedy policy. No third-party implementation was copied.

## Changes

- **Public observations:** append played-rank counts, the table's full rank
  counts, relative table ownership, and consecutive passes. The old observation
  prefix is retained so old checkpoints can ignore the added fields. Opponents'
  hidden cards are never encoded.
- **Action representation:** a small scalar Q network receives normalized state
  and action features, the remaining rank histogram, and counts describing
  singles, pairs, triples, bombs, runs, high cards, and broken bombs. Jokers are
  treated as singles, and runs exclude 2/jokers. These features neither select an
  action nor call the greedy bot. The network learns their values from gameplay.
- **Simpler architecture:** Double DQN remains online argmax plus target-network
  evaluation; new models no longer require dueling value/advantage streams. The
  resulting `HandQNetwork` has 24,065 parameters for four-player games.
- **Credit assignment:** three-step returns, with the correct bootstrap discount
  and terminal queue flush. A last move is never discarded. Abandoned episodes
  clear pending returns rather than mixing deals.
- **Replay use:** collect 1,000 transitions before learning, then update once per
  four transitions. Target synchronization counts gradient updates. For scalar
  Q, store/score only the selected current action and evaluate only the selected
  target action; scoring all current legal actions was unnecessary.
- **Training distribution:** DDQN defaults to 20% random opponents, 30% mixed,
  then 50% greedy. Exploration remains scheduled by episode. Default DDQN no
  longer requires demonstration warm-up. Other learners retain their prior
  default opponent/demo choices.
- **Evaluation isolation:** restore both NumPy and Python RNG state after
  evaluation, so reporting cannot alter future exploration draws.
- **Compatibility:** old dueling DDQN, DQN, and linear checkpoints load with their
  original input prefix. New planning checkpoints identify their feature schema.
  Loading weights resets stale replay/optimizer state; this is not a full
  training-resume checkpoint.

The environment's first-place reward is retained. Optional discount changes use
the same discount for potential shaping and the learner. Greedy's observation
parser now reads exactly the opponent-size fields, rather than treating appended
history as additional opponents. Existing greedy behavior tests pass.

## Final test protocol

Training seeds: 0 and 1. Four players, two decks. 2,500 RL games per model,
zero demonstration games, three-step returns, replay warm-up 1,000, one gradient
update per four transitions, batch size 64, target copy every 200 updates.
Validation: 100 greedy games every 500 episodes; save the best validation model.
Both final planning models selected episode 2,500, at 23% validation wins.

The final test uses **evaluation seed 98765**, 500 different deals per opponent
type, with exploration disabled. This set was reserved until the remaining-hand
representation was selected from validation results. Both new models and the
previous checkpoint see identical deals. No model is selected based on these
final test scores. The recommended checkpoint is seed 0, the normal CLI default.

| Policy | Wins vs greedy | Rate vs greedy | Rate vs random |
| --- | ---: | ---: | ---: |
| Previous user checkpoint | 53 / 500 | 10.6% | 80.2% |
| New DDQN, seed 0 | 105 / 500 | 21.0% | 81.4% |
| New DDQN, seed 1 | 99 / 500 | 19.8% | 89.2% |
| Greedy control | 127 / 500 | 25.4% | 86.4% |
| Random control | 5 / 500 | 1.0% | 23.0% |

Files: `final-planning.json`, `final-baselines.json`, and their logs.

A further test rotates the candidate through all four seats on each of 100 new
deals (400 games, seed 840000), always against three greedy bots:

| Policy | Wins | Win rate | Mean finish |
| --- | ---: | ---: | ---: |
| Previous user checkpoint | 34 / 400 | 8.5% | 3.078 |
| New DDQN, seed 0 | 78 / 400 | 19.5% | 2.668 |

The rotated games share deals within each four-game block; they should not be
treated as 400 independent deals for statistical confidence calculations.
Full records are in `rotated-previous.json` and `rotated-new.json`.

## Experiments that were not promoted

These helped distinguish representation problems from simply needing more
iterations. Their exploratory test set used seed 54321, separate from the final
test above. They are not directly comparable to final-set percentages.

| Configuration | RL games | Best validation vs greedy | Exploratory test vs greedy |
| --- | ---: | ---: | ---: |
| Scalar + public history, seed 0 | 2,500 | 9% | 7.4% |
| Scalar + public history, seed 1 | 2,500 | 16% | 6.4% |
| Dense per-card reward, mixed opponents, gamma .99, seed 0 | 2,500 | 6% | 7.6% |
| Same dense setup, seed 1 | 2,500 | 9% | 6.2% |
| Ten-step returns, seed 0 | 2,500 | 5% | Not tested |
| Ten-step returns, seed 1 | 2,500 | 5% | Not tested |
| Scalar without hand features, seed 0 | 10,000 | 11% | Not tested |

The longer scalar run ended at 3% validation wins, reinforcing that more training
alone did not solve the problem. The dense-reward experiment is only a related
ablation; it does not reproduce the reference repository's exact reward code.

Optional planning models with 1,000 demonstration games and 3,000 warm-up batches
also completed 2,500 RL games. Best validation rates were 24% and 26%. They were
not used for the main comparison, and no claim of a held-out improvement from
demonstrations is made. The simpler pure-RL setup is the new default.

Interrupted prototypes with temporary observation/parser issues are isolated in
`discarded-prototypes/` and excluded from every result. An early exploratory
evaluation stopped when the Random control exposed a Python-vs-NumPy RNG API
issue; the final evaluation above completed after that fix.

## Validation and use

- 89 unit tests pass, including legacy checkpoints, public-only observations,
  exact multi-step returns, terminal flushes, true Double-DQN selection, target
  synchronization, evaluation RNG restoration, and joker/combination features.
- A full game was exercised through the existing watch/play loading path.
- Previous user checkpoints were preserved. The ready-to-play seed-0 model is
  `checkpoints/ddqn_v3_agent.best.pt`; its source and SHA-256 are recorded in
  `published-checkpoints.json`. A matching latest checkpoint is also saved.
- README training commands remain consolidated in **Train the bots**. Startup
  prints `network=HandQNetwork`, the observation size, and replay schedule.

Reproduce the final comparison after training the two seeds with the README
command (choose distinct checkpoint paths):

```sh
python3.12 -B scripts/evaluate_ddqn.py checkpoints/ddqn_v3_agent.best.pt checkpoints/ddqn_win_agent.best.pt --include-baselines --seed 98765 --games 500 --output /tmp/ddqn-v3-evaluation.json
```

The pure-RL planning runs took approximately 137–138 seconds of training plus
32 seconds of validation on this CPU, while other experiments ran concurrently.
These are observed run times, not isolated speed benchmarks or GPU measurements.
