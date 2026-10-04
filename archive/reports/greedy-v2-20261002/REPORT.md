# Greedy strategy and retraining experiment

Five learners trained from scratch for **1000 episodes each**, against a seeded mixture of greedy and random opponents (four players, two decks).

Greedy and Random are fixed policies and need no training. GreedyBefore is the saved pre-change strategy. All evaluated learners were reloaded from the new checkpoints with exploration disabled.

## Tournament

840 games; all four-player subsets, 3 rounds, every lineup rotated through all four seats on the same deal. Starting seat is 0 and every participant takes that seat in each block. Shuffle seeds are held out from training. Results are descriptive; seat rotations share deals and are not independent trials.

| Bot | Appearances | First place | Mean place |
|---|---:|---:|---:|
| Greedy | 420 | 59.8% | 1.600 |
| GreedyBefore | 420 | 56.0% | 1.686 |
| DQN | 420 | 22.4% | 2.386 |
| ShapedQL | 420 | 18.1% | 2.414 |
| QLearning | 420 | 20.0% | 2.569 |
| DDQN | 420 | 16.7% | 2.650 |
| Shaped | 420 | 4.3% | 3.214 |
| Random | 420 | 2.9% | 3.481 |

## Fixed-opponent comparisons

Each row uses 50 new deals × four focal seats against three copies of the specified opponent. All focal bots receive the same deal seeds.

| Bot | vs 3 Greedy: first | vs 3 Greedy: mean | vs 3 Random: first | vs 3 Random: mean |
|---|---:|---:|---:|---:|
| QLearning | 4.5% | 3.320 | 69.0% | 1.435 |
| DQN | 16.5% | 2.765 | 67.5% | 1.470 |
| DDQN | 11.0% | 3.045 | 57.5% | 1.695 |
| Shaped | 0.5% | 3.635 | 23.0% | 2.250 |
| ShapedQL | 7.0% | 3.205 | 78.0% | 1.320 |
| Greedy | 25.0% | 2.500 | 91.0% | 1.110 |
| Random | 1.0% | 3.730 | 25.0% | 2.500 |
| GreedyBefore | 22.0% | 2.475 | 96.0% | 1.055 |

New Greedy against three previous Greedy bots: **24.5%** first place, mean placement **2.635** over 200 games.

## Interpretation

- New Greedy leads this mixed pool (59.8% first place versus 56.0% for the previous policy), but the fixed-opponent tests show a trade-off: 91% versus 96% against Random, and 24.5% against three copies of the old Greedy. These results do not establish universal superiority.
- DQN is the strongest trained agent in this mixed tournament and against three new Greedy opponents. Shaped Q-learning does best among the learners against Random (78%).
- Shaped DDQN is weak at this training budget (4.3% in the tournament and 23% against Random). Its extra reward terms are not demonstrated to help here.
- The behavioral fixes are independently covered by tests: preserve pairs and straights, play intact combinations before low orphans, take an immediate finish, and block one-card threats.
- Validation passed: 51 unit/regression tests, the existing rule checks (including 50 full games), and an end-to-end training/evaluation smoke run. The final artifacts contain 840 tournament games plus 3,400 fixed-opponent games. Every evaluated learner checkpoint contains finite weights and was successfully reloaded.

## Training and limits

| Agent | Episodes | Seconds | Checkpoint |
|---|---:|---:|---|
| QLearning | 1000 | 36.5 | `checkpoints/greedy-v2-20261002/qlearning_agent.npy` |
| DQN | 1000 | 59.3 | `checkpoints/greedy-v2-20261002/dqn_agent.pt` |
| ShapedQL | 1000 | 44.0 | `checkpoints/greedy-v2-20261002/shaped-ql_agent.npy` |
| DDQN | 1000 | 250.9 | `checkpoints/greedy-v2-20261002/ddqn_agent.pt` |
| Shaped | 1000 | 296.9 | `checkpoints/greedy-v2-20261002/shaped_agent.pt` |

This is one training seed and a finite training budget, not evidence of convergence or strength against human players. The small-hand solver minimizes the number of combinations without modeling opponent cards. Larger-hand estimates remain heuristic. Old checkpoints are preserved. No policy was selected using this held-out tournament.

Machine-readable configurations, source hashes, training times, and every game result are in `experiment.json`, `tournament.json`, and `baselines.json`.
