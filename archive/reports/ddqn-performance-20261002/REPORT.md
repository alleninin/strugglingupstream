# DDQN training performance — October 2, 2026

The main avoidable cost was processing the same state separately for every legal
move. Replay also stored repeated state features, and learning assembled many
small per-sample operations in Python. The original progress timer showed
cumulative time, which could be mistaken for time per reporting interval.

The updated network encodes each state once, batches variable-length legal move
sets, and computes action selection and centered advantages with tensor
operations. Replay stores each state once and caches the chosen action index.
Training defaults to CPU with one Torch thread; explicit device and thread flags
allow comparison on other machines. Logs now separate interval training time,
cumulative training time, and evaluation time.

## Measurements

One run per configuration, 100 episodes, seed 42, four players, two decks, greedy
opponents, CPU, Torch 2.13.0. Evaluation, checkpoint writing, and initialization
are excluded from the timed loop.

| Implementation | CPU threads | Seconds | Transitions | Learning updates | Replay array bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| Before | 8 | 25.60 | 2,151 | 2,088 | 7,370,508 |
| Before | 1 | 22.03 | 2,151 | 2,088 | 7,370,508 |
| After | 1 | 9.69 | 2,112 | 2,049 | 3,577,744 |

This is approximately 2.64 times faster than the original eight-thread run and
2.27 times faster at the same thread count. Replay NumPy payload is approximately
51% smaller; this is not a measurement of total process memory. Floating-point
rounding can change subsequent decisions and trajectories, so transition counts
differ slightly. The roughly 1.9% reduction in updates does not explain the speedup.

The reported two-minute runtime was not reproduced here. MPS and CUDA were not
available, so accelerator performance remains unmeasured. Larger game settings,
evaluation frequency, device selection, and machine load can affect actual timing.
These are short benchmark runs, not repeated statistical performance estimates.

## Correctness and compatibility

- All 67 unit tests pass, including comparisons of grouped Q values, gradients,
  Double DQN targets, terminal transitions, padding, and action selection.
- Existing DDQN checkpoint loads and agrees with the original implementation on
  decisions from 20 held-out deals.
- DDQN, shaped DDQN, and DQN training CLI smoke runs complete successfully;
  DDQN evaluation and checkpoint saving were also exercised.
- Learning still updates every eligible transition with batch size 64 and the
  same target synchronization cadence. Network parameter names and shapes are
  unchanged. No existing trained checkpoints were overwritten by this work.

## Reproduction

Run from the repository root:

```sh
python3.12 -B archive/experiments/benchmark_ddqn.py --episodes 100 --threads 8 --device cpu --reference reports/ddqn-performance-20261002/ddqn_before.py --json /tmp/ddqn-before.json
python3.12 -B archive/experiments/benchmark_ddqn.py --episodes 100 --threads 1 --device cpu --json /tmp/ddqn-after.json
python3.12 -B -m unittest discover -s tests -q
```

Raw measurements are in `before-8-threads.json`, `before-1-thread.json`, and
`after-1-thread.json`. `profile-before.txt` contains a separate 30-episode profile
with profiler overhead. `ddqn_before.py` and `network_before.py` preserve the
pre-optimization implementations. The reference benchmark uses the original
forward method, which remains unchanged in the current network class.

Restart any running training process to load the optimized code. For example:

```sh
python3.12 -m training.train --agent ddqn --episodes 5000 --device cpu --torch-threads 1 --progress-every 100 --save-path checkpoints/ddqn_fast_agent.pt
```
