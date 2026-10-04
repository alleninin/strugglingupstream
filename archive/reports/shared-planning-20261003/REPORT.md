# Shared planning features and archive cleanup

DQN, Q-learning, shaped DDQN and shaped Q-learning now accept
`--partition-features` through the shared training CLI. DDQN keeps the existing
default (enabled); the others remain opt-in pending proper performance comparisons.

- DQN reuses `HandQNetwork`, including remaining-hand and partition features,
  while retaining DQN's target-network maximum and replay algorithm.
- Q-learning appends the same five partition statistics to its existing linear
  state-action interaction features. It remains a linear function approximator.
- Shaped wrappers pass the feature setting to their inner learner while retaining
  their distinct reward objective.
- Old and new checkpoints load automatically in the existing play/evaluation
  paths. Neural schema metadata selects the network. Linear `.npy` weights use
  their length to distinguish additive, interaction and partition schemas.
- The DDQN feature encoder and training defaults are unchanged by this task.

`--opponent curriculum` selects the same opponent schedule as DDQN; use
`--demo-games 0` if demonstrations should also be disabled. These flags do not
turn DQN or Q-learning into Double DQN.

## Validation

97 unit tests pass. New tests compare neural/linear partition statistics, check
old/new checkpoint prediction round-trips (including old additive linear weights),
and exercise demonstration plus RL updates with the new representations.

All four adapted learners completed 120-episode smoke runs, evaluation, save and
reload, followed by ten evaluation games. DQN/Q-learning used only three demo games
and four warm-up updates to exercise that path. These deliberately short runs are
integration checks, **not evidence of improved win rates**. Artifacts are local to
this report directory; no production checkpoint was replaced. See `tests.log`,
`smoke.log` and `smoke.json`.

The four archived CLIs all passed `--help` from their new locations, confirming
repository-root imports resolve. `git diff --check` passed.

## Cleanup

Moved three historical benchmark tools and the redundant shaped-training CLI into
`archive/experiments/`, adjusted repository-root resolution and updated report
commands. The archive README explains their purpose and current alternatives.
Existing historical reports and checkpoint data were preserved.

Removed 15 tracked Python bytecode files and one Finder metadata file, all already
covered by `.gitignore`. The exact list is in `removed-generated-files.txt`.

Kept DQN, Q-learning, shaped agents, Greedy and Random as supported agents/baselines.
Kept `agents/dueling_dqn.py`: older DDQN checkpoints and compatibility tests still
use it, so moving or deleting it would break loading existing bots.
