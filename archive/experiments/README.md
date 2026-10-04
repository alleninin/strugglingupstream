# Archived experiment tools

These scripts reproduce older experiments or retain an older command interface.
Run them from the repository root. Their settings describe historical experiments,
not necessarily the current recommended training recipe. Existing reports link here.

| Script | Purpose | Current entry point |
| --- | --- | --- |
| `benchmark_bots.py` | Greedy-v2 comparison with historical source snapshots | `evaluate.py` for tournaments |
| `benchmark_learning.py` | Earlier learning/demonstration comparisons | `python3.12 -m training.train` |
| `benchmark_ddqn.py` | Historical DDQN speed comparison | Training CLI timing logs |
| `train_shaped.py` | Redundant shaped-training CLI | `python3.12 -m training.train --agent shaped` or `shaped-ql` |

Each script still supports `--help`. Use checkpoint/output paths dedicated to the
experiment, and check its defaults before comparing results with current models.
Historical source snapshots and results remain in `reports/`. Unless explicitly
given a source snapshot, archived tools import the current engine and learners.

Active learners remain in `agents/`. In particular, `dueling_dqn.py` is still
required to load older DDQN checkpoints; it is not dead code. Greedy and Random
remain useful fixed opponents and evaluation controls.
