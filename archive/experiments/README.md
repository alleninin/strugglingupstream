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


`demo.py` is the old fixed greedy demonstration; use `scripts/watch.py --agent greedy` instead.
Older learner implementations and multi-agent commands now live in `archive/legacy/`.
The active project uses DDQN plus Greedy/Random. See [the archive index](../README.md).
