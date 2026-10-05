# Archive

The active project focuses on DDQN, Greedy and the Random training baseline.
Nothing in these folders is required to run the current trained model.

| Folder | Contents |
| --- | --- |
| `legacy/` | DQN, Q-learning, shaped agents, demonstration training, older multi-agent CLIs and their tests |
| `checkpoints/` | All previous checkpoints, including older DDQN versions and recent DQN experiments |
| `reports/` | Historical comparisons, logs, source snapshots, and evaluation output |
| `experiments/` | Older benchmark entry points |

The current model is `checkpoints/ddqn_agent.best.pt` in the project root, trained
with guided exploration. The previous supplied model remains at
`archive/checkpoints/ddqn_v4_agent.best.pt`. New training can create a latest/best
pair in the active checkpoint folder. See [the learning comparison](reports/ddqn-strategy-20261004/REPORT.md)
and the [fading-supervision experiment](reports/ddqn-supervision-20261004/REPORT.md).

Run archived tools from the **project root**, using explicit checkpoint paths:

```bash
python3.12 -m archive.legacy.train --agent dqn --partition-features --episodes 5000 --save-path archive/checkpoints/new_dqn.pt
python3.12 -m archive.legacy.play --dqn-checkpoint archive/checkpoints/dqn_partition_agent.best.pt
python3.12 -B -m unittest discover -s archive/legacy/tests -v
```

To run the earlier DDQN/DQN/Q-learning tournament:

```bash
python3.12 -m archive.legacy.evaluate --games 400 \
  --ddqn-path checkpoints/ddqn_agent.best.pt \
  --dqn-path archive/checkpoints/dqn_win_agent.best.pt \
  --q-path archive/checkpoints/qlearning_win_agent.best.npy \
  --json reports/legacy-tournament.json
```

Archived tools use the current game engine. Historical reports, logs and saved
source snapshots retain their original paths/configurations for provenance; paths
starting with `reports/` or old `checkpoints/` generally now need the `archive/`
prefix. They are records of prior experiments, not the current quick-start guide.
Use the root README for current commands. Old command entry points were preserved
here before simplification; they are no longer exposed by the main training CLI.

Historical cleanup verification: 80 active tests and 97 archived tests passed. Current play,
watch, tournament, fixed-opponent evaluation and training commands completed smoke
checks. A matched 20-episode run produced exactly the same DDQN weights before and
after simplifying the trainer. That cleanup preserved the then-supplied best-checkpoint bytes.
An additional DQN checkpoint written after the initial move was preserved under a
`dqn_partition_agent.later-*.pt` name rather than overwriting the earlier snapshot.
