# Zheng ShangYou (征上游)/Struggling Upstream AI Game Engine + RL Training

A from-scratch Python environment for the climbing card game **Zheng Shang You**
("Struggle Upstream"). Be the first to shed all your cards. There's an RL-friendly environment and several agents you can train and
pit against each other.

## Rules implemented
- **Card ranking (low→high):** `3 4 5 6 7 8 9 10 J Q K A 2` then **Black Joker**
  then **Red Joker**.
- **Combinations:** single, pair, triple, full house (triple + pair), straight
  (5+ consecutive ranks from `3` to `A`, never `2`/jokers), **airplane** (two
  triples of distinct normal ranks, including `2`, plus two other distinct pairs
  = exactly 10 cards, ranked by the higher triple), **consecutive pairs** (3+
  ranks), **consecutive triples** (2+ ranks), and a **four-of-a-kind bomb**.
  Consecutive pairs/triples run from `3` to `A`, excluding `2` and jokers.
- **Beating a play:** you must play a *higher* combo of the *exact same type and
  length*, **or** a bomb. A bomb beats any non-bomb and is itself beaten only by
  a higher bomb. Jokers are individual highest singles (they do not form a pair).
- **Passing & clearing:** pass your turn; if every other active player passes in
  a row, the table clears and the last player to play leads anew.
- **Struggling Upstream (penalty):** across deals, the last- and next-to-last
  finishers hand their single highest card (face-up) from the **newly shuffled
  deal** to the first- and second-place winners, respectively. With 2–3 players,
  only last place pays first place so winners and losers never overlap.

Default setup is **4 players / 2 decks** (27 cards each, no leftovers); both are
configurable (`num_players`, `num_decks`).
When the deck does not divide evenly, the first seats receive one extra card;
no cards are discarded.

## Layout
```
game/        rules engine: cards, moves, rules (turn loop), match (penalty)
env/         RL wrapper: feature encoders + ZhengShangYouEnv
agents/      RandomAgent, QLearningAgent, DQNAgent, DDQNAgent (PyTorch)
training/    train.py  (train one seat vs fixed opponents)
evaluate.py  round-robin tournament across agents
scripts/     demo.py   (print a full game)
tests/       regression tests for rules, rewards, replay, and training
```

## Install
```bash
pip install -r requirements.txt   # torch + numpy
```

## Run

To train a model, use [Train the bots](#train-the-bots) below. The commands here
are for checking the game, evaluating saved models, and playing.

```bash
# 1. Sanity-check the rules with a greedy-bot game
python3 scripts/demo.py

# 2. Tournament: learned agents / Greedy / Random (rotate seats on matched deals)
python3 evaluate.py --games 300
python3 evaluate.py --rounds 3 --json reports/tournament.json

# 3. Play against trained agents (per-seat checkpoints; human is always P0)
python3 scripts/play.py --ddqn-checkpoint checkpoints/ddqn_win_agent.best.pt
python3 scripts/play.py --shaped-checkpoint checkpoints/shaped_agent.pt --shaped-arch ddqn
# Mix fixed opponents, with no checkpoints required
python3 scripts/play.py --greedy --random
python3 scripts/watch.py --agent random --games 20
```

Use the same Python interpreter for installation and execution, e.g.
`python3 -m pip install -r requirements.txt`. On the current development machine,
`python3.12` has NumPy and PyTorch installed; the default `python3` does not.

Run the checks with that interpreter:
```bash
python3.12 -B scripts/test_rules.py
python3.12 -B -m unittest discover -s tests -v
```

## RL design 
A card game has a **variable-sized** action set, so all agents score `Q(state, move)`
and pick `argmax` over the legal moves. The state and each candidate move are
encoded as fixed-length vectors (`env/features.py`). The neural agents concatenate
them; Q-learning also adds bounded state–move interaction features so its action
preferences can depend on the current hand and table. Its updates are normalized
by feature magnitude to avoid instability with large hands.

DDQN, DQN and Q-learning training now default to **first-place wins**: an episode
ends as soon as any player empties their hand, with +1 for winning and −1 for
losing. Dense reward is the change in `Phi = -cards_remaining / initial_hand_size`,
with terminal potential zero and discount 1. This telescopes to a fixed offset,
so shedding cards after a loss cannot become a substitute for winning.
`--reward default` retains the previous per-card shaping and placement bonus:
`+0.1` per card the agent plays during the game, plus, at episode end,
`+1.0 / +0.3 / −0.3 / −1.0` for 1st/2nd/3rd/4th place (a linear `+1 → −1` scale is
used for other player counts).

Training and periodic evaluation default to fixed greedy opponents. Use
`--opponent random --eval-opponent random` for the random baseline, or
`--opponent mixed` to independently choose greedy/random opponents for each seat
on every training episode. Exploration now decreases by episode from 0.5 to 0.05
over 80% of the requested run (at least 1,000 episodes), so games with many passes
do not prematurely exhaust exploration. A seeded
environment produces a reproducible sequence of fresh deals; `reset(seed=...)`
restarts that sequence. Tournament win rates use each agent's actual number of
appearances, since not every agent appears in every deal.
Missing or incompatible checkpoints are skipped instead of entering an untrained
agent under a trained model's name.

## Train the bots

**Use this section for training.** Choose one command for the bot you want to
train; each starts a fresh model and saves to its own checkpoint path.

```bash
python3.12 -m training.train --agent ddqn --episodes 5000 --save-path checkpoints/ddqn_win_agent.pt
python3.12 -m training.train --agent dqn --episodes 5000 --save-path checkpoints/dqn_win_agent.pt
python3.12 -m training.train --agent qlearning --episodes 5000 --save-path checkpoints/qlearning_win_agent.npy
# Alternative: custom placement/trick rewards, without demonstrations by default
python3.12 -m training.train --agent shaped --episodes 5000 --save-path checkpoints/shaped_agent.pt
```

For a larger neural warm-up, add `--demo-games 1000 --demo-updates 3000` to
the DDQN or DQN command. This is the larger warm-up tested in the learning report;
it does not guarantee that subsequent reinforcement learning improves the model.
For pure reinforcement learning without demonstrations, add `--demo-games 0`.

### How training against greedy works

Greedy already searches for useful combinations. An initially random policy
rarely wins against three greedy opponents, and sparse winning examples make
training from scratch difficult. By default, DDQN, DQN, and Q-learning first
collect **200 greedy demonstration games** and fit their own action preferences
for 1,000 warm-up minibatches. A small demonstration rehearsal term remains
during reinforcement learning to reduce forgetting. Neural models use a legal-action
classification loss; the linear Q-learning model uses normalized margin updates.
The trained bots use only their learned model at play time, not a greedy fallback.
This is expert-assisted learning, not pure reinforcement learning from scratch.

`--demo-games`, `--demo-updates`, and `--demo-weight` control the warm-up and
rehearsal (default weight 0.1). Set `--demo-weight 0` to disable rehearsal after
warm-up. Shaped agents retain their custom placement reward and default to no
demonstrations; `--reward win` is rejected for these wrappers because they replace
environment rewards. Neural TD updates use Huber loss and gradient clipping.

Evaluation disables exploration and uses seeds separate from both training and
demonstrations. The progress log's training win rate includes exploration and
should not be compared directly with evaluation win rate. Old checkpoints still
load, but training commands start fresh; old weights do not acquire these changes
automatically. Q-learning remains a linear approximation with limited ability to
represent combination planning, so low win rates are not necessarily an update bug.

When periodic evaluation is enabled, a separate `*.best.pt` / `*.best.npy` file
keeps the strongest validation checkpoint, including the demonstration warm-up.
The ordinary checkpoint always contains the latest model. Use the best checkpoint
for play when later learning regresses; validation selection is still noisy and
should be checked on a separate test set.

See the [learning comparison](reports/ddqn-learning-20261002/REPORT.md) for measured
results and limitations. More iterations alone are not a guarantee of improvement.

Tournament lineups rotate through every seat on the **same shuffled deal**;
`--games` rounds up to complete rotations. `--rounds` covers all four-player
subsets of the loaded pool in each round. `--json` includes every game's deal
seed, seating order, and finish order as well as the summary. Random chooses
uniformly among legal actions, including passing when legal.

### Greedy strategy

The greedy bot compares the hand left after each legal play. It prefers fewer
remaining combinations and favors larger intact combinations over low singles
when the structural costs are equal. Spare aces, 2s, and jokers can contest tricks
throughout the game; their control value breaks ties rather than forbidding play.
It normally preserves pairs, straights, and bombs, but accepts up to one extra
estimated future play when an opponent pulls ahead or approaches the final third
of a starting hand. Under that pressure it can also spend a whole bomb to regain
the lead. An immediate finish always wins over conservation, and it avoids
offering a low single to a player with one card left.

For hands of at most 12 cards, a cached exact search finds the minimum number of
legal combinations needed to empty the hand. Larger hands use two greedy
partition estimates. This search sees only the bot's own cards; it does not
predict opponents' hidden cards or guarantee the best game strategy.

### Reproduce the retraining experiment

```bash
python3.12 -B scripts/benchmark_bots.py \
  --episodes 1000 --rounds 3 --baseline-deals 50 --workers 3 \
  --output reports/greedy-v2-20261002 \
  --checkpoints checkpoints/greedy-v2-20261002 \
  --legacy-source reports/greedy-v2-20261002/greedy_before.py
```

This trains Q-learning, DQN, DDQN, shaped DDQN, and shaped Q-learning from scratch
against mixed opponents. It saves separate checkpoints, reloads them with
exploration disabled, and compares them with Greedy, Random, and the saved
previous greedy policy. The evaluation uses held-out deals and also tests each
bot against three greedy opponents and three random opponents. Use
`--skip-training` to rerun evaluation from these checkpoints. Reusing the same
output/checkpoint directories replaces that experiment's files; choose new
directories to preserve another run.

Results and limitations: [experiment report](reports/greedy-v2-20261002/REPORT.md).
That experiment predates the excessive-passing correction; its scores describe
the earlier policy. The [passing regression report](reports/greedy-passing-fix-20261002/REPORT.md)
records the correction and its targeted before/after checks. Greedy is a fixed
policy: restart the play command to use the correction; no retraining is needed.

Existing neural checkpoints keep their architecture. Legacy additive Q-learning
checkpoints load with zero interaction weights, preserving their predictions.
Retrain to benefit from the learning and reward fixes; existing saved checkpoints
are not automatically updated.

### `DDQNAgent` (default)
The default training agent combines three improvements over the vanilla `DQNAgent`:
- **Double DQN (DDQN):** the greedy next action is chosen by the online network but
  its value is read from the target network, cutting Q-value overestimation.
- **Dueling Q-Network:** after the shared feature layers the network splits into a
  value stream `V(s)` and an advantage stream `A(s,a)`, combined as
  `Q(s,a) = V(s) + (A(s,a) − mean_a A(s,a))`.
- **Prioritized Experience Replay (PER):** a `SumTree` enables O(log N) sampling by
  TD-error priority (`alpha` controls prioritization; `beta` controls the
  importance-sampling correction and anneals to 1.0). Priorities are updated after
  each loss via `update_priorities`.

Hyperparameters (`--agent ddqn` plus the buffer/network knobs in
`agents/ddqn_agent.py`) include `alpha`, `beta`, `beta_anneal_steps`, and the usual
`lr`, `gamma`, `epsilon`, `batch_size`, `buffer_size`, `target_update`.

#### Training speed and timing

The training commands default to **CPU with one PyTorch thread**, suitable for
these small networks. Device selection is explicit: use `--device cpu`, `mps`,
`cuda`, or `auto`, and `--torch-threads N` to compare on your machine. Standalone
agent constructors retain automatic device selection unless `device` is supplied.

The commands in [Train the bots](#train-the-bots) already use CPU and one thread.
Add `--progress-every 100` to set the reporting interval explicitly.

Startup prints the actual device and thread count. Progress distinguishes the
time for the **last 100 episodes** from total training time; evaluation games are
timed separately. The earlier `elapsed` field was cumulative, not time per batch.

DDQN now encodes each state once per replay item, scores its legal moves together,
and reduces advantages across the batch without separate gradient operations for
every item. Replay stores state vectors separately from contiguous move matrices.
It still trains after every transition with the same batch size, target-update
cadence, and prioritized replay; existing network checkpoints remain compatible.

In the local 100-episode CPU benchmark, training took **25.6 s before versus 9.7 s
after**, with evaluation/checkpoint writes excluded. These measurements predate
demonstration warm-up/rehearsal, which adds training work. GPU performance was not
measured. See [profiling and validation](reports/ddqn-performance-20261002/REPORT.md)
for the configuration and numerical-equivalence checks. Restart a running trainer
to load the optimized code.

### `ShapedRewardBot` (`--agent shaped`)
A drop-in wrapper around the DDQN (or Q-learning) agent that swaps the reward signal
for a hand-shaped one while leaving the underlying learner untouched — so it trains
and plays through the same harness as the other bots. Its reward is the sum of three
components (all constants live in `bots/shaped_reward.ShapedRewardConfig`, exposed as
`--lambda1/--lambda2/--potential-scale/--trick-scale`):

- **Potential-based dense shaping** (every step): `γ·Φ(s') − Φ(s)`, where
  `Φ(s) = (−moves_to_empty(hand) − λ1·orphan_count(hand)·progress + λ2·deficit) / initial_hand_size`.
  The dense term uses the learner's discount and zero terminal potential.
  `moves_to_empty` is a cached greedy estimate using legal combinations, not an
  exact solver; `orphan_count`
  counts leftover singles; `deficit` compares the estimated remaining moves with
  the smallest active opponent hand, converted using `moves_per_card`.
- **Trick-resolution** (only when the bot wins the trick):
  `trick_scale · (value(trick) − cost(combo_played))`, awarded when all other active
  players have passed. Merely beating the current play earns no trick bonus.
  Bombs have a higher cost than ordinary combinations. This extra reward changes
  the training objective; only the potential-difference term is policy invariant.
- **Terminal rank reward:** the same `+1.0 / +0.3 / −0.3 / −1.0` placement bonus.

`observe()` accumulates the `dense / trick / terminal / total` components and prints
an **averaged** breakdown once every `log_every` episodes (aligned to the eval cadence,
so it reads like the other bots' periodic status lines rather than one line per episode).
Use `--agent shaped-ql` to wrap a Q-learning inner instead of DDQN.

## Extending
- **Bombs:** only four-of-a-kind for now; a joker-pair "rocket" or
  consecutive-pair bomb is an easy follow-up.
- **Self-play:** `training/train.py` trains seat 0 vs fixed opponents; swap in a
  learned opponent or enable all-seat learning for self-play.
- **Match mode:** `game/match.py` runs consecutive deals with the penalty
  transfers and can be used for full-match training/eval.

Earlier testing reported a 92% win rate against random opponents. This is a
historical result, not a verified benchmark for the corrected engine or for greedy
opponents; rerun evaluation after retraining.
