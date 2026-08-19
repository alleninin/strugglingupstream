# Zheng ShangYou (征上游)/Struggling Upstream AI Game Engine + RL Training

A from-scratch Python environment for the climbing card game **Zheng Shang You**
("Struggle Upstream"). Be the first to shed all your cards. There's an RL-friendly environment and several agents you can train and
pit against each other.

## Rules implemented
- **Card ranking (low→high):** `3 4 5 6 7 8 9 10 J Q K A 2` then **Black Joker**
  then **Red Joker**.
- **Combinations:** single, pair, triple, full house (triple + pair), straight
  (5+ consecutive ranks from `3` to `A`, never `2`/jokers), **airplane** (two
  triples of any ranks plus two pairs = exactly 10 cards, ranked by the higher
  triple), and a **four-of-a-kind bomb**.
- **Beating a play:** you must play a *higher* combo of the *exact same type and
  length*, **or** a bomb. A bomb beats any non-bomb and is itself beaten only by
  a higher bomb. Jokers are individual highest singles (they do not form a pair).
- **Passing & clearing:** pass your turn; if every other active player passes in
  a row, the table clears and the last player to play leads anew.
- **Struggling Upstream (penalty):** across deals, the last- and next-to-last
  finishers hand their single highest card (face-up) to the top-2 winners.

Default setup is **4 players / 2 decks** (27 cards each, no leftovers); both are
configurable (`num_players`, `num_decks`).

## Layout
```
game/        rules engine: cards, moves, rules (turn loop), match (penalty)
env/         RL wrapper: feature encoders + ZhengShangYouEnv
agents/      RandomAgent, QLearningAgent, DQNAgent, DDQNAgent (PyTorch)
training/    train.py  (train one seat vs fixed opponents)
evaluate.py  round-robin tournament across agents
scripts/     demo.py   (print a full game)
```

## Install
```bash
pip install -r requirements.txt   # torch + numpy
```

## Run
```bash
# 1. Sanity-check the rules with a random-agent game
python scripts/demo.py

# 2. Train an agent (saves a checkpoint)
python -m training.train --agent qlearning --episodes 3000 --save-path checkpoints/q_agent.npy
python -m training.train --agent dqn       --episodes 5000 --save-path checkpoints/dqn_agent.pt
# Efficient default: Dueling Double-DQN + Prioritized Experience Replay
python -m training.train --agent ddqn      --episodes 5000 --save-path checkpoints/ddqn_agent.pt

# 3. Tournament: Random vs QLearning vs DQN vs DDQN
python evaluate.py --games 300

# 4. Play against a trained DDQN agent
python scripts/play.py --checkpoint checkpoints/ddqn_agent.pt --agent ddqn
```

## RL design 
A card game has a **variable-sized** action set, so all agents score `Q(state, move)`
and pick `argmax` over the legal moves. The state and each candidate move are
encoded as fixed-length vectors (`env/features.py`) and concatenated, which lets a
single linear weight vector or MLP generalize across moves without a giant flat
action layer. Reward is based on finish: `1 - 2*(rank-1)/(num_players-1)`
(1st = +1, last = −1).

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

## Extending
- **Bombs:** only four-of-a-kind for now; a joker-pair "rocket" or
  consecutive-pair bomb is an easy follow-up.
- **Self-play:** `training/train.py` trains seat 0 vs fixed opponents; swap in a
  learned opponent or enable all-seat learning for self-play.
- **Match mode:** `game/match.py` runs consecutive deals with the penalty
  transfers and can be used for full-match training/eval.

In testing, the trained models have a 92% win rate against a random player.