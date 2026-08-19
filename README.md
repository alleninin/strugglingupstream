# Zheng Shang You (征上游) — Game Engine + RL Training

A from-scratch Python environment for the climbing card game **Zheng Shang You**
("Struggle Upstream"). Be the first to shed all your cards. Ships with a clean
rules engine, an RL-friendly environment, and several agents you can train and
pit against each other.

## Rules implemented
- **Card ranking (low→high):** `3 4 5 6 7 8 9 10 J Q K A 2` then **Black Joker**
  then **Red Joker**.
- **Combinations:** single, pair, triple, full house (triple + pair), straight
  (5+ consecutive ranks from `3` to `A`, never `2`/jokers), and a **four-of-a-kind
  bomb**.
- **Beating a play:** you must play a *higher* combo of the *exact same type and
  length*, **or** a bomb. A bomb beats any non-bomb and is itself beaten only by
  a higher bomb. Jokers are individual highest singles (they do not form a pair).
- **Passing & clearing:** pass your turn; if every other active player passes in
  a row, the table clears and the last player to play leads anew.
- **Struggling Upstream (penalty):** across deals, the last- and next-to-last
  finishers hand their single highest card (face-up) to the top-2 winners.

Default setup is **3 players / 1 deck** (18 cards each, no leftovers); both are
configurable (`num_players`, `num_decks`).

## Layout
```
game/        rules engine: cards, moves, rules (turn loop), match (penalty)
env/         RL wrapper: feature encoders + ZhengShangYouEnv
agents/      RandomAgent, QLearningAgent, DQNAgent (PyTorch)
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

# 3. Tournament: Random vs QLearning vs DQN
python evaluate.py --games 300
```

## RL design note
A card game has a **variable-sized** action set, so both the Q-learning and DQN
agents score `Q(state, move)` and pick `argmax` over the legal moves. The state
and each candidate move are encoded as fixed-length vectors (`env/features.py`)
and concatenated, which lets a single linear weight vector or MLP generalize
across moves without a giant flat action layer. Reward is terminal only:
`1 - 2*(rank-1)/(num_players-1)` (1st = +1, last = −1).

## Extending
- **Bombs:** only four-of-a-kind for now; a joker-pair "rocket" or
  consecutive-pair bomb is an easy follow-up.
- **Self-play:** `training/train.py` trains seat 0 vs fixed opponents; swap in a
  learned opponent or enable all-seat learning for self-play.
- **Match mode:** `game/match.py` runs consecutive deals with the penalty
  transfers and can be used for full-match training/eval.
