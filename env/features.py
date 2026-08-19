"""Fixed-length feature encoders for the RL environment.

The action set of a card game is variable, so both Q-learning and DQN score
``Q(state, move)``. We therefore encode the *state* and the *move* separately
and concatenate them into one input vector.
"""
import numpy as np

from game.cards import Card
from game.moves import Move, MoveType
from game.rules import Game

NUM_RANKS = 15          # rank values 3..17 -> indices 0..14
TYPE_DIM = 7           # PASS, SINGLE, PAIR, TRIPLE, FULL_HOUSE, STRAIGHT, BOMB


def _rank_vec(cards) -> np.ndarray:
    v = np.zeros(NUM_RANKS, dtype=np.float32)
    for c in cards:
        v[c.rank - 3] += 1.0
    return v


def state_vector(game: Game, seat: int) -> np.ndarray:
    """Encode the game state from ``seat``'s point of view."""
    n = game.num_players
    hand = game.hands[seat]
    parts = [_rank_vec(hand)]                                  # 15

    t = np.zeros(TYPE_DIM, dtype=np.float32)
    if game.table_move is None:
        t[MoveType.PASS] = 1.0                                 # "no table / leading"
    else:
        t[game.table_move.type] = 1.0
    parts.append(t)                                            # 7

    if game.table_move is not None:
        parts.append(np.array([(game.table_move.rank - 3) / 14.0], dtype=np.float32))
    else:
        parts.append(np.array([0.0], dtype=np.float32))
    if game.table_move is not None:
        parts.append(np.array([game.table_move.length / 15.0], dtype=np.float32))
    else:
        parts.append(np.array([0.0], dtype=np.float32))
    parts.append(np.array([1.0 if game.table_move is not None else 0.0], dtype=np.float32))
    parts.append(np.array([len(hand) / max(1, game.initial_hand_size)], dtype=np.float32))

    for j in range(n):
        if j == seat:
            continue
        parts.append(np.array([len(game.hands[j]) / max(1, game.initial_hand_size)],
                              dtype=np.float32))
    return np.concatenate(parts).astype(np.float32)


def move_vector(move: Move) -> np.ndarray:
    """Encode a candidate move (or PASS)."""
    if move.is_pass:
        parts = [np.zeros(NUM_RANKS, dtype=np.float32)]
        t = np.zeros(TYPE_DIM, dtype=np.float32)
        t[MoveType.PASS] = 1.0
        parts.append(t)
        parts.append(np.array([0.0], dtype=np.float32))   # rank
        parts.append(np.array([0.0], dtype=np.float32))   # length
        parts.append(np.array([0.0], dtype=np.float32))   # is_bomb
        parts.append(np.array([1.0], dtype=np.float32))   # is_pass
        return np.concatenate(parts).astype(np.float32)

    parts = [_rank_vec(move.cards)]
    t = np.zeros(TYPE_DIM, dtype=np.float32)
    t[move.type] = 1.0
    parts.append(t)
    parts.append(np.array([(move.rank - 3) / 14.0], dtype=np.float32))
    parts.append(np.array([move.length / 15.0], dtype=np.float32))
    parts.append(np.array([1.0 if move.is_bomb else 0.0], dtype=np.float32))
    parts.append(np.array([0.0], dtype=np.float32))
    return np.concatenate(parts).astype(np.float32)


def combined_vector(state_vec: np.ndarray, move_vec: np.ndarray) -> np.ndarray:
    return np.concatenate([state_vec, move_vec]).astype(np.float32)


def feature_dims(num_players: int, num_decks: int = 1):
    """Return (state_dim, move_dim) for the given configuration."""
    g = Game(num_players=num_players, num_decks=num_decks)
    s = state_vector(g, 0)
    card = g.hands[0][0]
    m = Move(MoveType.SINGLE, (card,), card.rank, 1)
    mv = move_vector(m)
    return len(s), len(mv)
