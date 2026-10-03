import numpy as np

from game.moves import Move, MoveType
from game.rules import Game

NUM_RANKS = 15
TYPE_DIM = len(MoveType)


def _rank_vec(cards) -> np.ndarray:
    v = np.zeros(NUM_RANKS, dtype=np.float32)
    for c in cards:
        v[c.rank - 3] += 1.0
    return v


def _scalar(x) -> np.ndarray:
    return np.array([x], dtype=np.float32)


def state_vector(game: Game, seat: int) -> np.ndarray:
    n = game.num_players
    hand = game.hands[seat]
    tm = game.table_move
    parts = [_rank_vec(hand)]

    t = np.zeros(TYPE_DIM, dtype=np.float32)
    t[MoveType.PASS if tm is None else tm.type] = 1.0
    parts.append(t)

    parts.append(_scalar((tm.rank - 3) / 14.0 if tm is not None else 0.0))
    parts.append(_scalar(tm.length / 15.0 if tm is not None else 0.0))
    parts.append(_scalar(1.0 if tm is not None else 0.0))
    parts.append(_scalar(len(hand) / max(1, game.initial_hand_size)))

    for j in range(n):
        if j == seat:
            continue
        parts.append(_scalar(len(game.hands[j]) / max(1, game.initial_hand_size)))
    return np.concatenate(parts).astype(np.float32)


def move_vector(move: Move) -> np.ndarray:
    if move.is_pass:
        parts = [np.zeros(NUM_RANKS, dtype=np.float32)]
        t = np.zeros(TYPE_DIM, dtype=np.float32)
        t[MoveType.PASS] = 1.0
        parts.append(t)
        parts.append(_scalar(0.0))
        parts.append(_scalar(0.0))
        parts.append(_scalar(0.0))
        parts.append(_scalar(1.0))
        return np.concatenate(parts).astype(np.float32)

    parts = [_rank_vec(move.cards)]
    t = np.zeros(TYPE_DIM, dtype=np.float32)
    t[move.type] = 1.0
    parts.append(t)
    parts.append(_scalar((move.rank - 3) / 14.0))
    parts.append(_scalar(move.length / 15.0))
    parts.append(_scalar(1.0 if move.is_bomb else 0.0))
    parts.append(_scalar(0.0))
    return np.concatenate(parts).astype(np.float32)


def combined_vector(state_vec: np.ndarray, move_vec: np.ndarray) -> np.ndarray:
    return np.concatenate([state_vec, move_vec]).astype(np.float32)


def feature_dims(num_players: int, num_decks: int = 1):
    # Dimensions depend on the seat count, not on a randomly dealt sample game.
    return NUM_RANKS + TYPE_DIM + 4 + num_players - 1, NUM_RANKS + TYPE_DIM + 4
