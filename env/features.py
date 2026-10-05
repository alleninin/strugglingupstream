import numpy as np

from game.moves import Move, MoveType
from game.rules import HISTORY_LENGTH, Game

NUM_RANKS = 15
TYPE_DIM = len(MoveType)


def _rank_vec(cards) -> np.ndarray:
    v = np.zeros(NUM_RANKS, dtype=np.float32)
    for c in cards:
        v[c.rank - 3] += 1.0
    return v


def state_vector(game: Game, seat: int, history=False) -> np.ndarray:
    """Own cards and public state, with opponents ordered clockwise from this seat."""
    vector = np.zeros(feature_dims(game.num_players)[0], dtype=np.float32)
    hand, table = game.hands[seat], game.table_move
    vector[:NUM_RANKS] = _rank_vec(hand)
    vector[NUM_RANKS + (MoveType.PASS if table is None else table.type)] = 1
    offset = NUM_RANKS + TYPE_DIM
    if table is not None:
        vector[offset : offset + 3] = ((table.rank - 3) / 14, table.length / 15, 1)
    scale = max(1, game.initial_hand_size)
    vector[offset + 3] = len(hand) / scale
    for relative_seat in range(1, game.num_players):
        vector[offset + 3 + relative_seat] = (
            len(game.hands[(seat + relative_seat) % game.num_players]) / scale
        )
    offset += 3 + game.num_players
    deck_scale = max(1, 4 * game.num_decks)
    vector[offset : offset + NUM_RANKS] = _rank_vec(game.played_cards) / deck_scale
    offset += NUM_RANKS
    if table is not None:
        vector[offset : offset + NUM_RANKS] = _rank_vec(table.cards) / deck_scale
        vector[offset + NUM_RANKS + (game.table_owner - seat) % game.num_players] = 1
    vector[-1] = game.passes_in_a_row / max(1, game.num_players - 1)
    return np.concatenate((vector, history_vector(game, seat))) if history else vector


def move_vector(move: Move) -> np.ndarray:
    vector = np.zeros(NUM_RANKS + TYPE_DIM + 4, dtype=np.float32)
    if move.is_pass:
        vector[NUM_RANKS + MoveType.PASS] = vector[-1] = 1
    else:
        vector[:NUM_RANKS] = _rank_vec(move.cards)
        vector[NUM_RANKS + move.type] = 1
        vector[-4:-1] = ((move.rank - 3) / 14, move.length / 15, move.is_bomb)
    return vector


def combined_vector(state_vec: np.ndarray, move_vec: np.ndarray) -> np.ndarray:
    return np.concatenate([state_vec, move_vec]).astype(np.float32)


def feature_dims(num_players: int, num_decks: int = 1):

    old_state = NUM_RANKS + TYPE_DIM + 4 + num_players - 1
    return old_state + 2 * NUM_RANKS + num_players + 1, NUM_RANKS + TYPE_DIM + 4


def legacy_state_dim(state_dim):
    """Recognize an expanded game observation; synthetic test dimensions stay as-is."""
    constant = 3 * NUM_RANKS + TYPE_DIM + 4
    players, remainder = divmod(state_dim - constant, 2)
    if remainder or players < 2:
        return state_dim
    return NUM_RANKS + TYPE_DIM + 3 + players


def history_dim(base_state_dim):
    players, remainder = divmod(base_state_dim - (3 * NUM_RANKS + TYPE_DIM + 4), 2)
    if remainder or players < 2:
        raise ValueError("history requires full public-state observations")
    return HISTORY_LENGTH * (players + TYPE_DIM + 3)


def history_vector(game, seat):
    """Four public actions, oldest first, with relative actors and explicit padding."""
    history = tuple(game.action_history)
    vector = np.zeros(
        (HISTORY_LENGTH, game.num_players + TYPE_DIM + 3), dtype=np.float32
    )
    for row, (actor, move) in zip(vector[-len(history) :], history):
        row[(actor - seat) % game.num_players] = 1
        row[game.num_players + move.type] = 1
        if not move.is_pass:
            row[-3:-1] = ((move.rank - 3) / 14, move.length / 15)
        row[-1] = 1
    return vector.ravel()


def state_for(game, seat, agent):
    owner = getattr(agent, "__self__", agent)
    owner = getattr(owner, "inner", owner)
    return state_vector(game, seat, history=getattr(owner, "history_features", False))
