"""Shaped reward function for Zheng Shang You (potential-based + trick + terminal).

Three additive components, computed independently and summed per step / episode:

1. Potential-based dense shaping (every step):
       r_dense(t) = scale * (gamma * Phi(s_{t+1}) - Phi(s_t))
   Phi(s) = -(moves_to_empty/ih) - lambda1*(orphan_count/ih)*progress
            + lambda2*(deficit/ih)
   where progress grows as your hand shrinks (shed-early pressure), orphan_count
   pushes isolated singles out early, and deficit = min(opp hand) - your hand makes
   an opponent shedding cards implicitly cost you reward. This term is PURELY
   potential-based (always Phi(s') - Phi(s), never a standalone bonus), so it does
   not distort the optimal policy -- it only changes credit assignment.

2. Trick-resolution reward (only when this agent wins a trick):
       r_trick = scale_trick * (value(trick) - cost(combo))
   cost is a scarcity table (bombs expensive); value rewards winning
   high-card-count tricks late and denying opponents about to go out.

3. Terminal rank reward (episode end), zero-sum normalized across players:
       1st +1.0, 2nd +0.3, 3rd -0.3, 4th -1.0 (linear scale for other counts).

Why: naive bots burn a 4-of-a-kind bomb to beat a lone 3 at the start of a 27-card
hand, and hoard orphan "trash" singles. The potential terms penalize holding orphans
and reward shedding; the trick term makes bombs costly unless they win something
valuable (denying a near-empty opponent), so bombs are hoarded for moments that
matter.

All constants live in ShapedRewardConfig so they can be swept without code edits.
The deficit term uses +lambda2*deficit (sign chosen so an opponent's progress lowers
your potential, matching the stated goal); flip it to -lambda2*deficit in phi() if
you want the literal "-lambda2*opponent_relative_deficit" form.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from game.moves import MoveType
from env import features
from bots.greedy_bot import M, O


@dataclass
class ShapedRewardConfig:
    gamma: float = 0.95
    lambda1: float = 0.2
    lambda2: float = 0.2
    potential_scale: float = 0.5
    trick_scale: float = 0.05

    # Approx moves-per-card used to map opponents (encoded only as hand SIZES in the
    # state) onto the agent's move-space, so the deficit/lead term is move-consistent
    # rather than card-consistent. ~0.45 means a 27-card 2-deck hand takes ~12 moves.
    moves_per_card: float = 0.45

    progress_base: float = 1.0
    progress_scale: float = 1.0

    cost_single_low: float = 0.0
    cost_single_high: float = 0.05
    cost_pair: float = 0.1
    cost_triple: float = 0.15
    cost_full_house: float = 0.3
    cost_straight_base: float = 0.15
    cost_straight_per_len: float = 0.05
    cost_bomb: float = 1.0
    cost_airplane: float = 0.3
    cost_consec_pairs_base: float = 0.2
    cost_consec_pairs_per_len: float = 0.1
    cost_airplane_body: float = 0.3
    cost_airplane_body_per_len: float = 0.1

    opponent_lead_threshold: int = 5
    opponent_lead_bonus: float = 0.4

    terminal: tuple = (1.0, 0.3, -0.3, -1.0)
    num_players: int = 4
    num_decks: int = 2

    @property
    def initial_hand_size(self) -> int:
        return (self.num_decks * 54) // self.num_players


def moves_to_empty(hand_hist) -> int:
    """Use the same legal-combination estimate as the fixed opponent."""
    return M(hand_hist)


def orphan_count(hand_hist) -> int:
    return sum(group.type == "SINGLE" for group in O(hand_hist))


def _parse(state, cfg):
    n = features.NUM_RANKS
    td = features.TYPE_DIM
    hand = np.asarray(state[:n], dtype=np.float64)
    own = float(hand.sum())
    present = float(state[n + td + 2])
    opp_start = n + td + 4
    own_fraction = float(state[n + td + 3])
    scale = own / own_fraction if own_fraction > 0 else cfg.initial_hand_size
    opp = [float(state[opp_start + i]) * scale
           for i in range(cfg.num_players - 1)]
    return hand, own, present, [size for size in opp if size > 0]


def phi(state, cfg) -> float:
    if state is None:
        return 0.0
    hand, own, present, opp = _parse(state, cfg)
    ih = cfg.initial_hand_size
    mte = moves_to_empty(hand)
    progress = cfg.progress_base + cfg.progress_scale * (1.0 - own / ih)
    # Deficit / "lead" term, measured in MOVES-TO-EMPTY (not raw cards) so that
    # shedding a 4-card bomb is worth the same as shedding a single: both reduce the
    # agent's remaining moves by exactly one. The state only stores opponents as hand
    # SIZES, so we map their sizes onto move-space with the fixed moves_per_card ratio,
    # while the agent's own side uses the greedy estimate. This avoids "any card
    # removed = good" per-card inflation that used to over-reward bombs.
    opp_moves = [o * cfg.moves_per_card for o in opp]
    deficit = (min(opp_moves) if opp_moves else mte) - mte
    return (-(mte / ih)
            - cfg.lambda1 * (orphan_count(hand) / ih) * progress
            + cfg.lambda2 * (deficit / ih))


def combo_cost(move, cfg) -> float:
    if move.is_bomb:
        return cfg.cost_bomb
    t = move.type
    if t == MoveType.SINGLE:
        return cfg.cost_single_low if move.rank < 10 else cfg.cost_single_high
    if t == MoveType.PAIR:
        return cfg.cost_pair
    if t == MoveType.TRIPLE:
        return cfg.cost_triple
    if t == MoveType.FULL_HOUSE:
        return cfg.cost_full_house
    if t == MoveType.STRAIGHT:
        return cfg.cost_straight_base + cfg.cost_straight_per_len * (move.length - 5)
    if t == MoveType.CONSEC_PAIRS:
        return cfg.cost_consec_pairs_base + cfg.cost_consec_pairs_per_len * (move.length - 3)
    if t == MoveType.CONSEC_TRIPLES:
        return cfg.cost_airplane_body + cfg.cost_airplane_body_per_len * (move.length - 2)
    if t == MoveType.AIRPLANE:
        return cfg.cost_airplane
    return 0.0


def trick_value(state, move, cfg) -> float:
    hand, own, present, opp = _parse(state, cfg)
    ih = cfg.initial_hand_size
    progress = cfg.progress_base + cfg.progress_scale * (1.0 - own / ih)
    v = len(move.cards) * progress
    if any(o <= cfg.opponent_lead_threshold for o in opp):
        v += cfg.opponent_lead_bonus
    return v


def trick_reward(state, action, cfg) -> float:
    if action is None or action.is_pass:
        return 0.0
    return cfg.trick_scale * (trick_value(state, action, cfg) - combo_cost(action, cfg))


def terminal_reward(finish_order, seat, cfg) -> float:
    try:
        rank = finish_order.index(seat) + 1
    except ValueError:
        rank = cfg.num_players

    if cfg.num_players == 4:
        return cfg.terminal[rank - 1]
    return 1.0 - 2.0 * (rank - 1) / (cfg.num_players - 1)


class ShapedReward:
    def __init__(self, cfg: Optional[ShapedRewardConfig] = None):
        self.cfg = cfg or ShapedRewardConfig()

    def components(self, state, action, next_state, done, finish_order, seat,
                   won_trick=False) -> dict:
        cfg = self.cfg
        r_trick = trick_reward(state, action, cfg) if won_trick else 0.0
        r_dense = cfg.potential_scale * (
            cfg.gamma * phi(None if done else next_state, cfg) - phi(state, cfg))
        r_term = terminal_reward(finish_order, seat, cfg) if done else 0.0
        return {"dense": r_dense, "trick": r_trick, "terminal": r_term,
                "total": r_dense + r_trick + r_term}
