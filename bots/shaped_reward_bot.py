"""Shaped-reward bot: wraps a DDQN (or Q-learning) agent and recomputes the reward
in observe() using bots.shaped_reward, leaving the underlying agent's learning code
untouched. Drop-in compatible with the existing training loop (BaseAgent interface).
"""

from agents.base import BaseAgent
from agents.ddqn_agent import DDQNAgent
from agents.qlearning import QLearningAgent

from .shaped_reward import ShapedReward, ShapedRewardConfig


class ShapedRewardBot(BaseAgent):
    def __init__(self, state_dim, action_dim, cfg=None, inner=None,
                 agent_type="ddqn", log_every=500, **inner_kwargs):
        self.cfg = cfg or ShapedRewardConfig()
        self.reward = ShapedReward(self.cfg)
        inner_kwargs.setdefault("gamma", self.cfg.gamma)
        if inner is not None:
            self.inner = inner
        elif agent_type == "qlearning":
            self.inner = QLearningAgent(state_dim, action_dim, **inner_kwargs)
        else:
            self.inner = DDQNAgent(state_dim, action_dim, **inner_kwargs)
        self._is_ddqn = isinstance(self.inner, DDQNAgent)
        if self.inner.gamma != self.cfg.gamma:
            raise ValueError("shaping and learner gamma must match")
        self.env = None
        self.ep_dense = 0.0
        self.ep_trick = 0.0
        self.ep_terminal = 0.0
        self.last_breakdown = None
        self.log_every = log_every
        self._acc_dense = 0.0
        self._acc_trick = 0.0
        self._acc_terminal = 0.0
        self._acc_count = 0

    @property
    def epsilon(self):
        return self.inner.epsilon

    @epsilon.setter
    def epsilon(self, v):
        self.inner.epsilon = v

    def set_env(self, env):
        self.env = env

    def act(self, obs, legal_moves):
        return self.inner.act(obs, legal_moves)

    def observe(self, transition) -> None:
        if len(transition) == 7:
            state, action, reward, next_state, done, next_legal, cur_legal = transition
        else:
            state, action, reward, next_state, done, next_legal = transition
            cur_legal = None
        seat = self.env.agent_seat if self.env is not None else 0
        finish_order = (self.env.game.finish_order
                        if (done and self.env is not None) else [])
        comp = self.reward.components(state, action, next_state, done,
                                      finish_order, seat,
                                      won_trick=(self.env is not None and
                                                 seat in self.env.last_trick_winners))
        self.ep_dense += comp["dense"]
        self.ep_trick += comp["trick"]
        self.ep_terminal += comp["terminal"]
        self.last_breakdown = comp
        total = comp["total"]
        if self._is_ddqn:
            self.inner.observe((state, action, total, next_state, done,
                                next_legal, cur_legal))
        else:
            self.inner.observe((state, action, total, next_state, done, next_legal))
        if done:
            self._acc_dense += self.ep_dense
            self._acc_trick += self.ep_trick
            self._acc_terminal += self.ep_terminal
            self._acc_count += 1
            if self.log_every and self._acc_count >= self.log_every:
                self._log_window(self._acc_count)
                self._acc_dense = self._acc_trick = self._acc_terminal = 0.0
                self._acc_count = 0

    def _log_window(self, n) -> None:
        d = self._acc_dense / n
        t = self._acc_trick / n
        te = self._acc_terminal / n
        play_total = d + t + te
        print(f"[shaped] avg reward / {n} eps -> dense={d:+.3f} "
              f"trick={t:+.3f} terminal={te:+.3f} total={play_total:+.3f}")

    def reset_episode(self) -> None:
        self.inner.reset_episode()
        self.ep_dense = self.ep_trick = self.ep_terminal = 0.0

    def save(self, path) -> None:
        self.inner.save(path)

    def load(self, path) -> None:
        self.inner.load(path)
