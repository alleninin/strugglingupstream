import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from env import features
from env.env import ZhengShangYouEnv
from agents.random_agent import RandomAgent
from agents.qlearning import QLearningAgent


def make_opponent_policies(num_players, seed):
    return [RandomAgent(seed=seed * 100 + s).act for s in range(1, num_players)]


def evaluate_vs_random(agent, num_players, num_decks, seed, n=100):
    old_eps = getattr(agent, "epsilon", 0.0)
    agent.epsilon = 0.0
    wins = 0
    try:
        for i in range(n):
            opp = [RandomAgent(seed=seed * 7 + i * 13 + j).act
                   for j in range(1, num_players)]
            env = ZhengShangYouEnv(num_players=num_players, num_decks=num_decks,
                                   opponent_policies=opp, seed=seed * 3 + i)
            state = env.reset()
            done = False
            while not done:
                legal = env.get_legal_moves()
                action = agent.act(state, legal)
                next_state, reward, done, info = env.step(action)
                state = next_state
            if reward == 1.0:
                wins += 1
    finally:
        agent.epsilon = old_eps
    return wins / n


def train(agent_kind, episodes, num_players, num_decks, seed, save_path, eval_every):
    s_dim, a_dim = features.feature_dims(num_players, num_decks)
    if agent_kind == "qlearning":
        agent = QLearningAgent(s_dim, a_dim, alpha=0.05, gamma=0.95, epsilon=0.2,
                               epsilon_decay=0.9995, min_epsilon=0.02, seed=seed)
    else:
        from agents.dqn_agent import DQNAgent
        agent = DQNAgent(s_dim, a_dim, lr=1e-3, gamma=0.95, epsilon=0.5,
                         epsilon_decay=0.995, min_epsilon=0.05, seed=seed)

    opp = make_opponent_policies(num_players, seed)
    env = ZhengShangYouEnv(num_players=num_players, num_decks=num_decks,
                           opponent_policies=opp, seed=seed)

    for ep in range(episodes):
        state = env.reset(seed=(seed * 1000 + ep) if seed is not None else None)
        agent.reset_episode()
        done = False
        while not done:
            legal = env.get_legal_moves()
            action = agent.act(state, legal)
            next_state, reward, done, info = env.step(action)
            agent.observe((state, action, reward, next_state, done,
                           info["next_legal_moves"]))
            state = next_state

        if eval_every and (ep + 1) % eval_every == 0:
            wr = evaluate_vs_random(agent, num_players, num_decks, seed, n=100)
            print(f"[ep {ep + 1:>{len(str(episodes))}}] win_rate_vs_random = {wr:.3f}")

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    agent.save(save_path)
    print(f"Saved {agent_kind} agent -> {save_path}")
    return agent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", choices=["qlearning", "dqn"], default="dqn")
    ap.add_argument("--episodes", type=int, default=5000)
    ap.add_argument("--num-players", type=int, default=3)
    ap.add_argument("--num-decks", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-path", default="checkpoints/dqn_agent.pt")
    ap.add_argument("--eval-every", type=int, default=500)
    args = ap.parse_args()

    train(args.agent, args.episodes, args.num_players, args.num_decks,
          args.seed, args.save_path, args.eval_every)


if __name__ == "__main__":
    main()
