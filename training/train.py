import argparse
import os
import sys
import random
import time
import copy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from env import features
from env.env import ZhengShangYouEnv
from agents.qlearning import QLearningAgent


def make_opponent_policies(num_players, seed, num_decks=2, opponent="greedy"):
    from bots.greedy_bot import GreedyBot
    from agents.random_agent import RandomAgent
    if opponent not in ("greedy", "random", "mixed"):
        raise ValueError(f"unknown opponent: {opponent}")
    policies = []
    rng = random.Random(seed)
    for seat in range(1, num_players):
        opponent_seed = None if seed is None else seed * 100 + seat
        kind = rng.choice(("greedy", "random")) if opponent == "mixed" else opponent
        bot = (RandomAgent(seed=opponent_seed) if kind == "random" else
               GreedyBot(num_players=num_players, num_decks=num_decks, seed=opponent_seed))
        policies.append(bot.act)
    return policies


def evaluate_vs_opponent(agent, num_players, num_decks, seed, n=100, opponent="greedy"):
    old_eps = getattr(agent, "epsilon", 0.0)
    agent.epsilon = 0.0
    wins = 0
    try:
        for i in range(n):
            # Separate evaluation deals from training's seed * 1000 + episode.
            # Previously seed=0 evaluated on the first training deals again.
            game_seed = None if seed is None else (1 << 62) + seed * 1000003 + i
            opp = make_opponent_policies(num_players, game_seed, num_decks, opponent)
            env = ZhengShangYouEnv(num_players=num_players, num_decks=num_decks,
                                   opponent_policies=opp, seed=game_seed)
            state = env.reset()
            done = env.game.done
            while not done:
                legal = env.get_legal_moves()
                action = agent.act(state, legal)
                next_state, reward, done, info = env.step(action)
                state = next_state
            if env.game.finish_order.index(env.agent_seat) == 0:
                wins += 1
    finally:
        agent.epsilon = old_eps
    return wins / n


def train(agent_kind, episodes, num_players, num_decks, seed, save_path, eval_every,
          lambda1=0.2, lambda2=0.2, potential_scale=0.5, trick_scale=0.05,
          opponent="greedy", eval_opponent="greedy", reward_scheme=None,
          progress_every=0, eval_games=100, device="cpu", torch_threads=1,
          demo_games=None, demo_updates=1000, demo_weight=0.1):
    if demo_games is None:
        demo_games = 0 if agent_kind in ("shaped", "shaped-ql") else 200
    if episodes < 0 or demo_games < 0 or demo_updates < 0 or not 0 <= demo_weight <= 1:
        raise ValueError("episode/demo counts must be nonnegative and demo_weight between 0 and 1")
    if eval_every and eval_games < 1:
        raise ValueError("eval_games must be positive when evaluation is enabled")
    s_dim, a_dim = features.feature_dims(num_players, num_decks)
    if reward_scheme is None:
        reward_scheme = "default" if agent_kind in ("shaped", "shaped-ql") else "win"
    if reward_scheme == "win" and agent_kind in ("shaped", "shaped-ql"):
        raise ValueError("shaped agents replace rewards; use ddqn with --reward win")
    win_objective = reward_scheme == "win"
    if agent_kind in ("dqn", "ddqn", "shaped"):
        import torch
        if torch_threads < 1:
            raise ValueError("torch_threads must be positive")
        torch.set_num_threads(torch_threads)
    if agent_kind == "qlearning":
        agent = QLearningAgent(s_dim, a_dim, alpha=0.05, gamma=0.95, epsilon=0.3,
                               epsilon_decay=0.9998, min_epsilon=0.05, seed=seed)
    elif agent_kind == "dqn":
        from agents.dqn_agent import DQNAgent
        agent = DQNAgent(s_dim, a_dim, lr=3e-4, gamma=0.95, epsilon=0.5,
                         epsilon_decay=0.9998, min_epsilon=0.05, seed=seed, device=device)
    elif agent_kind in ("shaped", "shaped-ql"):
        from bots.shaped_reward import ShapedRewardConfig
        from bots.shaped_reward_bot import ShapedRewardBot
        inner_type = "qlearning" if agent_kind == "shaped-ql" else "ddqn"
        cfg = ShapedRewardConfig(num_players=num_players, num_decks=num_decks,
                                 lambda1=lambda1, lambda2=lambda2,
                                 potential_scale=potential_scale,
                                 trick_scale=trick_scale)
        agent = ShapedRewardBot(s_dim, a_dim, cfg=cfg, agent_type=inner_type,
                                log_every=eval_every, seed=seed,
                                epsilon=0.3 if inner_type == "qlearning" else 0.5,
                                epsilon_decay=0.9998, min_epsilon=0.05,
                                **({"device": device} if inner_type == "ddqn" else {}))
    elif agent_kind == "ddqn":
        from agents.ddqn_agent import DDQNAgent
        agent = DDQNAgent(s_dim, a_dim, lr=3e-4, gamma=1.0 if win_objective else 0.95,
                          epsilon=0.5, epsilon_decay=1.0, min_epsilon=0.05,
                          seed=seed, device=device)
    else:
        raise ValueError(f"unknown agent: {agent_kind}")

    opp = make_opponent_policies(num_players, seed, num_decks, opponent)
    env = ZhengShangYouEnv(num_players=num_players, num_decks=num_decks,
                           opponent_policies=opp, seed=seed,
                           reward_scheme=reward_scheme)
    if hasattr(agent, "set_env"):
        agent.set_env(env)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    inner = getattr(agent, "inner", agent)
    inner.epsilon_decay = 1.0
    if win_objective:
        inner.gamma = 1.0  # Must match the undiscounted potential in the environment.
    if hasattr(inner, "device"):
        print(f"[training] agent={agent_kind} reward={reward_scheme} device={inner.device} "
              f"torch_threads={torch.get_num_threads()}", flush=True)

    stem, extension = os.path.splitext(save_path)
    best_path = stem + ".best" + extension
    best_win_rate = -1.0

    if demo_games:
        from training.demonstrations import collect_demonstrations, Demonstrations
        warmup_started = time.perf_counter()
        examples = collect_demonstrations(demo_games, num_players, num_decks, seed)
        demonstrations = Demonstrations(examples, seed=seed, weight=demo_weight)
        demonstrations.pretrain(inner, demo_updates)
        inner.demonstrations = demonstrations
        print(f"[warmup] {demo_games} greedy games, {len(examples)} decisions, "
              f"{time.perf_counter() - warmup_started:.1f}s", flush=True)
        if eval_every:
            # The extra pre-training evaluation must not change exploration draws.
            rng_state = copy.deepcopy(inner.rng.bit_generator.state)
            try:
                best_win_rate = evaluate_vs_opponent(agent, num_players, num_decks, seed,
                                                     n=eval_games, opponent=eval_opponent)
            finally:
                inner.rng.bit_generator.state = rng_state
            agent.save(best_path)
            print(f"[warmup] win_rate_vs_{eval_opponent}={best_win_rate:.3f}; "
                  f"best checkpoint -> {best_path}", flush=True)

    started = time.perf_counter()
    progress_started = started
    progress_episode = 0
    evaluation_seconds = 0.0
    progress_evaluation_seconds = 0.0
    progress_wins = 0
    for ep in range(episodes):
        agent.epsilon = exploration_epsilon(ep, episodes)
        episode_seed = (seed * 1000 + ep) if seed is not None else None
        if opponent == "mixed":
            env.opponent_policies = make_opponent_policies(num_players, episode_seed, num_decks, opponent)
        state = env.reset(seed=episode_seed)
        agent.reset_episode()
        done = env.done
        while not done:
            legal = env.get_legal_moves()
            action = agent.act(state, legal)
            next_state, reward, done, info = env.step(action)
            if agent_kind in ("ddqn", "shaped"):
                agent.observe((state, action, reward, next_state, done,
                               info["next_legal_moves"], legal))
            else:
                agent.observe((state, action, reward, next_state, done,
                               info["next_legal_moves"]))
            state = next_state

        progress_wins += int(bool(env.game.finish_order) and env.game.finish_order[0] == env.agent_seat)

        if eval_every and (ep + 1) % eval_every == 0:
            eval_started = time.perf_counter()
            wr = evaluate_vs_opponent(agent, num_players, num_decks, seed, n=eval_games,
                                      opponent=eval_opponent)
            eval_seconds = time.perf_counter() - eval_started
            evaluation_seconds += eval_seconds
            print(f"[ep {ep + 1:>{len(str(episodes))}}] win_rate_vs_{eval_opponent} = {wr:.3f} "
                  f"({eval_games} evaluation games in {eval_seconds:.1f}s)", flush=True)
            if wr > best_win_rate:
                best_win_rate = wr
                agent.save(best_path)
            # periodic checkpoint so a kill/timeout doesn't discard all progress
            agent.save(save_path)
        if progress_every and (ep + 1) % progress_every == 0:
            now = time.perf_counter()
            interval_episodes = ep + 1 - progress_episode
            interval_seconds = now - progress_started - (evaluation_seconds - progress_evaluation_seconds)
            print(f"[{agent_kind} {ep + 1}/{episodes}] epsilon={agent.epsilon:.3f} "
                  f"training_win_rate={progress_wins / interval_episodes:.3f} "
                  f"last {interval_episodes} episodes={interval_seconds:.1f}s "
                  f"total_train={now - started - evaluation_seconds:.1f}s "
                  f"total_eval={evaluation_seconds:.1f}s", flush=True)
            progress_started, progress_episode = now, ep + 1
            progress_evaluation_seconds = evaluation_seconds
            progress_wins = 0
            agent.save(save_path)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    agent.save(save_path)
    print(f"Saved {agent_kind} agent -> {save_path}")
    if best_win_rate >= 0:
        print(f"Best validation win rate={best_win_rate:.3f} -> {best_path}")
    return agent


def exploration_epsilon(episode, episodes):
    """Schedule by games, so long/pass-heavy games don't exhaust exploration."""
    duration = max(1000, int(episodes * 0.8))
    return 0.5 + (0.05 - 0.5) * min(1.0, episode / duration)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", choices=["qlearning", "dqn", "ddqn", "shaped", "shaped-ql"],
                    default="ddqn")
    ap.add_argument("--episodes", type=int, default=5000)
    ap.add_argument("--num-players", type=int, default=4)
    ap.add_argument("--num-decks", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-path", default=None,
                    help="checkpoint path (defaults to checkpoints/<agent>_agent.pt/.npy)")
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--eval-games", type=int, default=100)
    ap.add_argument("--progress-every", type=int, default=100)
    ap.add_argument("--device", choices=["cpu", "mps", "cuda", "auto"], default="cpu",
                    help="training device; CPU is the default for these small networks")
    ap.add_argument("--torch-threads", type=int, default=1,
                    help="CPU threads for neural-network operations")
    ap.add_argument("--demo-games", type=int, default=None,
                    help="greedy warm-up games: 200 for ddqn/dqn/qlearning, 0 for shaped; use 0 for pure RL")
    ap.add_argument("--demo-updates", type=int, default=1000,
                    help="warm-up minibatches (32 examples each)")
    ap.add_argument("--demo-weight", type=float, default=0.1,
                    help="demonstration rehearsal weight during reinforcement learning")
    ap.add_argument("--lambda1", type=float, default=0.2)
    ap.add_argument("--lambda2", type=float, default=0.2)
    ap.add_argument("--potential-scale", type=float, default=0.5)
    ap.add_argument("--trick-scale", type=float, default=0.05)
    ap.add_argument("--opponent", choices=["greedy", "random", "mixed"], default="greedy",
                    help="opponent policy used for the other seats during training")
    ap.add_argument("--eval-opponent", choices=["greedy", "random", "mixed"], default="greedy",
                    help="opponent policy used for the periodic win-rate evaluation")
    ap.add_argument("--reward", choices=["win", "default", "basic"], default=None,
                    help="defaults to win (first-place outcome with potential shaping), except "
                         "shaped agents; default uses per-card plus placement; basic uses per-card plus win")
    args = ap.parse_args()

    if args.save_path is None:
        ext = "npy" if args.agent in ("qlearning", "shaped-ql") else "pt"
        name = f"{args.agent}_agent.{ext}"
        if args.reward == "basic":
            name = f"basic_{name}"
        args.save_path = os.path.join("checkpoints", name)

    train(args.agent, args.episodes, args.num_players, args.num_decks,
          args.seed, args.save_path, args.eval_every,
          lambda1=args.lambda1, lambda2=args.lambda2,
          potential_scale=args.potential_scale, trick_scale=args.trick_scale,
          opponent=args.opponent, eval_opponent=args.eval_opponent,
          reward_scheme=args.reward, progress_every=args.progress_every,
          eval_games=args.eval_games, device=args.device, torch_threads=args.torch_threads,
          demo_games=args.demo_games, demo_updates=args.demo_updates, demo_weight=args.demo_weight)


if __name__ == "__main__":
    main()
