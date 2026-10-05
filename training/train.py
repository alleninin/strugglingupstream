import argparse
import copy
import os
import random
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from env import features
from env.env import ZhengShangYouEnv


def make_opponent_policies(num_players, seed, num_decks=2, opponent="greedy"):
    from bots.greedy_bot import GreedyBot
    from bots.random_bot import RandomAgent

    if opponent not in ("greedy", "random", "mixed"):
        raise ValueError(f"unknown opponent: {opponent}")
    policies = []
    rng = random.Random(seed)
    for seat in range(1, num_players):
        opponent_seed = None if seed is None else seed * 100 + seat
        kind = rng.choice(("greedy", "random")) if opponent == "mixed" else opponent
        bot = (
            RandomAgent(seed=opponent_seed)
            if kind == "random"
            else GreedyBot(
                num_players=num_players, num_decks=num_decks, seed=opponent_seed
            )
        )
        policies.append(bot.act)
    return policies


def evaluate_vs_opponent(agent, num_players, num_decks, seed, n=100, opponent="greedy"):
    if n < 1:
        raise ValueError("evaluation requires at least one game")
    inner = getattr(agent, "inner", agent)
    rng = getattr(inner, "rng", None)
    numpy_rng = rng is not None and hasattr(rng, "bit_generator")
    rng_state = (
        copy.deepcopy(rng.bit_generator.state)
        if numpy_rng
        else rng.getstate()
        if rng is not None
        else None
    )
    old_eps = getattr(agent, "epsilon", 0.0)
    agent.epsilon = 0.0
    wins = 0
    try:
        for i in range(n):
            game_seed = None if seed is None else (1 << 62) + seed * 1000003 + i
            opp = make_opponent_policies(num_players, game_seed, num_decks, opponent)
            env = ZhengShangYouEnv(
                num_players=num_players,
                num_decks=num_decks,
                opponent_policies=opp,
                seed=game_seed,
                reward_scheme="win",
                history_features=getattr(inner, "history_features", False),
            )
            state = env.reset()
            done = env.done
            while not done:
                legal = env.get_legal_moves()
                action = agent.act(state, legal)
                next_state, reward, done, info = env.step(action)
                state = next_state
            if env.game.finish_order[0] == env.agent_seat:
                wins += 1
    finally:
        agent.epsilon = old_eps
        if rng_state is not None:
            if numpy_rng:
                rng.bit_generator.state = rng_state
            else:
                rng.setstate(rng_state)
    return wins / n


def train(
    agent_kind,
    episodes,
    num_players,
    num_decks,
    seed,
    save_path,
    eval_every,
    opponent="curriculum",
    eval_opponent="greedy",
    reward_scheme="win",
    progress_every=0,
    eval_games=100,
    device="cpu",
    torch_threads=1,
    n_step=3,
    learning_starts=1000,
    train_every=4,
    gamma=None,
    planning_features=True,
    partition_features=None,
    expert_exploration=0.5,
    endgame_rate=0.0,
    target_method="ddqn",
    history_features=False,
    selfplay_checkpoint=None,
    supervision_weight=0.0,
    supervision_fraction=0.6,
):
    """Train the current DDQN; historical learners live in archive/legacy."""
    import torch

    from agents.ddqn_agent import DDQNAgent

    if agent_kind != "ddqn":
        raise ValueError(
            "only DDQN is active; use archive/legacy/train.py for older learners"
        )
    if selfplay_checkpoint and opponent != "selfplay":
        raise ValueError("a self-play checkpoint requires --opponent selfplay")
    if reward_scheme != "win":
        raise ValueError("current training uses the first-place win objective")
    if gamma is not None and not 0 < gamma <= 1:
        raise ValueError("gamma must be in (0, 1]")
    if episodes < 0 or torch_threads < 1:
        raise ValueError("episodes must be nonnegative and torch_threads positive")
    if eval_every and eval_games < 1:
        raise ValueError("evaluation requires at least one game")
    if not 0 <= endgame_rate <= 1:
        raise ValueError("endgame rate must be in [0, 1]")
    if not 0 <= expert_exploration <= 1:
        raise ValueError("expert exploration must be in [0, 1]")
    if partition_features is None:
        partition_features = planning_features
    torch.set_num_threads(torch_threads)
    s_dim, a_dim = features.feature_dims(num_players, num_decks)
    agent = DDQNAgent(
        s_dim,
        a_dim,
        lr=3e-4,
        gamma=1.0 if gamma is None else gamma,
        epsilon=0.5,
        epsilon_decay=1.0,
        min_epsilon=0.05,
        seed=seed,
        device=device,
        n_step=n_step,
        learning_starts=learning_starts,
        train_every=train_every,
        planning_features=planning_features,
        partition_features=partition_features,
        target_method=target_method,
        history_features=history_features,
    )
    from bots.greedy_bot import GreedyBot

    teacher = GreedyBot(num_players=num_players, num_decks=num_decks)

    from training.supervision import GreedySupervision

    supervision = GreedySupervision(supervision_weight, supervision_fraction, seed)
    if supervision_weight:
        agent.demonstrations = supervision

    from training.curriculum import EndgamePool, SelfPlayPool

    pool = (
        SelfPlayPool(num_players, num_decks, seed) if opponent == "selfplay" else None
    )
    if selfplay_checkpoint:
        anchor = DDQNAgent(s_dim, a_dim, epsilon=0, device=device)
        anchor.load(selfplay_checkpoint)
        pool.capture(anchor, anchor=True)
    practice = EndgamePool(seed) if endgame_rate else None
    initial_opponent = (
        curriculum_opponent(0, episodes)
        if opponent in ("curriculum", "selfplay")
        else opponent
    )
    opp = make_opponent_policies(num_players, seed, num_decks, initial_opponent)
    env = ZhengShangYouEnv(
        num_players=num_players,
        num_decks=num_decks,
        opponent_policies=opp,
        seed=seed,
        reward_scheme=reward_scheme,
        reward_discount=1.0 if gamma is None else gamma,
        history_features=history_features,
    )
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    print(
        f"[DDQN] device={agent.device} torch_threads={torch.get_num_threads()} "
        f"partition_features={agent.partition_features} expert_exploration={expert_exploration} n_step={agent.n_step} "
        f"learning_starts={agent.learning_starts} train_every={agent.train_every} "
        f"target_method={target_method} endgame_rate={endgame_rate} history_features={history_features} "
        f"selfplay_checkpoint={selfplay_checkpoint} supervision_weight={supervision_weight} "
        f"supervision_fraction={supervision_fraction}",
        flush=True,
    )

    stem, extension = os.path.splitext(save_path)
    best_path = stem + ".best" + extension
    best_win_rate = -1.0

    started = time.perf_counter()
    progress_started = started
    progress_episode = 0
    evaluation_seconds = 0.0
    progress_evaluation_seconds = 0.0
    progress_wins = 0
    practice_episodes = 0
    for ep in range(episodes):
        supervision.schedule(ep, episodes)
        agent.epsilon = exploration_epsilon(ep, episodes)
        episode_seed = (seed * 1000 + ep) if seed is not None else None
        episode_opponent = (
            curriculum_opponent(ep, episodes)
            if opponent in ("curriculum", "selfplay")
            else opponent
        )
        if pool is not None and (pool.snapshots or pool.anchor):
            episode_opponent = "selfplay"
            env.opponent_policies = pool.policies(episode_seed)
        elif opponent in ("mixed", "curriculum", "selfplay"):
            env.opponent_policies = make_opponent_policies(
                num_players, episode_seed, num_decks, episode_opponent
            )
        position = practice.sample(endgame_rate) if practice else None
        is_practice = position is not None
        state = (
            env.reset_from(position) if is_practice else env.reset(seed=episode_seed)
        )
        practice_episodes += is_practice
        agent.reset_episode()
        done = env.done
        while not done:
            legal = env.get_legal_moves()
            if practice and not is_practice:
                practice.consider(env, legal)
            supervision.add(agent, state, legal, teacher)
            action = training_action(agent, state, legal, teacher, expert_exploration)
            next_state, reward, done, info = env.step(action)
            agent.observe(
                (
                    state,
                    action,
                    reward,
                    next_state,
                    done,
                    info["next_legal_moves"],
                    legal,
                )
            )
            state = next_state

        progress_wins += int(
            bool(env.game.finish_order) and env.game.finish_order[0] == env.agent_seat
        )

        if practice and not is_practice:
            practice.finish(env.game.finish_order[0] != env.agent_seat)
        if pool is not None and (ep + 1) % 500 == 0:
            pool.capture(agent)

        if eval_every and (ep + 1) % eval_every == 0:
            eval_started = time.perf_counter()
            wr = evaluate_vs_opponent(
                agent,
                num_players,
                num_decks,
                seed,
                n=eval_games,
                opponent=eval_opponent,
            )
            eval_seconds = time.perf_counter() - eval_started
            evaluation_seconds += eval_seconds
            print(
                f"[ep {ep + 1:>{len(str(episodes))}}] win_rate_vs_{eval_opponent} = {wr:.3f} "
                f"({eval_games} evaluation games in {eval_seconds:.1f}s)",
                flush=True,
            )
            if wr > best_win_rate:
                best_win_rate = wr
                agent.save(best_path)

            agent.save(save_path)
        if progress_every and (ep + 1) % progress_every == 0:
            now = time.perf_counter()
            interval_episodes = ep + 1 - progress_episode
            interval_seconds = (
                now
                - progress_started
                - (evaluation_seconds - progress_evaluation_seconds)
            )
            print(
                f"[{agent_kind} {ep + 1}/{episodes}] epsilon={agent.epsilon:.3f} "
                f"opponent={episode_opponent} "
                f"training_win_rate={progress_wins / interval_episodes:.3f} "
                f"practice_episodes={practice_episodes} supervision_weight={supervision.weight:.3f} "
                f"last {interval_episodes} episodes={interval_seconds:.1f}s "
                f"total_train={now - started - evaluation_seconds:.1f}s "
                f"total_eval={evaluation_seconds:.1f}s",
                flush=True,
            )
            progress_started, progress_episode = now, ep + 1
            progress_evaluation_seconds = evaluation_seconds
            progress_wins = 0
            agent.save(save_path)

    agent.save(save_path)
    print(f"Saved {agent_kind} agent -> {save_path}")
    if best_win_rate >= 0:
        print(f"Best validation win rate={best_win_rate:.3f} -> {best_path}")
    return agent


def training_action(agent, state, legal, teacher, expert_fraction):
    """Split epsilon between expert and uniform exploration; keep exploitation.

    Only training consults the teacher. Its actions receive ordinary environment
    rewards and enter the same replay buffer; Q values are never imitation logits.
    """
    epsilon = agent.epsilon
    if not expert_fraction or not epsilon:
        return agent.act(state, legal)
    expert_probability = epsilon * expert_fraction
    if agent.rng.random() < expert_probability:
        return teacher.act(state, legal)

    agent.epsilon = epsilon * (1 - expert_fraction) / (1 - expert_probability)
    try:
        return agent.act(state, legal)
    finally:
        agent.epsilon = epsilon


def exploration_epsilon(episode, episodes):
    """Schedule by games, so long/pass-heavy games don't exhaust exploration."""
    duration = max(1000, int(episodes * 0.8))
    return 0.5 + (0.05 - 0.5) * min(1.0, episode / duration)


def curriculum_opponent(episode, episodes):
    progress = episode / max(1, episodes)
    return "random" if progress < 0.2 else "mixed" if progress < 0.5 else "greedy"


def main():
    ap = argparse.ArgumentParser(
        description="Train DDQN against a random/mixed/greedy curriculum."
    )
    ap.add_argument("--agent", choices=["ddqn"], default="ddqn")
    ap.add_argument("--episodes", type=int, default=2500)
    ap.add_argument("--num-players", type=int, default=4)
    ap.add_argument("--num-decks", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-path", default="checkpoints/ddqn_agent.pt")
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--eval-games", type=int, default=100)
    ap.add_argument("--progress-every", type=int, default=100)
    ap.add_argument("--device", choices=["cpu", "mps", "cuda", "auto"], default="cpu")
    ap.add_argument("--torch-threads", type=int, default=1)
    ap.add_argument("--n-step", type=int, default=3)
    ap.add_argument("--learning-starts", type=int, default=1000)
    ap.add_argument("--train-every", type=int, default=4)
    ap.add_argument("--gamma", type=float, default=None)
    ap.add_argument(
        "--planning-features", action=argparse.BooleanOptionalAction, default=True
    )
    ap.add_argument(
        "--partition-features", action=argparse.BooleanOptionalAction, default=None
    )
    ap.add_argument(
        "--expert-exploration",
        type=float,
        default=0.5,
        help="fraction of epsilon exploration following Greedy (0 disables)",
    )
    ap.add_argument(
        "--opponent",
        choices=["greedy", "random", "mixed", "curriculum", "selfplay"],
        default="curriculum",
    )
    ap.add_argument(
        "--eval-opponent", choices=["greedy", "random", "mixed"], default="greedy"
    )
    ap.add_argument(
        "--endgame-rate",
        type=float,
        default=0.0,
        help="fraction of episodes replaying difficult training endgames",
    )
    ap.add_argument("--target-method", choices=["ddqn", "monte-carlo"], default="ddqn")
    ap.add_argument(
        "--history-features", action=argparse.BooleanOptionalAction, default=False
    )
    ap.add_argument(
        "--selfplay-checkpoint", help="keep this trained DDQN as a frozen opponent"
    )
    ap.add_argument(
        "--supervision-weight",
        type=float,
        default=0.0,
        help="initial Greedy ranking-loss weight (0 disables)",
    )
    ap.add_argument(
        "--supervision-fraction",
        type=float,
        default=0.6,
        help="fraction of training over which Greedy supervision fades to zero",
    )
    args = vars(ap.parse_args())
    train(agent_kind=args.pop("agent"), **args)


if __name__ == "__main__":
    main()
