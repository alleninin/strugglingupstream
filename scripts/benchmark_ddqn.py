"""Measure DDQN training without evaluation or checkpoint writes in the timed loop."""
import argparse
from contextlib import ExitStack
from unittest.mock import patch
import importlib.util
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from agents.ddqn_agent import DDQNAgent
from env.env import ZhengShangYouEnv
from env.features import feature_dims


def array_bytes(value):
    if isinstance(value, np.ndarray):
        return value.nbytes
    if isinstance(value, (tuple, list)):
        return sum(array_bytes(item) for item in value)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', choices=['cpu', 'auto'], default='cpu')
    parser.add_argument('--reference', type=Path, help='saved pre-optimization ddqn_agent.py')
    parser.add_argument('--json', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    factory = DDQNAgent
    if args.reference:
        spec = importlib.util.spec_from_file_location('agents._ddqn_reference', args.reference)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        factory = module.DDQNAgent
    # The saved reference constructor predates an explicit device argument.
    # Disable accelerator selection only during construction for a CPU comparison.
    with ExitStack() as stack:
        if args.device == 'cpu':
            stack.enter_context(patch('torch.backends.mps.is_available', return_value=False))
            stack.enter_context(patch('torch.cuda.is_available', return_value=False))
        agent = factory(*feature_dims(4, 2), epsilon=.5, epsilon_decay=.9998,
                        min_epsilon=.05, seed=args.seed)
    env = ZhengShangYouEnv(seed=args.seed)
    started = time.perf_counter()
    for episode in range(args.episodes):
        state = env.reset(seed=args.seed * 1000 + episode)
        while not env.game.done:
            legal = env.get_legal_moves()
            action = agent.act(state, legal)
            next_state, reward, done, info = env.step(action)
            agent.observe((state, action, reward, next_state, done, info['next_legal_moves'], legal))
            state = next_state
    seconds = time.perf_counter() - started
    report = {'implementation': str(args.reference) if args.reference else 'current',
              'episodes': args.episodes, 'seed': args.seed, 'num_players': 4, 'num_decks': 2,
              'opponent': 'greedy', 'device': str(agent.device), 'cpu_threads': torch.get_num_threads(),
              'seconds': seconds, 'transitions': agent.step_count,
              'updates': max(0, agent.step_count - agent.batch_size + 1),
              'replay_array_bytes': array_bytes(agent.buffer.tree.data),
              'torch': torch.__version__,
              'finite_weights': all(torch.isfinite(p).all().item() for p in agent.policy_net.parameters())}
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
