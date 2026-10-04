"""Reproduce the checkpoint audit; run from the repository root."""
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from agents.dqn_agent import DQNAgent
from agents.ddqn_agent import DDQNAgent
from agents.random_agent import RandomAgent
from bots.greedy_bot import GreedyBot
from env.features import feature_dims
from training.train import evaluate_vs_opponent
from training.demonstrations import collect_demonstrations, Demonstrations

OUT = Path(__file__).resolve().parent
GAMES, SEED = 300, 731926

def main():
    torch.set_num_threads(1)
    policies = []
    metadata = {}
    # Load all policies before evaluation, so a running trainer cannot change them.
    for label, cls, path in [
        ('dqn_best', DQNAgent, 'checkpoints/dqn_win_agent.best.pt'),
        ('dqn_latest', DQNAgent, 'checkpoints/dqn_win_agent.pt'),
        ('ddqn_best', DDQNAgent, 'checkpoints/ddqn_v3_agent.best.pt'),
    ]:
        data = Path(path).read_bytes()
        snapshot = OUT / (label + '.pt')
        snapshot.write_bytes(data)
        agent = cls(*feature_dims(4, 2), seed=0, device='cpu')
        agent.load(str(snapshot))
        policies.append((label, agent))
        metadata[label] = dict(source=path, sha256=hashlib.sha256(data).hexdigest())
    policies += [('greedy', GreedyBot(seed=0)), ('random', RandomAgent(seed=0))]
    results = dict(games=GAMES, seed=SEED, checkpoints=metadata, rates={})
    for label, agent in policies:
        rates = {op: evaluate_vs_opponent(agent, 4, 2, SEED, n=GAMES, opponent=op)
                 for op in ('greedy', 'random')}
        results['rates'][label] = rates
        print(label, rates, flush=True)
        (OUT / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
    # Reconstruct default seed-0 DQN warm-up to measure what RL changes.
    agent = DQNAgent(*feature_dims(4, 2), lr=3e-4, gamma=1, seed=0, device='cpu')
    demo = Demonstrations(collect_demonstrations(200, 4, 2, 0), seed=0)
    demo.pretrain(agent, 1000)
    agent.save(str(OUT / 'dqn_warmup.pt'))
    rates = {op: evaluate_vs_opponent(agent, 4, 2, SEED, n=GAMES, opponent=op)
             for op in ('greedy', 'random')}
    results['rates']['dqn_warmup_seed0'] = rates
    print('dqn_warmup_seed0', rates, flush=True)
    (OUT / 'results.json').write_text(json.dumps(results, indent=2) + '\n')

if __name__ == '__main__':
    main()
