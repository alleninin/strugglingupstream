"""Inspect learned decisions on a common held-out greedy demonstration set."""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from agents.dqn_agent import DQNAgent
from agents.ddqn_agent import DDQNAgent
from env.features import feature_dims
from training.demonstrations import collect_demonstrations

out = Path(__file__).resolve().parent
torch.set_num_threads(1)
examples = collect_demonstrations(100, 4, 2, 982711)
results = {}
for name in ('dqn_best', 'dqn_latest', 'ddqn_best', 'dqn_warmup'):
    path = out / (name + '.pt')
    if not path.exists():
        continue
    cls = DDQNAgent if name.startswith('ddqn') else DQNAgent
    agent = cls(*feature_dims(4, 2), seed=0, device='cpu')
    agent.load(str(path))
    agent.epsilon = 0
    agree = optional_pass = pass_available = 0
    chosen_values = []
    for item in examples:
        chosen = agent.act(item.state, item.moves)
        agree += chosen == item.moves[item.chosen]
        optional_pass += chosen.is_pass
        pass_available += any(m.is_pass for m in item.moves)
        if name.startswith('dqn'):
            with torch.no_grad():
                chosen_values.append(agent._q_batch([agent._phi(item.state, chosen)], agent.policy_net).item())
    results[name] = dict(decisions=len(examples), teacher_agreement=agree/len(examples),
        optional_pass_fraction=optional_pass/max(1, pass_available))
    if chosen_values:
        results[name]['q_quantiles'] = np.quantile(chosen_values, [0, .1, .5, .9, 1]).tolist()
results['note'] = 'Common teacher-visited states, not on-policy games; disagreement is not necessarily an error.'
(out / 'probe.json').write_text(json.dumps(results, indent=2) + '\n')
print(json.dumps(results, indent=2))
