"""Device selection and construction of inference bots."""

import torch


def resolve_device(device="auto"):
    if device == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    selected = torch.device(device)
    if selected.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is not available in this Python environment")
    if selected.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is not available in this Python environment")
    return selected


def load_agent(kind, checkpoint=None, num_players=4, num_decks=2, seed=None):
    """Construct a bot for inference; requested checkpoints must load successfully."""
    if kind == "greedy":
        from bots.greedy_bot import GreedyBot

        return GreedyBot(num_players=num_players, num_decks=num_decks, seed=seed)
    if kind == "random":
        from bots.random_bot import RandomAgent

        return RandomAgent(seed=seed)
    if kind != "ddqn":
        raise ValueError(f"unknown agent: {kind}")
    from agents.ddqn_agent import DDQNAgent
    from env.features import feature_dims

    agent = DDQNAgent(
        *feature_dims(num_players, num_decks), epsilon=0, seed=seed, device="cpu"
    )
    agent.load(checkpoint or "checkpoints/ddqn_agent.best.pt")
    return agent
