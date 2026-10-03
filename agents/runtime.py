"""Explicit device selection shared by the neural agents."""
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
