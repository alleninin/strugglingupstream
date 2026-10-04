"""Compatibility CLI for shaped training; the shared harness owns the loop."""

import argparse
from pathlib import Path
import sys

ROOT = str(Path(__file__).resolve().parents[2])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from archive.legacy.train import train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent-type", choices=["ddqn", "qlearning"], default="ddqn")
    ap.add_argument("--episodes", type=int, default=5000)
    ap.add_argument("--num-players", type=int, default=4)
    ap.add_argument("--num-decks", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-path")
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--device", choices=["cpu", "mps", "cuda", "auto"], default="cpu")
    ap.add_argument("--torch-threads", type=int, default=1)
    ap.add_argument("--lambda1", type=float, default=0.2)
    ap.add_argument("--lambda2", type=float, default=0.2)
    ap.add_argument("--potential-scale", type=float, default=0.5)
    ap.add_argument("--trick-scale", type=float, default=0.05)
    args = ap.parse_args()
    kind = "shaped-ql" if args.agent_type == "qlearning" else "shaped"
    extension = "npy" if args.agent_type == "qlearning" else "pt"
    save_path = args.save_path or f"checkpoints/shaped_{args.agent_type}_agent.{extension}"
    train(kind, args.episodes, args.num_players, args.num_decks, args.seed,
          save_path, args.eval_every, lambda1=args.lambda1, lambda2=args.lambda2,
          potential_scale=args.potential_scale, trick_scale=args.trick_scale,
          device=args.device, torch_threads=args.torch_threads)


if __name__ == "__main__":
    main()
