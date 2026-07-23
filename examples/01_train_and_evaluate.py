"""Tutorial 1: train an agent and inspect its run directory.

Run it:

    uv run python examples/01_train_and_evaluate.py            # full demo
    uv run python examples/01_train_and_evaluate.py --updates 2  # quick smoke

Everything the run produced — config, per-update metrics, final result,
model weights, a summary, and a learning-curve plot — lands in one run
directory. That directory, not the console output, is the record.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rlcore.agents.ppo.train import PpoConfig, train


def main() -> None:
    """Train tiny-budget PPO on CartPole and print the recorded result."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--updates", type=int, default=15)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/tutorial-01"))
    args = parser.parse_args()

    config = PpoConfig(
        seed=0,
        total_updates=args.updates,
        eval_every=0,  # final evaluation always runs; skip periodic ones here
        stop_return=None,  # fixed budget: comparable across invocations
    )
    result = train(config, out_dir=args.out_dir)

    print(
        f"final deterministic eval: {result.final_eval.mean_return:.1f} "
        f"± {result.final_eval.std_return:.1f}"
    )
    print(f"env steps consumed:       {result.total_env_steps}")

    # The run directory is the ground truth; result.json matches the
    # returned object, and run.json records seed/versions/commit/device.
    recorded = json.loads((args.out_dir / "result.json").read_text())
    assert recorded["total_env_steps"] == result.total_env_steps
    print(
        f"run directory:            {args.out_dir} "
        f"({sorted(p.name for p in args.out_dir.iterdir())})"
    )


if __name__ == "__main__":
    main()
