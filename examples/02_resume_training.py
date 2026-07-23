"""Tutorial 2: checkpoint a run, kill it, and resume it exactly.

Run it:

    uv run python examples/02_resume_training.py
    uv run python examples/02_resume_training.py --steps 3000  # quick smoke

The demonstration trains DQN twice: once straight through, and once as
checkpoint-then-resume in two stages. Exact resume means the two produce
IDENTICAL metric histories — the platform treats anything less as a bug
(this property is fresh-process-tested for every algorithm).
"""

from __future__ import annotations

import argparse
import math
import shutil
from pathlib import Path

from rlcore.agents.dqn.train import DqnConfig, train


def main() -> None:
    """Show that resume-from-checkpoint equals an uninterrupted run."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/tutorial-02"))
    args = parser.parse_args()
    half = args.steps // 2

    def config(total_steps: int, checkpoint_every: int = 0) -> DqnConfig:
        return DqnConfig(
            seed=0,
            total_steps=total_steps,
            checkpoint_every=checkpoint_every,
            warmup_steps=500,
            log_every=500,
            eval_every=0,
            stop_return=None,
        )

    # Reference: one uninterrupted run over the full budget.
    ref_dir = args.out_dir / "reference"
    shutil.rmtree(ref_dir, ignore_errors=True)
    reference = train(config(args.steps), out_dir=ref_dir)

    # Interrupted: train half the budget with a checkpoint at the midpoint...
    resumed_dir = args.out_dir / "resumed"
    shutil.rmtree(resumed_dir, ignore_errors=True)
    train(config(half, checkpoint_every=half), out_dir=resumed_dir)
    # ...then resume with the budget extended to the full amount. Extending
    # the budget is the ONLY config change resume accepts; anything else is
    # rejected as a mismatch.
    resumed = train(
        config(args.steps, checkpoint_every=half),
        out_dir=resumed_dir,
        resume_from=resumed_dir / "checkpoint.pt",
    )

    identical = len(reference.history) == len(resumed.history) and all(
        r.keys() == s.keys()
        and all((math.isnan(r[k]) and math.isnan(s[k])) or r[k] == s[k] for k in r)
        for r, s in zip(reference.history, resumed.history, strict=True)
    )
    print(f"histories identical after resume: {identical}")
    if not identical:
        raise SystemExit("exact-resume property violated — please file a bug")


if __name__ == "__main__":
    main()
