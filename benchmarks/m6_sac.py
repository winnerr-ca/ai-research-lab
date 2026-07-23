"""M6 end-to-end implementation validation: multi-seed SAC on Pendulum-v1.

Deterministic and stochastic final evaluations per seed, plus a 2-seed
fixed-alpha configuration check. MuJoCo-scale validation is explicitly out
of scope on this hardware and no such results are claimed.

Usage::

    uv run python benchmarks/m6_sac.py [--seeds 0 1 2 3 4]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import statistics
from pathlib import Path
from typing import Any

import gymnasium as gym
import torch

from rlcore.agents.sac.train import SacConfig, train
from rlcore.evaluation import evaluate_policy
from rlcore.experiments import run_metadata

logger = logging.getLogger("m6_validation")

STOCHASTIC_EVAL_SEED_OFFSET = 40_000


def run_seed(seed: int, *, auto_alpha: bool = True) -> dict[str, Any]:
    """Train one seed; evaluate deterministically and stochastically."""
    config = SacConfig(seed=seed, auto_alpha=auto_alpha)
    result = train(config)

    eval_env: gym.Env[Any, Any] = gym.make(config.env_id)
    eval_env.reset(seed=STOCHASTIC_EVAL_SEED_OFFSET + seed)
    generator = torch.Generator().manual_seed(STOCHASTIC_EVAL_SEED_OFFSET + seed)

    def stochastic_action(obs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            action, _ = result.actor.sample(obs, generator=generator)
        return action

    stochastic = evaluate_policy(eval_env, stochastic_action, episodes=config.eval_episodes)
    eval_env.close()

    record = {
        "seed": seed,
        "auto_alpha": auto_alpha,
        "stopped_early": result.stopped_early,
        "total_env_steps": result.total_env_steps,
        "wall_time_s": round(result.wall_time_s, 1),
        "deterministic_return_mean": result.final_eval.mean_return,
        "deterministic_return_std": result.final_eval.std_return,
        "deterministic_returns": list(result.final_eval.episode_returns),
        "stochastic_return_mean": stochastic.mean_return,
        "stochastic_return_std": stochastic.std_return,
        "final_alpha": result.history[-1]["alpha"],
    }
    logger.info(
        "seed %d%s: det %.1f±%.1f | stoch %.1f±%.1f | %d steps | %.0fs%s",
        seed,
        "" if auto_alpha else " (fixed-alpha)",
        record["deterministic_return_mean"],
        record["deterministic_return_std"],
        record["stochastic_return_mean"],
        record["stochastic_return_std"],
        record["total_env_steps"],
        record["wall_time_s"],
        " | stopped early" if result.stopped_early else "",
    )
    return record


def section_table(records: list[dict[str, Any]]) -> list[str]:
    """Markdown table rows for one section."""
    det = [r["deterministic_return_mean"] for r in records]
    return [
        "| Seed | Env steps | Wall time (s) | Stopped early | "
        "Deterministic eval (mean±std) | Stochastic eval (mean±std) | Final alpha |",
        "|---|---|---|---|---|---|---|",
        *[
            f"| {r['seed']} | {r['total_env_steps']} | {r['wall_time_s']} | "
            f"{'yes' if r['stopped_early'] else 'no'} | "
            f"{r['deterministic_return_mean']:.1f} ± {r['deterministic_return_std']:.1f} | "
            f"{r['stochastic_return_mean']:.1f} ± {r['stochastic_return_std']:.1f} | "
            f"{r['final_alpha']:.3f} |"
            for r in records
        ],
        "",
        f"Aggregate deterministic means: mean {statistics.mean(det):.1f}, "
        f"median {statistics.median(det):.1f}, min/max {min(det):.1f}/{max(det):.1f}.",
        "",
    ]


def main() -> None:
    """Run the validation experiment end to end."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--out-dir", type=Path, default=Path("benchmarks/results"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    sections = {
        "pendulum_auto": [run_seed(seed) for seed in args.seeds],
        "pendulum_fixed_alpha": [run_seed(seed, auto_alpha=False) for seed in args.seeds[:2]],
    }
    environment = run_metadata()
    (args.out_dir / "m6_sac.json").write_text(
        json.dumps(
            {
                "configs": {
                    "auto": dataclasses.asdict(SacConfig(seed=0)),
                    "fixed_alpha": dataclasses.asdict(SacConfig(seed=0, auto_alpha=False)),
                },
                "environment": environment,
                "runs": sections,
            },
            indent=2,
        )
    )

    lines = [
        "# M6 end-to-end implementation validation: SAC on Pendulum-v1",
        "",
        "**Scope of this result.** End-to-end *implementation* validation on",
        "a single accessible continuous-control task. Not evidence of",
        "algorithmic superiority; no MuJoCo results are claimed (MuJoCo-scale",
        "validation is out of scope on this hardware). The fixed-alpha section",
        "is a configuration-extension check, not a tuned comparison.",
        "",
        "Evaluation protocol matches earlier reports (child-seeded train/eval",
        "envs, periodic deterministic evaluation with early stop, fresh final",
        "deterministic evaluation, independent stochastic evaluation seeded",
        f"`{STOCHASTIC_EVAL_SEED_OFFSET} + seed`). Full configs in the JSON record.",
        "",
        "## Environment",
        "",
        *[f"- {key}: `{value}`" for key, value in environment.items()],
        "",
        "## SAC (automatic temperature) — Pendulum-v1",
        "",
        *section_table(sections["pendulum_auto"]),
        "## SAC (fixed alpha = 0.2) — Pendulum-v1 (configuration check, 2 seeds)",
        "",
        *section_table(sections["pendulum_fixed_alpha"]),
        "Reproducibility scope: pinned software versions above, CPU, comparable hardware only.",
        "",
        "## Reproduce",
        "",
        "```sh",
        f"uv run python benchmarks/m6_sac.py --seeds {' '.join(map(str, args.seeds))}",
        "```",
        "",
    ]
    report = args.out_dir / "m6_sac.md"
    report.write_text("\n".join(lines))
    logger.info("report written to %s", report)


if __name__ == "__main__":
    main()
