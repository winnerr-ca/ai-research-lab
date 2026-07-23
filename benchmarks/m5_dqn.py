"""M5 end-to-end implementation validation: multi-seed DQN.

Three sections: standard DQN on CartPole-v1 and Acrobot-v1 (5 seeds each),
and Double DQN on CartPole-v1 (2 seeds, labeled a configuration-extension
check, not a comparison claim). Writes machine-readable JSON plus a
Markdown report to ``benchmarks/results/``.

Usage::

    uv run python benchmarks/m5_dqn.py [--seeds 0 1 2 3 4]
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

from rlcore.agents.dqn.train import DqnConfig, train
from rlcore.evaluation import evaluate_policy
from rlcore.experiments import run_metadata

logger = logging.getLogger("m5_validation")

STOCHASTIC_EVAL_SEED_OFFSET = 30_000


def config_for(env_id: str, seed: int, *, double_q: bool = False) -> DqnConfig:
    """Per-env validation config (documented in the report)."""
    if env_id == "Acrobot-v1":
        return DqnConfig(env_id=env_id, seed=seed, double_q=double_q, stop_return=-100.0)
    return DqnConfig(env_id=env_id, seed=seed, double_q=double_q)


def run_seed(env_id: str, seed: int, *, double_q: bool = False) -> dict[str, Any]:
    """Train one seed; evaluate greedily (fresh) and eps-greedily (0.05)."""
    config = config_for(env_id, seed, double_q=double_q)
    result = train(config)

    eval_env: gym.Env[Any, Any] = gym.make(config.env_id)
    eval_env.reset(seed=STOCHASTIC_EVAL_SEED_OFFSET + seed)
    generator = torch.Generator().manual_seed(STOCHASTIC_EVAL_SEED_OFFSET + seed)
    from rlcore.agents.dqn.model import epsilon_greedy_action

    stochastic = evaluate_policy(
        eval_env,
        lambda obs: epsilon_greedy_action(result.q_net, obs, epsilon=0.05, generator=generator),
        episodes=config.eval_episodes,
    )
    eval_env.close()

    record = {
        "env_id": env_id,
        "seed": seed,
        "double_q": double_q,
        "stopped_early": result.stopped_early,
        "total_env_steps": result.total_env_steps,
        "wall_time_s": round(result.wall_time_s, 1),
        "greedy_return_mean": result.final_eval.mean_return,
        "greedy_return_std": result.final_eval.std_return,
        "greedy_returns": list(result.final_eval.episode_returns),
        "eps05_return_mean": stochastic.mean_return,
        "eps05_return_std": stochastic.std_return,
    }
    logger.info(
        "%s seed %d%s: greedy %.1f±%.1f | eps0.05 %.1f±%.1f | %d steps | %.0fs%s",
        env_id,
        seed,
        " (double)" if double_q else "",
        record["greedy_return_mean"],
        record["greedy_return_std"],
        record["eps05_return_mean"],
        record["eps05_return_std"],
        record["total_env_steps"],
        record["wall_time_s"],
        " | stopped early" if result.stopped_early else "",
    )
    return record


def section_table(records: list[dict[str, Any]]) -> list[str]:
    """Markdown table rows for one section."""
    greedy = [r["greedy_return_mean"] for r in records]
    return [
        "| Seed | Env steps | Wall time (s) | Stopped early | "
        "Greedy eval (mean±std) | eps=0.05 eval (mean±std) |",
        "|---|---|---|---|---|---|",
        *[
            f"| {r['seed']} | {r['total_env_steps']} | {r['wall_time_s']} | "
            f"{'yes' if r['stopped_early'] else 'no'} | "
            f"{r['greedy_return_mean']:.1f} ± {r['greedy_return_std']:.1f} | "
            f"{r['eps05_return_mean']:.1f} ± {r['eps05_return_std']:.1f} |"
            for r in records
        ],
        "",
        f"Aggregate greedy means: mean {statistics.mean(greedy):.1f}, "
        f"median {statistics.median(greedy):.1f}, "
        f"min/max {min(greedy):.1f}/{max(greedy):.1f}.",
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
        "cartpole": [run_seed("CartPole-v1", seed) for seed in args.seeds],
        "acrobot": [run_seed("Acrobot-v1", seed) for seed in args.seeds],
        "cartpole_double": [
            run_seed("CartPole-v1", seed, double_q=True) for seed in args.seeds[:2]
        ],
    }
    environment = run_metadata()
    (args.out_dir / "m5_dqn.json").write_text(
        json.dumps(
            {
                "configs": {
                    "cartpole": dataclasses.asdict(config_for("CartPole-v1", 0)),
                    "acrobot": dataclasses.asdict(config_for("Acrobot-v1", 0)),
                    "cartpole_double": dataclasses.asdict(
                        config_for("CartPole-v1", 0, double_q=True)
                    ),
                },
                "environment": environment,
                "runs": sections,
            },
            indent=2,
        )
    )

    lines = [
        "# M5 end-to-end implementation validation: DQN",
        "",
        "**Scope of this result.** End-to-end *implementation* validation:",
        "evidence that the implementation trains successfully across seeds and",
        "two discrete-action environments. Not evidence of algorithmic",
        "superiority; no cross-algorithm or standard-vs-Double comparison claim",
        "is made (budgets and sample sizes here do not support one).",
        "",
        "Evaluation protocol matches the M1/M2 reports (child-seeded train/",
        "eval envs, periodic greedy evaluation with early stop, fresh final",
        "greedy evaluation, independent eps=0.05 evaluation seeded "
        f"`{STOCHASTIC_EVAL_SEED_OFFSET} + seed`). Full configs in the JSON",
        "record; Acrobot uses stop_return=-100, all else default.",
        "",
        "## Environment",
        "",
        *[f"- {key}: `{value}`" for key, value in environment.items()],
        "",
        "## Standard DQN — CartPole-v1",
        "",
        *section_table(sections["cartpole"]),
        "## Standard DQN — Acrobot-v1",
        "",
        *section_table(sections["acrobot"]),
        "## Double DQN — CartPole-v1 (configuration-extension check, 2 seeds)",
        "",
        *section_table(sections["cartpole_double"]),
        "Reproducibility scope: pinned software versions above, CPU, comparable hardware only.",
        "",
        "## Reproduce",
        "",
        "```sh",
        f"uv run python benchmarks/m5_dqn.py --seeds {' '.join(map(str, args.seeds))}",
        "```",
        "",
    ]
    report = args.out_dir / "m5_dqn.md"
    report.write_text("\n".join(lines))
    logger.info("report written to %s", report)


if __name__ == "__main__":
    main()
