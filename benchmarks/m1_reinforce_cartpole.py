"""M1 validation experiment: multi-seed REINFORCE on CartPole-v1.

Trains one run per seed with the default :class:`ReinforceConfig`, evaluates
each trained policy greedily and stochastically, and writes a JSON record
plus a Markdown report (config, seeds, package versions, per-seed and
aggregate results) to ``benchmarks/results/``.

Usage::

    uv run python benchmarks/m1_reinforce_cartpole.py [--seeds 0 1 2 3 4]
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

from rlcore.agents.reinforce.train import ReinforceConfig, train
from rlcore.evaluation import evaluate_policy
from rlcore.experiments import run_metadata

logger = logging.getLogger("m1_validation")

STOCHASTIC_EVAL_SEED_OFFSET = 10_000


def run_seed(seed: int) -> dict[str, Any]:
    """Train one seed and evaluate the result greedily and stochastically."""
    config = ReinforceConfig(seed=seed)
    result = train(config)

    eval_env: gym.Env[Any, Any] = gym.make(config.env_id)
    eval_env.reset(seed=STOCHASTIC_EVAL_SEED_OFFSET + seed)
    generator = torch.Generator().manual_seed(STOCHASTIC_EVAL_SEED_OFFSET + seed)
    stochastic = evaluate_policy(
        eval_env,
        lambda obs: result.policy.act(obs, generator=generator, deterministic=False),
        episodes=config.eval_episodes,
    )
    eval_env.close()

    record = {
        "seed": seed,
        "stopped_early": result.stopped_early,
        "updates_used": len(result.history),
        "total_episodes": result.total_episodes,
        "total_env_steps": result.total_env_steps,
        "wall_time_s": round(result.wall_time_s, 1),
        "greedy_return_mean": result.final_eval.mean_return,
        "greedy_return_std": result.final_eval.std_return,
        "greedy_returns": list(result.final_eval.episode_returns),
        "stochastic_return_mean": stochastic.mean_return,
        "stochastic_return_std": stochastic.std_return,
        "stochastic_returns": list(stochastic.episode_returns),
    }
    logger.info(
        "seed %d: greedy %.1f±%.1f | stochastic %.1f±%.1f | %d episodes | %.0fs%s",
        seed,
        record["greedy_return_mean"],
        record["greedy_return_std"],
        record["stochastic_return_mean"],
        record["stochastic_return_std"],
        record["total_episodes"],
        record["wall_time_s"],
        " | stopped early" if result.stopped_early else "",
    )
    return record


def write_report(records: list[dict[str, Any]], config: ReinforceConfig, out_dir: Path) -> Path:
    """Write the JSON record and Markdown report; return the report path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    environment = run_metadata()
    (out_dir / "m1_reinforce_cartpole.json").write_text(
        json.dumps(
            {"config": dataclasses.asdict(config), "environment": environment, "runs": records},
            indent=2,
        )
    )

    greedy_means = [r["greedy_return_mean"] for r in records]
    config_dict: dict[str, Any] = dataclasses.asdict(config)
    config_dict["seed"] = "per-run (see table)"
    lines = [
        "# M1 end-to-end implementation validation: REINFORCE on CartPole-v1",
        "",
        f"Seeds: {[r['seed'] for r in records]} — one full training run each, "
        "default `ReinforceConfig`, greedy + stochastic final evaluation "
        f"({config.eval_episodes} episodes each).",
        "",
        "**Scope of this result.** This is an end-to-end *implementation*",
        "validation: evidence that the implementation trains successfully on",
        "this task across seeds. It is not evidence of algorithmic",
        "superiority, and no performance claim beyond this task, this",
        "configuration, and the environment recorded below is intended.",
        "",
        "## Evaluation protocol",
        "",
        "- **Seed streams:** the root seed derives independent child seeds",
        "  via `SeedSequence` — child 0 seeds the training env, child 1 the",
        "  training action space, child 2 the *separate* evaluation env.",
        "  Action sampling during training draws from the bundle's explicit",
        "  torch generator.",
        f"- **Evaluation frequency:** a greedy ({config.eval_episodes}-episode)",
        f"  evaluation every {config.eval_every} updates on the dedicated eval",
        "  env; its RNG stream continues across evaluations (episodes are",
        "  fresh draws from the stream, not re-seeded per evaluation).",
        "- **Stopping rule:** training stops early when a periodic greedy",
        f"  evaluation's mean return reaches {config.stop_return}, with a hard",
        f"  budget of {config.updates} updates otherwise.",
        "- **Final evaluation:** after training ends (early stop or budget),",
        f"  a *fresh* greedy {config.eval_episodes}-episode evaluation runs on",
        "  the same eval env's continuing stream — the greedy numbers in the",
        "  table are this re-evaluation, not the evaluation that triggered",
        "  the stop.",
        "- **Stochastic evaluation:** a separate env and sampling generator,",
        f"  both seeded `{STOCHASTIC_EVAL_SEED_OFFSET} + seed`, independent of",
        "  all training streams.",
        "",
        "## Configuration",
        "",
        "```json",
        json.dumps(config_dict, indent=2),
        "```",
        "",
        "## Environment",
        "",
        *[f"- {key}: `{value}`" for key, value in environment.items()],
        "",
        "## Per-seed results",
        "",
        "| Seed | Episodes | Env steps | Wall time (s) | Stopped early | "
        "Greedy eval (mean±std) | Stochastic eval (mean±std) |",
        "|---|---|---|---|---|---|---|",
        *[
            f"| {r['seed']} | {r['total_episodes']} | {r['total_env_steps']} | "
            f"{r['wall_time_s']} | {'yes' if r['stopped_early'] else 'no'} | "
            f"{r['greedy_return_mean']:.1f} ± {r['greedy_return_std']:.1f} | "
            f"{r['stochastic_return_mean']:.1f} ± {r['stochastic_return_std']:.1f} |"
            for r in records
        ],
        "",
        "## Aggregate (greedy eval means across seeds)",
        "",
        f"- mean: {statistics.mean(greedy_means):.1f}",
        f"- median: {statistics.median(greedy_means):.1f}",
        f"- min / max: {min(greedy_means):.1f} / {max(greedy_means):.1f}",
        "",
        "Small-sample caveat: these are per-seed point estimates over "
        f"{len(records)} seeds; the rliable aggregate-metrics protocol "
        "(IQM, bootstrap CIs) arrives with the M4 evaluation engine.",
        "",
        "Reproducibility scope: these numbers are expected to reproduce only "
        "under the pinned software versions recorded above, on CPU, on "
        "comparable hardware; no broader reproducibility claim is intended.",
        "",
        "## Reproduce",
        "",
        "```sh",
        f"uv run python benchmarks/m1_reinforce_cartpole.py --seeds "
        f"{' '.join(str(r['seed']) for r in records)}",
        "```",
        "",
    ]
    report = out_dir / "m1_reinforce_cartpole.md"
    report.write_text("\n".join(lines))
    return report


def main() -> None:
    """Run the validation experiment end to end."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--out-dir", type=Path, default=Path("benchmarks/results"))
    args = parser.parse_args()

    records = [run_seed(seed) for seed in args.seeds]
    report = write_report(records, ReinforceConfig(), args.out_dir)
    logger.info("report written to %s", report)


if __name__ == "__main__":
    main()
