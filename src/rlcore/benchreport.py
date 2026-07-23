"""Benchmark aggregation, statistics, plots, and Markdown reporting.

Consumes a benchmark directory produced by :mod:`rlcore.benchmark` and
writes ``aggregate.json`` (machine-readable statistics), sample-efficiency
and performance-profile plots, and ``report.md``. Failed and timed-out
runs are first-class rows in the report — hiding them would misrepresent
the benchmark.

Cross-algorithm sample-efficiency curves interpolate each seed's periodic
evaluations onto a common env-step grid (algorithms evaluate on different
cadences); the interpolation is linear and the caveat is stated in the
report.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from rlcore.benchmark import KNOWN_ALGOS
from rlcore.stats import probability_of_improvement, summarize

logger = logging.getLogger(__name__)

# Validated categorical palette slots (see rlcore.reporting), fixed order:
# color follows the algorithm identity across every plot in a report.
_ALGO_COLORS = dict(zip(KNOWN_ALGOS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"], strict=True))
_INK = "#3d3d3a"
_MUTED = "#6f6e66"


def _load_runs(out_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load status records and, for successful runs, their results/metrics."""
    records = [
        json.loads(line) for line in (out_dir / "results.jsonl").read_text().strip().splitlines()
    ]
    runs: dict[str, Any] = {}
    for record in records:
        if record["status"] != "ok":
            continue
        run_dir = out_dir / "runs" / record["run_key"]
        result = json.loads((run_dir / "result.json").read_text())
        history = [
            json.loads(line)
            for line in (run_dir / "metrics.jsonl").read_text().strip().splitlines()
        ]
        runs[record["run_key"]] = {"record": record, "result": result, "history": history}
    return records, runs


def aggregate(out_dir: Path) -> dict[str, Any]:
    """Compute per-(algo, env) statistics and pairwise comparisons.

    Writes and returns ``aggregate.json``: per-group final-score summaries
    (mean/median/IQM with bootstrap CI), per-seed curves, per-env pairwise
    probability of improvement, and the run status census.
    """
    records, runs = _load_runs(out_dir)
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for payload in runs.values():
        record = payload["record"]
        groups.setdefault((record["algo"], record["env_id"]), []).append(payload)

    rng = np.random.default_rng(0)  # fixed: aggregation is deterministic
    group_stats: dict[str, Any] = {}
    for (algo, env_id), payloads in sorted(groups.items()):
        scores = np.array(
            [p["result"]["final_eval_return_mean"] for p in payloads], dtype=np.float64
        )
        curves = [
            {
                "seed": p["record"]["seed"],
                "env_steps": [h["env_steps"] for h in p["history"] if "eval_return_mean" in h],
                "eval_return_mean": [
                    h["eval_return_mean"] for h in p["history"] if "eval_return_mean" in h
                ],
            }
            for p in payloads
        ]
        group_stats[f"{algo}|{env_id}"] = {
            "algo": algo,
            "env_id": env_id,
            "seeds": sorted(p["record"]["seed"] for p in payloads),
            "final_scores": scores.tolist(),
            "summary": summarize(scores, rng=rng),
            "curves": curves,
        }

    comparisons: dict[str, Any] = {}
    envs = sorted({env for _, env in groups})
    for env_id in envs:
        algos = sorted(algo for algo, env in groups if env == env_id)
        matrix: dict[str, float] = {}
        for algo_x in algos:
            for algo_y in algos:
                if algo_x == algo_y:
                    continue
                x = np.array(group_stats[f"{algo_x}|{env_id}"]["final_scores"])
                y = np.array(group_stats[f"{algo_y}|{env_id}"]["final_scores"])
                matrix[f"P({algo_x} > {algo_y})"] = round(probability_of_improvement(x, y), 3)
        comparisons[env_id] = matrix

    result = {
        "status_census": {
            status: sum(1 for r in records if r["status"] == status)
            for status in sorted({r["status"] for r in records})
        },
        "records": records,
        "groups": group_stats,
        "comparisons": comparisons,
    }
    (out_dir / "aggregate.json").write_text(json.dumps(result, indent=2))
    return result


def _plot_curves(out_dir: Path, agg: dict[str, Any], env_id: str) -> Path | None:
    """Sample-efficiency plot: per-algo mean curve with per-seed traces."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    groups = [g for g in agg["groups"].values() if g["env_id"] == env_id]
    if not all(any(c["env_steps"] for c in g["curves"]) for g in groups):
        return None
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=150)
    for group in groups:
        color = _ALGO_COLORS.get(group["algo"], _INK)
        max_step = max(max(c["env_steps"]) for c in group["curves"] if c["env_steps"])
        grid = np.linspace(0, max_step, 60)
        interpolated = []
        for curve in group["curves"]:
            if not curve["env_steps"]:
                continue
            ax.plot(
                curve["env_steps"],
                curve["eval_return_mean"],
                color=color,
                alpha=0.25,
                linewidth=0.9,
            )
            interpolated.append(np.interp(grid, curve["env_steps"], curve["eval_return_mean"]))
        if interpolated:
            ax.plot(
                grid,
                np.mean(interpolated, axis=0),
                color=color,
                linewidth=1.9,
                label=f"{group['algo']} (mean of {len(interpolated)} seeds)",
            )
    ax.set_title(f"Sample efficiency — {env_id}", color=_INK, fontsize=11)
    ax.set_xlabel("environment steps", color=_MUTED, fontsize=9)
    ax.set_ylabel("evaluation return", color=_MUTED, fontsize=9)
    ax.tick_params(colors=_MUTED, labelsize=8)
    ax.grid(color="#e6e5df", linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    legend = ax.legend(loc="lower right", fontsize=8, frameon=False)
    for text in legend.get_texts():
        text.set_color(_INK)
    path = out_dir / "plots" / f"curves_{env_id.lower().replace('/', '-')}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def _plot_profiles(out_dir: Path, agg: dict[str, Any], env_id: str) -> Path | None:
    """Performance-profile plot: fraction of runs at or above each score."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    from rlcore.stats import performance_profile

    groups = [g for g in agg["groups"].values() if g["env_id"] == env_id]
    if not groups:
        return None
    all_scores = np.concatenate([np.array(g["final_scores"]) for g in groups])
    thresholds = np.linspace(all_scores.min(), all_scores.max(), 41)
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=150)
    for group in groups:
        profile = performance_profile(np.array(group["final_scores"]), thresholds)
        ax.step(
            thresholds,
            profile,
            where="post",
            color=_ALGO_COLORS.get(group["algo"], _INK),
            linewidth=1.9,
            label=group["algo"],
        )
    ax.set_title(f"Performance profile — {env_id}", color=_INK, fontsize=11)
    ax.set_xlabel("final evaluation return threshold", color=_MUTED, fontsize=9)
    ax.set_ylabel("fraction of runs ≥ threshold", color=_MUTED, fontsize=9)
    ax.tick_params(colors=_MUTED, labelsize=8)
    ax.grid(color="#e6e5df", linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    legend = ax.legend(loc="lower left", fontsize=8, frameon=False)
    for text in legend.get_texts():
        text.set_color(_INK)
    path = out_dir / "plots" / f"profile_{env_id.lower().replace('/', '-')}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def write_report(out_dir: Path, name: str) -> Path:
    """Aggregate and render the full benchmark report; return its path."""
    agg = aggregate(out_dir)
    environment = json.loads((out_dir / "environment.json").read_text())
    envs = sorted({g["env_id"] for g in agg["groups"].values()})

    lines = [
        f"# Benchmark report: {name}",
        "",
        "**Scope.** Fixed-budget engineering benchmark on the environments and",
        "budgets listed in `manifest.json`. Statistics follow the protocol of",
        "Agarwal et al. (NeurIPS 2021) computed by `rlcore.stats` (native,",
        "tested implementation). Small-sample caveat: with these seed counts,",
        "bootstrap intervals are rough uncertainty indicators. Environments",
        "differ per algorithm family (discrete vs continuous action spaces),",
        "so cross-*environment* rows are not comparable to each other.",
        "",
        "## Run status census",
        "",
        "| Status | Count |",
        "|---|---|",
        *[f"| {k} | {v} |" for k, v in agg["status_census"].items()],
        "",
    ]
    failed = [r for r in agg["records"] if r["status"] != "ok"]
    if failed:
        lines += [
            "### Non-successful runs (reported, never hidden)",
            "",
            "| Run | Status | Wall time (s) | Error (tail) |",
            "|---|---|---|---|",
            *[
                f"| {r['run_key']} | {r['status']} | {r.get('wall_time_s', '—')} | "
                f"{(r.get('error', '') or '—').splitlines()[-1][:120]} |"
                for r in failed
            ],
            "",
        ]

    lines += ["## Environment", ""]
    lines += [f"- {key}: `{value}`" for key, value in environment.items()]
    lines += [""]

    for env_id in envs:
        lines += [f"## {env_id}", ""]
        lines += [
            "| Algorithm | Seeds | Mean | Median | IQM [95% bootstrap CI] | Min | Max |",
            "|---|---|---|---|---|---|---|",
        ]
        for group in sorted(
            (g for g in agg["groups"].values() if g["env_id"] == env_id),
            key=lambda g: str(g["algo"]),
        ):
            s = group["summary"]
            lines.append(
                f"| {group['algo']} | {len(group['seeds'])} | {s['mean']:.1f} | "
                f"{s['median']:.1f} | {s['iqm']:.1f} "
                f"[{s['iqm_ci_low']:.1f}, {s['iqm_ci_high']:.1f}] | "
                f"{s['min']:.1f} | {s['max']:.1f} |"
            )
        lines.append("")
        if agg["comparisons"].get(env_id):
            lines += [
                "Pairwise probability of improvement (descriptive; no",
                "significance claim at these sample sizes):",
                "",
                *[f"- {k}: {v}" for k, v in agg["comparisons"][env_id].items()],
                "",
            ]
        curve_path = _plot_curves(out_dir, agg, env_id)
        if curve_path is not None:
            lines.append(f"![sample efficiency]({curve_path.relative_to(out_dir)})")
        profile_path = _plot_profiles(out_dir, agg, env_id)
        if profile_path is not None:
            lines.append(f"![performance profile]({profile_path.relative_to(out_dir)})")
        lines.append("")

    report = out_dir / "report.md"
    report.write_text("\n".join(lines))
    logger.info("benchmark report written to %s", report)
    return report
