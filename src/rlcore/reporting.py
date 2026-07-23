"""Human-readable run summaries and learning-curve plots (local, offline).

Matplotlib is imported lazily so `import rlcore` stays light and headless
environments work without display configuration (the Agg backend is forced).
Plot styling follows the platform's chart conventions: one axis, at most two
series with a legend, thin lines, recessive grid, colorblind-validated
series colors, neutral ink for text.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from rlcore.evaluation import EvalStats

# Categorical slots 1-2 of the platform's validated palette (light surface):
# an adjacent pair that passes CVD-separation and contrast checks.
TRAIN_COLOR = "#2a78d6"
EVAL_COLOR = "#eb6834"
INK = "#3d3d3a"
MUTED_INK = "#6f6e66"


def _series(
    history: list[dict[str, float]], x_key: str, y_key: str
) -> tuple[list[float], list[float]]:
    """Extract an (x, y) series from history entries, dropping missing/NaN."""
    xs: list[float] = []
    ys: list[float] = []
    for entry in history:
        y = entry.get(y_key)
        if y is None or math.isnan(y):
            continue
        xs.append(entry[x_key])
        ys.append(y)
    return xs, ys


def write_learning_curve(out_dir: Path, history: list[dict[str, float]], *, title: str) -> Path:
    """Render ``plots/learning_curve.png``: return vs. environment steps.

    Two series at most: per-iteration training return (line) and periodic
    greedy evaluation return (marked line — evaluations are sparse). Entries
    with missing or NaN returns (e.g. PPO iterations with no completed
    episode) are dropped rather than interpolated.
    """
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=150)
    train_x, train_y = _series(history, "env_steps", "train_return_mean")
    eval_x, eval_y = _series(history, "env_steps", "eval_return_mean")
    if train_x:
        ax.plot(train_x, train_y, color=TRAIN_COLOR, linewidth=1.8, label="train return (mean)")
    if eval_x:
        ax.plot(
            eval_x,
            eval_y,
            color=EVAL_COLOR,
            linewidth=1.8,
            marker="o",
            markersize=4.5,
            label="eval return (greedy mean)",
        )

    ax.set_title(title, color=INK, fontsize=11)
    ax.set_xlabel("environment steps", color=MUTED_INK, fontsize=9)
    ax.set_ylabel("episode return", color=MUTED_INK, fontsize=9)
    ax.tick_params(colors=MUTED_INK, labelsize=8)
    ax.grid(color="#e6e5df", linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#c9c8c0")
    if train_x or eval_x:
        legend = ax.legend(loc="lower right", fontsize=8, frameon=False)
        for text in legend.get_texts():
            text.set_color(INK)

    path = plots_dir / "learning_curve.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def write_summary(
    out_dir: Path,
    *,
    run_id: str,
    algo: str,
    env_id: str,
    seed: int,
    history: list[dict[str, float]],
    final_eval: EvalStats,
    total_env_steps: int,
    stopped_early: bool,
    wall_time_s: float,
) -> Path:
    """Write ``summary.md`` and the learning-curve plot for one run."""
    plot_path = write_learning_curve(out_dir, history, title=f"{algo} on {env_id} (seed {seed})")
    lines = [
        f"# Run summary: `{run_id}`",
        "",
        f"- algorithm: **{algo}**",
        f"- environment: **{env_id}**",
        f"- seed: {seed}",
        f"- final greedy evaluation: **{final_eval.mean_return:.1f} ± "
        f"{final_eval.std_return:.1f}** over {len(final_eval.episode_returns)} episodes "
        f"(min {final_eval.min_return:.1f}, max {final_eval.max_return:.1f})",
        f"- environment steps: {total_env_steps}",
        f"- training iterations: {len(history)}",
        f"- stopped early: {'yes' if stopped_early else 'no'}",
        f"- wall time: {wall_time_s:.1f}s",
        "",
        f"![learning curve]({plot_path.relative_to(out_dir)})",
        "",
        "Full identity and environment metadata: `run.json`. Per-iteration "
        "metrics: `metrics.jsonl`. Resolved configuration: `config.yaml`.",
        "",
    ]
    path = out_dir / "summary.md"
    path.write_text("\n".join(lines))
    return path
