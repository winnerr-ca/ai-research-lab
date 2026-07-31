"""Run-directory discovery, loading, and comparison for ``rlcore-lab``.

Strictly read-only over run directories written by the trainers
(``run.json``, ``result.json``, ``metrics.jsonl``). Comparison uses the
platform's standard aggregate statistics (:func:`rlcore.stats.summarize`)
with a fixed bootstrap seed so the same selection always reports the
same interval, and it states its own caveats (small samples, unequal
budgets, repeated seeds) instead of hiding them. Runs that cannot
contribute a score are listed as excluded with a reason — never silently
dropped.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from rlcore.stats import summarize

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

#: Fixed bootstrap seed: identical run selections yield identical intervals.
_COMPARE_RNG_SEED = 0

#: Below this many scores per group, CIs are labeled as indicators only.
_SMALL_SAMPLE_N = 5

#: ``result.json`` keys surfaced in run summaries when present.
_RESULT_KEYS = (
    "final_eval_return_mean",
    "final_eval_return_std",
    "total_env_steps",
    "wall_time_s",
)


def resolve_run_dir(root: Path, rel: str) -> Path:
    """Resolve a client-supplied relative path to a run directory under ``root``.

    Args:
        root: The runs root the server was started with.
        rel: Relative path received from the dashboard client.

    Returns:
        The resolved run directory.

    Raises:
        ValueError: If the path escapes ``root`` or holds no ``run.json``
            (the dashboard must never read arbitrary filesystem paths).
    """
    candidate = (root / rel).resolve()
    if not candidate.is_relative_to(root.resolve()):
        raise ValueError(f"run path {rel!r} escapes the runs root.")
    if not (candidate / "run.json").is_file():
        raise ValueError(f"{rel!r} is not a run directory (no run.json).")
    return candidate


def _read_json(path: Path) -> dict[str, Any] | None:
    """Read a JSON object, returning ``None`` on missing or malformed files."""
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _summarize_run(root: Path, run_dir: Path, record: dict[str, Any]) -> dict[str, Any]:
    """Build one run-summary dict from an already-loaded ``run.json`` record."""
    summary: dict[str, Any] = {
        "dir": run_dir.relative_to(root).as_posix(),
        "run_id": record.get("run_id"),
        "algo": record.get("algo"),
        "env_id": record.get("env_id"),
        "seed": record.get("seed"),
        "created_at": record.get("created_at"),
        "has_metrics": (run_dir / "metrics.jsonl").is_file(),
        "has_result": False,
    }
    result = _read_json(run_dir / "result.json")
    if result is not None:
        summary["has_result"] = True
        for key in _RESULT_KEYS:
            if key in result:
                summary[key] = result[key]
    return summary


def discover_runs(root: Path) -> list[dict[str, Any]]:
    """Find every run directory (any directory holding ``run.json``) under ``root``.

    Returns:
        Run summaries sorted newest-first by ``created_at`` (ISO-8601
        strings sort chronologically). Unreadable ``run.json`` files are
        skipped with a log line rather than failing the whole listing.
    """
    if not root.is_dir():
        return []
    summaries: list[dict[str, Any]] = []
    for record_path in sorted(root.rglob("run.json")):
        record = _read_json(record_path)
        if record is None:
            logger.warning("skipping unreadable run record: %s", record_path)
            continue
        summaries.append(_summarize_run(root, record_path.parent, record))
    summaries.sort(key=lambda s: str(s.get("created_at") or ""), reverse=True)
    return summaries


def load_metrics(run_dir: Path) -> list[dict[str, Any]]:
    """Load a run's ``metrics.jsonl`` history (empty list if absent).

    Malformed lines are skipped with a log line; the trainers write one
    JSON object per line, so anything else is a truncated write.
    """
    path = run_dir / "metrics.jsonl"
    if not path.is_file():
        return []
    history: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            logger.warning("skipping malformed metrics line %s:%d", path, line_no)
            continue
        if isinstance(entry, dict):
            history.append(entry)
    return history


def _group_caveats(group: dict[str, Any]) -> list[str]:
    """Honesty caveats for one (algo, env) comparison group."""
    caveats: list[str] = []
    runs: list[dict[str, Any]] = group["runs"]
    label = f"{group['algo']} on {group['env_id']}"
    if len(runs) < _SMALL_SAMPLE_N:
        caveats.append(
            f"{label}: n={len(runs)} — bootstrap CIs at this sample size are "
            "uncertainty indicators, not significance tests."
        )
    budgets = {run.get("total_env_steps") for run in runs}
    if len(budgets) > 1:
        caveats.append(
            f"{label}: unequal budgets across runs ({sorted(budgets, key=str)}) — "
            "this is not a fair comparison; fix budgets to compare."
        )
    seeds = [run.get("seed") for run in runs]
    if len(seeds) != len(set(seeds)):
        caveats.append(f"{label}: repeated seeds — these runs are not independent samples.")
    return caveats


def compare_runs(root: Path, rels: list[str]) -> dict[str, Any]:
    """Compare selected runs, grouped by (algorithm, environment).

    Scores are each run's recorded ``final_eval_return_mean``; groups are
    aggregated with :func:`rlcore.stats.summarize` under a fixed bootstrap
    seed. Runs without a usable score are reported under ``excluded``
    with a reason.

    Args:
        root: The runs root.
        rels: Relative run-directory paths (validated against ``root``).

    Returns:
        ``{"groups": [...], "excluded": [...], "caveats": [...]}``.

    Raises:
        ValueError: If any path fails :func:`resolve_run_dir` validation.
    """
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    excluded: list[dict[str, Any]] = []
    for rel in rels:
        run_dir = resolve_run_dir(root, rel)
        record = _read_json(run_dir / "run.json")
        if record is None:
            excluded.append({"dir": rel, "reason": "unreadable run.json"})
            continue
        summary = _summarize_run(root, run_dir, record)
        if not isinstance(summary.get("final_eval_return_mean"), int | float):
            excluded.append(
                {"dir": rel, "reason": "no recorded final_eval_return_mean in result.json"}
            )
            continue
        key = (str(summary.get("algo")), str(summary.get("env_id")))
        grouped.setdefault(key, []).append(summary)

    groups: list[dict[str, Any]] = []
    caveats: list[str] = []
    for (algo, env_id), runs in sorted(grouped.items()):
        scores = np.asarray([float(run["final_eval_return_mean"]) for run in runs])
        rng = np.random.default_rng(_COMPARE_RNG_SEED)
        group: dict[str, Any] = {
            "algo": algo,
            "env_id": env_id,
            "runs": runs,
            "stats": summarize(scores, rng=rng),
        }
        caveats.extend(_group_caveats(group))
        groups.append(group)
    if len(groups) > 1:
        caveats.append(
            "Cross-group numbers are shown side by side, not ranked: budgets are "
            "per-algorithm as declared, and no superiority claim is made for you."
        )
    return {"groups": groups, "excluded": excluded, "caveats": caveats}
