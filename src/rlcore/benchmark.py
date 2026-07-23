"""Manifest-driven benchmark execution across algorithms, envs, and seeds.

Design goals (ROADMAP M7): fixed budgets (early stopping is disabled by
the manifest schema itself — fair comparison requires equal budgets),
independent evaluation seeds via the platform's standard protocol, one
subprocess per run (isolation + real wall-clock timeouts), failed-run and
timeout recording as first-class results, and resumability (finished runs
— including failed ones — are skipped unless ``force``).

Outputs under ``out_dir``:

```
manifest.json          # the manifest as executed
environment.json       # hardware/software metadata
results.jsonl          # one status record per run (ok/failed/timeout/cached)
runs/<run_key>/        # each run's standard rlcore run directory
```

Aggregation and reporting live in :mod:`rlcore.benchreport`.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rlcore.experiments import run_metadata

logger = logging.getLogger(__name__)

KNOWN_ALGOS = ("reinforce", "ppo", "dqn", "sac")

_CHILD_SNIPPET = """
import json, sys
from pathlib import Path
algo, overrides, out_dir = sys.argv[1], json.loads(sys.argv[2]), Path(sys.argv[3])
if algo == "reinforce":
    from rlcore.agents.reinforce.train import ReinforceConfig as Config, train
elif algo == "ppo":
    from rlcore.agents.ppo.train import PpoConfig as Config, train
elif algo == "dqn":
    from rlcore.agents.dqn.train import DqnConfig as Config, train
elif algo == "sac":
    from rlcore.agents.sac.train import SacConfig as Config, train
else:
    raise SystemExit(f"unknown algo {algo!r}")
train(Config(**overrides), out_dir=out_dir)
"""


@dataclass(frozen=True, slots=True)
class BenchmarkRun:
    """One (algorithm, environment, seed) cell of a benchmark.

    Attributes:
        algo: One of ``reinforce``/``ppo``/``dqn``/``sac``.
        env_id: Gymnasium env id (must be action-space compatible with the
            algorithm; the manifest lists explicit pairs rather than a
            cross product for exactly this reason).
        seed: Root seed.
        overrides: Config-field overrides (typically the fixed budget).
        timeout_s: Wall-clock limit for the run's subprocess.
    """

    algo: str
    env_id: str
    seed: int
    overrides: dict[str, Any] = field(default_factory=dict)
    timeout_s: float = 3600.0

    @property
    def run_key(self) -> str:
        """Stable directory-safe identifier for this cell."""
        env_slug = self.env_id.lower().replace("/", "-")
        return f"{self.algo}-{env_slug}-s{self.seed}"


@dataclass(frozen=True, slots=True)
class BenchmarkManifest:
    """A named list of benchmark runs with fixed budgets.

    Attributes:
        name: Benchmark identifier (used in reports).
        runs: The cells to execute.
    """

    name: str
    runs: list[BenchmarkRun]


def load_manifest(path: Path) -> BenchmarkManifest:
    """Load and validate a manifest from JSON.

    Schema: ``{"name": str, "runs": [{"algo", "env_id", "seed",
    "overrides"?, "timeout_s"?}, ...]}``. Validation enforces known
    algorithms and refuses overrides that re-enable early stopping
    (``stop_return`` must be absent or null — fixed budgets are the point).

    Raises:
        ValueError: On schema violations, with the offending entry named.
    """
    data = json.loads(path.read_text())
    if not isinstance(data, dict) or "name" not in data or "runs" not in data:
        raise ValueError(f"{path}: manifest must be an object with 'name' and 'runs'.")
    runs: list[BenchmarkRun] = []
    for i, entry in enumerate(data["runs"]):
        algo = entry.get("algo")
        if algo not in KNOWN_ALGOS:
            raise ValueError(f"{path} runs[{i}]: unknown algo {algo!r}; expected {KNOWN_ALGOS}.")
        overrides = dict(entry.get("overrides", {}))
        if overrides.get("stop_return") is not None and "stop_return" in overrides:
            raise ValueError(
                f"{path} runs[{i}]: stop_return must be null in benchmarks — fixed "
                "budgets are required for fair comparison."
            )
        overrides["stop_return"] = None
        overrides.setdefault("env_id", entry["env_id"])
        overrides.setdefault("seed", int(entry["seed"]))
        runs.append(
            BenchmarkRun(
                algo=algo,
                env_id=str(entry["env_id"]),
                seed=int(entry["seed"]),
                overrides=overrides,
                timeout_s=float(entry.get("timeout_s", 3600.0)),
            )
        )
    if not runs:
        raise ValueError(f"{path}: manifest contains no runs.")
    return BenchmarkManifest(name=str(data["name"]), runs=runs)


def execute_run(run: BenchmarkRun, run_dir: Path) -> dict[str, Any]:
    """Execute one benchmark cell in a subprocess; never raises on failure.

    Returns:
        A status record: ``status`` is ``ok``, ``failed`` (nonzero exit,
        stderr tail included), or ``timeout`` (killed at ``timeout_s``).
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    record: dict[str, Any] = {
        "run_key": run.run_key,
        "algo": run.algo,
        "env_id": run.env_id,
        "seed": run.seed,
        "timeout_s": run.timeout_s,
    }
    try:
        argv = [sys.executable, "-c", _CHILD_SNIPPET, run.algo, json.dumps(run.overrides)]
        completed = subprocess.run(
            [*argv, str(run_dir)],
            capture_output=True,
            text=True,
            timeout=run.timeout_s,
        )
        record["wall_time_s"] = round(time.perf_counter() - started, 1)
        if completed.returncode == 0:
            record["status"] = "ok"
        else:
            record["status"] = "failed"
            record["error"] = completed.stderr.strip()[-2000:]
    except subprocess.TimeoutExpired:
        record["wall_time_s"] = round(time.perf_counter() - started, 1)
        record["status"] = "timeout"
    (run_dir / "status.json").write_text(json.dumps(record, indent=2))
    return record


def run_benchmark(
    manifest: BenchmarkManifest, out_dir: Path, *, force: bool = False
) -> list[dict[str, Any]]:
    """Execute a manifest, resumably; return all status records.

    Cells whose ``status.json`` already exists are skipped (recorded as
    their previous status with ``"cached": true``) unless ``force`` — this
    makes interrupted benchmarks restartable without redoing finished work,
    while still surfacing previous failures honestly.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(
        json.dumps(
            {
                "name": manifest.name,
                "runs": [
                    {
                        "algo": r.algo,
                        "env_id": r.env_id,
                        "seed": r.seed,
                        "overrides": r.overrides,
                        "timeout_s": r.timeout_s,
                    }
                    for r in manifest.runs
                ],
            },
            indent=2,
        )
    )
    (out_dir / "environment.json").write_text(json.dumps(run_metadata(), indent=2))

    records: list[dict[str, Any]] = []
    for run in manifest.runs:
        run_dir = out_dir / "runs" / run.run_key
        status_path = run_dir / "status.json"
        if status_path.exists() and not force:
            record = json.loads(status_path.read_text())
            record["cached"] = True
            logger.info("%s: cached (%s)", run.run_key, record["status"])
        else:
            logger.info("%s: running (timeout %.0fs)", run.run_key, run.timeout_s)
            record = execute_run(run, run_dir)
            logger.info("%s: %s in %.0fs", run.run_key, record["status"], record["wall_time_s"])
        records.append(record)

    with (out_dir / "results.jsonl").open("w") as stream:
        for record in records:
            stream.write(json.dumps(record) + "\n")
    return records
