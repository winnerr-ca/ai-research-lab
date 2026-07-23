"""Tests for the benchmark runner: manifest validation, failure recording."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from rlcore.benchmark import BenchmarkRun, load_manifest, run_benchmark
from rlcore.benchreport import write_report

if TYPE_CHECKING:
    from pathlib import Path


def write_manifest(path: Path, runs: list[dict[str, object]]) -> Path:
    path.write_text(json.dumps({"name": "test-bench", "runs": runs}))
    return path


TINY_REINFORCE = {
    "updates": 2,
    "episodes_per_update": 2,
    "eval_every": 1,
    "eval_episodes": 2,
}


class TestManifest:
    def test_loads_and_normalizes(self, tmp_path: Path) -> None:
        manifest = load_manifest(
            write_manifest(
                tmp_path / "m.json",
                [{"algo": "reinforce", "env_id": "CartPole-v1", "seed": 3}],
            )
        )
        assert manifest.name == "test-bench"
        run = manifest.runs[0]
        assert run.run_key == "reinforce-cartpole-v1-s3"
        assert run.overrides["stop_return"] is None  # fixed budgets enforced
        assert run.overrides["seed"] == 3

    def test_rejects_unknown_algo(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="unknown algo"):
            load_manifest(
                write_manifest(tmp_path / "m.json", [{"algo": "a3c", "env_id": "X", "seed": 0}])
            )

    def test_rejects_early_stopping_override(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="stop_return"):
            load_manifest(
                write_manifest(
                    tmp_path / "m.json",
                    [
                        {
                            "algo": "ppo",
                            "env_id": "CartPole-v1",
                            "seed": 0,
                            "overrides": {"stop_return": 400.0},
                        }
                    ],
                )
            )

    def test_rejects_empty(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="no runs"):
            load_manifest(write_manifest(tmp_path / "m.json", []))


class TestExecution:
    def test_ok_failed_and_cached(self, tmp_path: Path) -> None:
        manifest = load_manifest(
            write_manifest(
                tmp_path / "m.json",
                [
                    {
                        "algo": "reinforce",
                        "env_id": "CartPole-v1",
                        "seed": 0,
                        "overrides": dict(TINY_REINFORCE),
                        "timeout_s": 300,
                    },
                    {
                        "algo": "reinforce",
                        "env_id": "NoSuchEnv-v0",  # deliberate failure
                        "seed": 0,
                        "overrides": dict(TINY_REINFORCE),
                        "timeout_s": 300,
                    },
                ],
            )
        )
        out_dir = tmp_path / "bench"
        records = run_benchmark(manifest, out_dir)
        by_key = {r["run_key"]: r for r in records}
        assert by_key["reinforce-cartpole-v1-s0"]["status"] == "ok"
        failed = by_key["reinforce-nosuchenv-v0-s0"]
        assert failed["status"] == "failed"
        assert failed["error"]  # stderr tail captured

        # Resumability: a second invocation reuses both outcomes.
        rerun = run_benchmark(manifest, out_dir)
        assert all(r.get("cached") for r in rerun)
        assert {r["status"] for r in rerun} == {"ok", "failed"}

        # Results file lists every run including the failure.
        lines = (out_dir / "results.jsonl").read_text().strip().splitlines()
        assert len(lines) == 2

        # Reporting works with a partial (one-algo, one-failure) benchmark
        # and surfaces the failure.
        report = write_report(out_dir, "test-bench")
        text = report.read_text()
        assert "Non-successful runs" in text
        assert "reinforce-nosuchenv-v0-s0" in text
        assert (out_dir / "aggregate.json").exists()

    def test_timeout_recorded(self, tmp_path: Path) -> None:
        manifest = load_manifest(
            write_manifest(
                tmp_path / "m.json",
                [
                    {
                        "algo": "reinforce",
                        "env_id": "CartPole-v1",
                        "seed": 0,
                        # Budget far beyond the timeout: guaranteed to be killed.
                        "overrides": {"updates": 1_000_000, "eval_every": 0},
                        "timeout_s": 12,
                    }
                ],
            )
        )
        records = run_benchmark(manifest, tmp_path / "bench")
        assert records[0]["status"] == "timeout"
        assert records[0]["wall_time_s"] >= 11


class TestRunKey:
    def test_slugging(self) -> None:
        run = BenchmarkRun(algo="dqn", env_id="ALE/Pong-v5", seed=2)
        assert run.run_key == "dqn-ale-pong-v5-s2"
