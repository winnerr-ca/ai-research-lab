"""Tests for the lab data layer: discovery, metrics, comparison, path safety."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from rlcore.lab.data import compare_runs, discover_runs, load_metrics, resolve_run_dir

if TYPE_CHECKING:
    from pathlib import Path


def make_run(
    root: Path,
    name: str,
    *,
    algo: str = "ppo",
    env_id: str = "CartPole-v1",
    seed: int = 0,
    created_at: str = "2026-07-01T00:00:00+00:00",
    final_eval: float | None = 400.0,
    total_env_steps: int = 1000,
    metrics: list[dict[str, Any]] | None = None,
) -> Path:
    run_dir = root / name
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "run_id": name,
                "algo": algo,
                "env_id": env_id,
                "seed": seed,
                "created_at": created_at,
            }
        )
    )
    if final_eval is not None:
        (run_dir / "result.json").write_text(
            json.dumps(
                {
                    "final_eval_return_mean": final_eval,
                    "final_eval_return_std": 1.0,
                    "total_env_steps": total_env_steps,
                }
            )
        )
    if metrics is not None:
        (run_dir / "metrics.jsonl").write_text(
            "\n".join(json.dumps(entry) for entry in metrics) + "\n"
        )
    return run_dir


class TestDiscovery:
    def test_finds_nested_runs_newest_first(self, tmp_path: Path) -> None:
        make_run(tmp_path, "old", created_at="2026-01-01T00:00:00+00:00")
        make_run(tmp_path, "sub/new", created_at="2026-06-01T00:00:00+00:00")
        found = discover_runs(tmp_path)
        assert [run["run_id"] for run in found] == ["sub/new", "old"]
        assert found[0]["dir"] == "sub/new"
        assert found[0]["final_eval_return_mean"] == 400.0
        assert found[0]["has_result"] is True

    def test_missing_root_and_unreadable_record(self, tmp_path: Path) -> None:
        assert discover_runs(tmp_path / "nope") == []
        bad = tmp_path / "bad"
        bad.mkdir()
        (bad / "run.json").write_text("{not json")
        assert discover_runs(tmp_path) == []

    def test_run_without_result_is_listed(self, tmp_path: Path) -> None:
        make_run(tmp_path, "r", final_eval=None)
        (found,) = discover_runs(tmp_path)
        assert found["has_result"] is False
        assert "final_eval_return_mean" not in found


class TestMetrics:
    def test_loads_and_skips_malformed_lines(self, tmp_path: Path) -> None:
        run_dir = make_run(tmp_path, "r", metrics=[{"update": 1.0, "eval_return_mean": 9.0}])
        with (run_dir / "metrics.jsonl").open("a") as stream:
            stream.write("{broken\n")
        history = load_metrics(run_dir)
        assert history == [{"update": 1.0, "eval_return_mean": 9.0}]

    def test_absent_metrics_is_empty(self, tmp_path: Path) -> None:
        run_dir = make_run(tmp_path, "r")
        assert load_metrics(run_dir) == []


class TestPathSafety:
    def test_rejects_escape_and_non_run_dirs(self, tmp_path: Path) -> None:
        root = tmp_path / "runs"
        make_run(root, "ok")
        (tmp_path / "outside").mkdir()
        (tmp_path / "outside" / "run.json").write_text("{}")
        with pytest.raises(ValueError, match="escapes"):
            resolve_run_dir(root, "../outside")
        with pytest.raises(ValueError, match="not a run directory"):
            resolve_run_dir(root, "missing")
        assert resolve_run_dir(root, "ok") == (root / "ok").resolve()


class TestCompare:
    def test_groups_stats_and_exclusions(self, tmp_path: Path) -> None:
        make_run(tmp_path, "a", seed=0, final_eval=100.0)
        make_run(tmp_path, "b", seed=1, final_eval=200.0)
        make_run(tmp_path, "c", seed=0, algo="dqn", final_eval=50.0)
        make_run(tmp_path, "d", seed=2, final_eval=None)  # no result.json
        out = compare_runs(tmp_path, ["a", "b", "c", "d"])
        assert [(g["algo"], g["stats"]["n"]) for g in out["groups"]] == [
            ("dqn", 1.0),
            ("ppo", 2.0),
        ]
        ppo = out["groups"][1]["stats"]
        assert ppo["mean"] == 150.0
        assert ppo["min"] == 100.0 and ppo["max"] == 200.0
        assert out["excluded"] == [
            {"dir": "d", "reason": "no recorded final_eval_return_mean in result.json"}
        ]
        # Small-sample caveats for both groups plus the cross-group note.
        assert any("n=2" in caveat for caveat in out["caveats"])
        assert any("not ranked" in caveat for caveat in out["caveats"])

    def test_deterministic_bootstrap(self, tmp_path: Path) -> None:
        for i, score in enumerate([10.0, 20.0, 30.0]):
            make_run(tmp_path, f"r{i}", seed=i, final_eval=score)
        first = compare_runs(tmp_path, ["r0", "r1", "r2"])
        second = compare_runs(tmp_path, ["r0", "r1", "r2"])
        assert first["groups"][0]["stats"] == second["groups"][0]["stats"]

    def test_flags_unequal_budgets_and_repeated_seeds(self, tmp_path: Path) -> None:
        make_run(tmp_path, "a", seed=0, total_env_steps=1000)
        make_run(tmp_path, "b", seed=0, total_env_steps=2000)
        out = compare_runs(tmp_path, ["a", "b"])
        assert any("unequal budgets" in caveat for caveat in out["caveats"])
        assert any("repeated seeds" in caveat for caveat in out["caveats"])

    def test_invalid_path_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="not a run directory"):
            compare_runs(tmp_path, ["ghost"])
