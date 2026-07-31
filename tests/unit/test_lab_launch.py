"""Tests for lab algorithm discovery, launch validation, and real launches."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rlcore.lab.launch import JobManager, algo_fields, discover_algos, validate_launch
from rlcore.lab.replay import record_trajectory

TINY_REINFORCE = {
    "updates": 2,
    "episodes_per_update": 2,
    "eval_every": 1,
    "eval_episodes": 2,
}


class TestDiscovery:
    def test_finds_builtin_algorithms(self) -> None:
        algos = discover_algos()
        assert {"reinforce", "ppo", "dqn", "sac"} <= set(algos)
        assert algos["ppo"].config_cls.__name__ == "PpoConfig"

    def test_fields_have_names_types_defaults(self) -> None:
        fields = algo_fields(discover_algos()["reinforce"])
        by_name = {field["name"]: field for field in fields}
        assert by_name["env_id"]["default"] == "CartPole-v1"
        assert by_name["seed"]["default"] == 0
        assert isinstance(by_name["hidden_sizes"]["default"], list)  # default_factory

    def test_nan_defaults_become_strings(self) -> None:
        # SAC's auto target_entropy sentinel is NaN; browsers reject NaN in
        # JSON, so the field description must fall back to its repr.
        fields = algo_fields(discover_algos()["sac"])
        by_name = {field["name"]: field for field in fields}
        assert by_name["target_entropy"]["default"] == "nan"
        for field in fields:
            json.dumps(field, allow_nan=False)  # strict-JSON clean


class TestValidation:
    def test_unknown_algo_rejected(self) -> None:
        algos = discover_algos()
        with pytest.raises(ValueError, match="unknown algorithm"):
            validate_launch(algos, "a3c", {})
        with pytest.raises(ValueError, match="unknown algorithm"):
            validate_launch(algos, "ppo; rm -rf /", {})

    def test_unknown_override_key_rejected(self) -> None:
        algos = discover_algos()
        with pytest.raises(ValueError, match="unknown override keys"):
            validate_launch(algos, "ppo", {"not_a_field": 1})

    def test_valid_request_returns_spec(self) -> None:
        algos = discover_algos()
        assert validate_launch(algos, "dqn", {"seed": 3}).name == "dqn"


class TestJobManager:
    def test_launch_runs_to_ok_and_writes_run_dir(self, tmp_path: Path) -> None:
        manager = JobManager(tmp_path)
        algos = discover_algos()
        overrides = {"seed": 0, **TINY_REINFORCE}
        job = manager.launch(validate_launch(algos, "reinforce", overrides), overrides)
        final = manager.wait(job["job_id"], timeout_s=300.0)
        assert final["status"] == "ok", final.get("error")
        run_dir = Path(final["run_dir"])
        assert (run_dir / "run.json").is_file()
        assert (run_dir / "result.json").is_file()
        launch_record = json.loads((run_dir / "lab-launch.json").read_text())
        assert launch_record == {"algo": "reinforce", "overrides": overrides}
        assert manager.jobs()[0]["job_id"] == job["job_id"]

        # The finished run replays for the agent view: CartPole states are
        # 4-vectors, rewards accumulate to the episode return, and the
        # payload is strict-JSON clean (browsers reject NaN).
        trajectory = record_trajectory(run_dir)
        assert trajectory["env_id"] == "CartPole-v1"
        assert trajectory["steps"] >= 1
        assert len(trajectory["states"]) == trajectory["steps"] + 1
        assert all(len(state) == 4 for state in trajectory["states"])
        assert sum(trajectory["rewards"]) == trajectory["episode_return"]
        json.dumps(trajectory, allow_nan=False)

    def test_failed_launch_records_stderr(self, tmp_path: Path) -> None:
        manager = JobManager(tmp_path)
        algos = discover_algos()
        overrides = {"env_id": "NoSuchEnv-v0", **TINY_REINFORCE}
        job = manager.launch(validate_launch(algos, "reinforce", overrides), overrides)
        final = manager.wait(job["job_id"], timeout_s=120.0)
        assert final["status"] == "failed"
        assert "NoSuchEnv" in final["error"]

    def test_stop_terminates_running_job(self, tmp_path: Path) -> None:
        manager = JobManager(tmp_path)
        algos = discover_algos()
        # A budget far beyond the test's lifetime, so stop() is what ends it.
        overrides = {"updates": 100_000, "episodes_per_update": 10}
        job = manager.launch(validate_launch(algos, "reinforce", overrides), overrides)
        stopped = manager.stop(job["job_id"])
        assert stopped["status"] in ("running", "stopped")  # terminate may need a beat
        final = manager.wait(job["job_id"], timeout_s=30.0)
        assert final["status"] == "stopped"

    def test_wait_unknown_job_raises(self, tmp_path: Path) -> None:
        with pytest.raises(KeyError):
            JobManager(tmp_path).wait("ghost", timeout_s=1.0)
        with pytest.raises(KeyError):
            JobManager(tmp_path).stop("ghost")
