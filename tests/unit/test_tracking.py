"""Tests for M4C: tracker protocol wiring and optional adapters."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

import pytest

from rlcore.agents.reinforce.train import ReinforceConfig
from rlcore.agents.reinforce.train import train as reinforce_train
from rlcore.tracking import make_tracker

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path


class RecordingTracker:
    """Test double implementing the Tracker protocol."""

    def __init__(self) -> None:
        self.started: list[dict[str, Any]] = []
        self.metrics: list[tuple[int, dict[str, float]]] = []
        self.summaries: list[dict[str, Any]] = []
        self.finished = 0

    def start(
        self, *, run_id: str, algo: str, env_id: str, seed: int, config: Mapping[str, Any]
    ) -> None:
        self.started.append(
            {"run_id": run_id, "algo": algo, "env_id": env_id, "seed": seed, "config": dict(config)}
        )

    def log_metrics(self, step: int, metrics: Mapping[str, float]) -> None:
        self.metrics.append((step, dict(metrics)))

    def log_summary(self, summary: Mapping[str, Any]) -> None:
        self.summaries.append(dict(summary))

    def finish(self) -> None:
        self.finished += 1


def tiny_config() -> ReinforceConfig:
    return ReinforceConfig(
        seed=3,
        updates=2,
        episodes_per_update=2,
        eval_every=1,
        eval_episodes=2,
        stop_return=None,
    )


class TestMakeTracker:
    def test_none_returns_none_without_optional_imports(self) -> None:
        already_loaded = {name for name in ("wandb", "mlflow") if name in sys.modules}
        assert make_tracker("none") is None
        newly_loaded = {
            name for name in ("wandb", "mlflow") if name in sys.modules
        } - already_loaded
        assert not newly_loaded

    def test_unknown_kind_rejected(self) -> None:
        with pytest.raises(ValueError, match="Unknown tracker"):
            make_tracker("tensorboard")


class TestTrainerIntegration:
    def test_trainer_drives_full_lifecycle(self) -> None:
        tracker = RecordingTracker()
        result = reinforce_train(tiny_config(), tracker=tracker)

        assert len(tracker.started) == 1
        assert tracker.started[0]["run_id"] == result.run_id
        assert tracker.started[0]["algo"] == "reinforce"
        assert tracker.started[0]["config"]["updates"] == 2

        assert len(tracker.metrics) == len(result.history)
        assert [step for step, _ in tracker.metrics] == [1, 2]
        assert tracker.metrics[0][1] == result.history[0]

        assert len(tracker.summaries) == 1
        assert tracker.summaries[0]["eval_return_mean"] == result.final_eval.mean_return
        assert tracker.finished == 1

    def test_local_only_training_needs_no_tracker(self) -> None:
        result = reinforce_train(tiny_config(), tracker=None)
        assert len(result.history) == 2


class TestWandbAdapter:
    def test_offline_run_without_credentials(self, tmp_path: Path) -> None:
        pytest.importorskip("wandb")
        from rlcore.tracking import WandbTracker

        tracker = WandbTracker(project="rlcore-test", mode="offline", dir=str(tmp_path))
        tracker.start(
            run_id="test-run", algo="reinforce", env_id="CartPole-v1", seed=0, config={"lr": 0.01}
        )
        tracker.log_metrics(1, {"loss": 0.5})
        tracker.log_summary({"eval_return_mean": 100.0})
        tracker.finish()
        assert list(tmp_path.glob("wandb/offline-run-*")), "offline run directory expected"


class TestMlflowAdapter:
    def test_local_file_store_without_credentials(self, tmp_path: Path) -> None:
        pytest.importorskip("mlflow")
        from rlcore.tracking import MlflowTracker

        tracker = MlflowTracker(
            experiment="rlcore-test", tracking_uri=f"sqlite:///{tmp_path}/mlflow.db"
        )
        tracker.start(
            run_id="test-run", algo="ppo", env_id="CartPole-v1", seed=0, config={"lr": 3e-4}
        )
        tracker.log_metrics(1, {"loss": 0.5, "maybe_nan": float("nan")})
        tracker.log_summary({"eval_return_mean": 100.0, "stopped_early": False})
        tracker.finish()
        assert (tmp_path / "mlflow.db").exists()
