"""Tests for M4A: run identity, run-directory artifacts, and the CLI."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import torch
from hydra import compose, initialize
from omegaconf import OmegaConf

from rlcore.agents.ppo.train import PpoConfig
from rlcore.agents.reinforce.train import ReinforceConfig
from rlcore.agents.reinforce.train import train as reinforce_train
from rlcore.cli import TrainRunConfig, evaluate_run
from rlcore.experiments import load_final_model, new_run_id

if TYPE_CHECKING:
    from pathlib import Path


def tiny_config(seed: int = 3) -> ReinforceConfig:
    return ReinforceConfig(
        seed=seed,
        updates=2,
        episodes_per_update=2,
        eval_every=1,
        eval_episodes=2,
        stop_return=None,
    )


class TestRunIdentity:
    def test_run_id_contains_identity_fields(self) -> None:
        run_id = new_run_id("ppo", "CartPole-v1", 7)
        assert "ppo" in run_id
        assert "cartpole-v1" in run_id
        assert "-s7-" in run_id

    def test_run_ids_are_unique(self) -> None:
        ids = {new_run_id("ppo", "CartPole-v1", 0) for _ in range(50)}
        assert len(ids) == 50


class TestRunDirectory:
    def test_full_run_directory_layout(self, tmp_path: Path) -> None:
        result = reinforce_train(tiny_config(), out_dir=tmp_path)

        record = json.loads((tmp_path / "run.json").read_text())
        assert record["run_id"] == result.run_id
        assert record["algo"] == "reinforce"
        assert record["env_id"] == "CartPole-v1"
        assert record["seed"] == 3
        assert {"torch", "numpy", "hydra", "device", "commit", "git_dirty"} <= set(
            record["metadata"]
        )

        state_dict = load_final_model(tmp_path / "final_model.pt", expected_algo="reinforce")
        for key, value in result.policy.state_dict().items():
            assert torch.equal(state_dict[key], value)

        summary = (tmp_path / "summary.md").read_text()
        assert result.run_id in summary
        assert "final greedy evaluation" in summary

        plot = tmp_path / "plots" / "learning_curve.png"
        assert plot.exists()
        assert plot.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"

        payload = json.loads((tmp_path / "result.json").read_text())
        assert payload["run_id"] == result.run_id

    def test_final_model_rejects_wrong_algo(self, tmp_path: Path) -> None:
        reinforce_train(tiny_config(), out_dir=tmp_path)
        try:
            load_final_model(tmp_path / "final_model.pt", expected_algo="ppo")
        except ValueError as error:
            assert "reinforce" in str(error)
        else:
            raise AssertionError("expected ValueError for algo mismatch")


class TestTrainCli:
    def test_compose_dispatches_algo_group(self) -> None:
        with initialize(version_base="1.3"):
            cfg = compose(config_name="train_run", overrides=["algo=ppo", "algo.total_updates=1"])
            resolved = OmegaConf.to_object(cfg)
        assert isinstance(resolved, TrainRunConfig)
        assert isinstance(resolved.algo, PpoConfig)
        assert resolved.algo.total_updates == 1

    def test_compose_reinforce_with_overrides(self) -> None:
        with initialize(version_base="1.3"):
            cfg = compose(config_name="train_run", overrides=["algo=reinforce", "algo.lr=0.01"])
            resolved = OmegaConf.to_object(cfg)
        assert isinstance(resolved, TrainRunConfig)
        assert isinstance(resolved.algo, ReinforceConfig)
        assert resolved.algo.lr == 0.01


class TestEvaluateCli:
    def test_evaluate_run_from_directory(self, tmp_path: Path) -> None:
        reinforce_train(tiny_config(), out_dir=tmp_path)
        stats = evaluate_run(tmp_path, episodes=2, deterministic=True, eval_seed=5)
        assert len(stats.episode_returns) == 2

        entries = (tmp_path / "evaluations.jsonl").read_text().strip().splitlines()
        record = json.loads(entries[-1])
        assert record["episodes"] == 2
        assert record["deterministic"] is True
        assert record["mean_return"] == stats.mean_return

    def test_evaluate_is_seed_deterministic(self, tmp_path: Path) -> None:
        reinforce_train(tiny_config(), out_dir=tmp_path)
        first = evaluate_run(tmp_path, episodes=3, deterministic=True, eval_seed=9)
        second = evaluate_run(tmp_path, episodes=3, deterministic=True, eval_seed=9)
        assert first.episode_returns == second.episode_returns

    def test_stochastic_evaluation(self, tmp_path: Path) -> None:
        reinforce_train(tiny_config(), out_dir=tmp_path)
        stats = evaluate_run(tmp_path, episodes=2, deterministic=False, eval_seed=9)
        assert len(stats.episode_returns) == 2
