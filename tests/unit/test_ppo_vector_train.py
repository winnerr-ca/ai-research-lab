"""Trainer-level tests for vectorized PPO and the device config field."""

from __future__ import annotations

import dataclasses
import json
import math
import subprocess
import sys
from typing import TYPE_CHECKING

import pytest

from rlcore.agents.ppo.train import PpoConfig, train

if TYPE_CHECKING:
    from pathlib import Path


def assert_histories_equal(first: list[dict[str, float]], second: list[dict[str, float]]) -> None:
    assert len(first) == len(second)
    for a, b in zip(first, second, strict=True):
        assert a.keys() == b.keys()
        for key in a:
            if math.isnan(a[key]) and math.isnan(b[key]):
                continue
            assert a[key] == b[key], key


def tiny_config(
    total_updates: int,
    *,
    n_envs: int = 2,
    checkpoint_every: int = 0,
    device: str = "cpu",
) -> PpoConfig:
    return PpoConfig(
        seed=3,
        n_envs=n_envs,
        total_updates=total_updates,
        n_steps=64,
        minibatch_size=32,
        n_epochs=2,
        hidden_sizes=[16],
        eval_every=0,
        stop_return=None,
        checkpoint_every=checkpoint_every,
        device=device,
    )


class TestVectorTraining:
    def test_smoke_and_step_accounting(self) -> None:
        result = train(tiny_config(total_updates=2))
        assert len(result.history) == 2
        # n_steps is the TOTAL batch per update regardless of n_envs.
        assert result.total_env_steps == 2 * 64

    def test_single_and_vector_consume_equal_budgets(self) -> None:
        single = train(tiny_config(total_updates=2, n_envs=1))
        vector = train(tiny_config(total_updates=2, n_envs=4))
        assert single.total_env_steps == vector.total_env_steps

    def test_same_seed_is_deterministic(self) -> None:
        first = train(tiny_config(total_updates=2))
        second = train(tiny_config(total_updates=2))
        assert_histories_equal(first.history, second.history)
        for key, value in first.model.state_dict().items():
            import torch

            assert torch.equal(value, second.model.state_dict()[key]), key

    def test_config_validation(self) -> None:
        with pytest.raises(ValueError, match="n_envs"):
            train(tiny_config(total_updates=1, n_envs=0))
        with pytest.raises(ValueError, match="divisible"):
            config = dataclasses.replace(tiny_config(total_updates=1), n_steps=63)
            train(config)

    def test_vector_resume_equals_uninterrupted(self, tmp_path: Path) -> None:
        continuous = train(tiny_config(total_updates=4))
        train(tiny_config(total_updates=2, checkpoint_every=2), out_dir=tmp_path)
        resumed = train(
            tiny_config(total_updates=4, checkpoint_every=2),
            out_dir=tmp_path,
            resume_from=tmp_path / "checkpoint.pt",
        )
        assert_histories_equal(continuous.history, resumed.history)

    def test_vector_fresh_process_resume(self, tmp_path: Path) -> None:
        continuous = train(tiny_config(total_updates=4))
        train(tiny_config(total_updates=2, checkpoint_every=2), out_dir=tmp_path)
        snippet = (
            "import json, sys, torch\n"
            "from pathlib import Path\n"
            "from rlcore.agents.ppo.train import PpoConfig, train\n"
            "cfg = PpoConfig(**json.loads(sys.argv[1]))\n"
            "result = train(cfg, resume_from=Path(sys.argv[2]))\n"
            "json.dump(result.history, open(sys.argv[3], 'w'))\n"
        )
        out = tmp_path / "fresh-history.json"
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                snippet,
                # checkpoint_every is resume-mutable; 0 avoids needing out_dir.
                json.dumps(dataclasses.asdict(tiny_config(total_updates=4))),
                str(tmp_path / "checkpoint.pt"),
                str(out),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr[-2000:]
        fresh_history = json.loads(out.read_text())
        assert_histories_equal(continuous.history, fresh_history)


class TestDeviceField:
    def test_cpu_device_trains(self) -> None:
        result = train(tiny_config(total_updates=1, device="cpu"))
        assert len(result.history) == 1

    def test_explicit_cpu_matches_default(self) -> None:
        default = train(tiny_config(total_updates=2))
        explicit = train(tiny_config(total_updates=2, device="cpu"))
        assert_histories_equal(default.history, explicit.history)

    def test_cuda_request_falls_back_and_trains(self) -> None:
        import torch

        if torch.cuda.is_available():  # pragma: no cover - GPU CI only
            pytest.skip("CUDA available; fallback path not reachable.")
        result = train(tiny_config(total_updates=1, device="cuda"))
        assert len(result.history) == 1

    def test_unknown_device_rejected(self) -> None:
        with pytest.raises(ValueError, match="device spec"):
            train(tiny_config(total_updates=1, device="tpu"))
