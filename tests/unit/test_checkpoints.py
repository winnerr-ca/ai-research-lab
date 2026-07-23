"""Tests for M4B: checkpoint schema, atomicity, RNG state, resume equality."""

from __future__ import annotations

import dataclasses
import json
import random
import subprocess
import sys
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest
import torch

from rlcore._testing import ScriptedEnv
from rlcore.agents.ppo.train import PpoConfig
from rlcore.agents.ppo.train import train as ppo_train
from rlcore.agents.reinforce.train import ReinforceConfig
from rlcore.agents.reinforce.train import train as reinforce_train
from rlcore.checkpoints import (
    check_resume_config,
    load_checkpoint,
    pickle_env,
    save_checkpoint,
    unpickle_env,
)
from rlcore.utils.seeding import restore_rng, rng_state, seed_everything

if TYPE_CHECKING:
    from pathlib import Path


def reinforce_config(updates: int, checkpoint_every: int = 0) -> ReinforceConfig:
    return ReinforceConfig(
        seed=5,
        updates=updates,
        episodes_per_update=2,
        eval_every=1,
        eval_episodes=2,
        stop_return=None,
        checkpoint_every=checkpoint_every,
    )


def ppo_config(total_updates: int, checkpoint_every: int = 0) -> PpoConfig:
    return PpoConfig(
        seed=5,
        total_updates=total_updates,
        n_steps=64,
        minibatch_size=32,
        n_epochs=2,
        eval_every=1,
        eval_episodes=2,
        stop_return=None,
        checkpoint_every=checkpoint_every,
    )


class TestEnvelope:
    def test_save_load_roundtrip_and_no_temp_file(self, tmp_path: Path) -> None:
        path = tmp_path / "checkpoint.pt"
        save_checkpoint(path, algo="ppo", payload={"x": torch.ones(3)})
        assert not list(tmp_path.glob("*.tmp"))
        payload = load_checkpoint(path, expected_algo="ppo")
        assert torch.equal(payload["x"], torch.ones(3))

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_checkpoint(tmp_path / "absent.pt", expected_algo="ppo")

    def test_corrupted_file(self, tmp_path: Path) -> None:
        path = tmp_path / "checkpoint.pt"
        path.write_bytes(b"not a checkpoint at all")
        with pytest.raises(ValueError, match="unreadable or corrupted"):
            load_checkpoint(path, expected_algo="ppo")

    def test_wrong_format(self, tmp_path: Path) -> None:
        path = tmp_path / "checkpoint.pt"
        torch.save({"something": "else"}, path)
        with pytest.raises(ValueError, match="not an rlcore checkpoint"):
            load_checkpoint(path, expected_algo="ppo")

    def test_newer_version_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "checkpoint.pt"
        torch.save(
            {"format": "rlcore-checkpoint", "version": 999, "algo": "ppo", "payload": {}}, path
        )
        with pytest.raises(ValueError, match="schema version 999"):
            load_checkpoint(path, expected_algo="ppo")

    def test_algo_mismatch(self, tmp_path: Path) -> None:
        path = tmp_path / "checkpoint.pt"
        save_checkpoint(path, algo="reinforce", payload={})
        with pytest.raises(ValueError, match="written by algorithm 'reinforce'"):
            load_checkpoint(path, expected_algo="ppo")


class TestEnvPickling:
    def test_scripted_env_roundtrip(self) -> None:
        env = ScriptedEnv([1.0, 2.0], terminate=True)
        env.reset()
        env.step(0)
        restored = unpickle_env(pickle_env(env))
        _obs, reward, terminated, _truncated, _ = restored.step(1)
        assert reward == 2.0
        assert terminated

    def test_unpicklable_env_fails_at_save_with_boundary_message(self) -> None:
        env = ScriptedEnv([1.0])
        env.attached_lambda = lambda: None  # type: ignore[attr-defined]
        with pytest.raises(ValueError, match="picklable environment"):
            pickle_env(env)


class TestRngState:
    def test_roundtrip_continues_all_streams(self) -> None:
        rng = seed_everything(11)
        torch.rand(5, generator=rng.torch)  # advance some streams
        torch.rand(3)
        np.random.rand(2)
        state = rng_state(rng)

        expected = (
            torch.rand(4, generator=rng.torch),
            rng.numpy.random(4),
            rng.python.random(),
            torch.rand(4),
            np.random.rand(4),
            random.random(),
        )
        restored = restore_rng(state)
        actual = (
            torch.rand(4, generator=restored.torch),
            restored.numpy.random(4),
            restored.python.random(),
            torch.rand(4),
            np.random.rand(4),
            random.random(),
        )
        assert torch.equal(expected[0], actual[0])
        assert np.array_equal(expected[1], actual[1])
        assert expected[2] == actual[2]
        assert torch.equal(expected[3], actual[3])
        assert np.array_equal(expected[4], actual[4])
        assert expected[5] == actual[5]


class TestResumeValidation:
    def test_budget_field_may_differ(self) -> None:
        stored = dataclasses.asdict(ppo_config(2))
        requested = dataclasses.asdict(ppo_config(4))
        check_resume_config(stored, requested, budget_field="total_updates")

    def test_other_fields_may_not(self) -> None:
        stored = dataclasses.asdict(ppo_config(2))
        requested = dataclasses.asdict(dataclasses.replace(ppo_config(2), lr=1e-2))
        with pytest.raises(ValueError, match="lr"):
            check_resume_config(stored, requested, budget_field="total_updates")

    def test_checkpoint_every_requires_out_dir(self) -> None:
        with pytest.raises(ValueError, match="requires out_dir"):
            ppo_train(ppo_config(1, checkpoint_every=1))


class TestResumeEqualsContinuous:
    """The M4B gate: 2N continuous vs N + save + restore + N, in-process."""

    def test_reinforce(self, tmp_path: Path) -> None:
        continuous = reinforce_train(reinforce_config(4))
        reinforce_train(reinforce_config(2, checkpoint_every=2), out_dir=tmp_path)
        resumed = reinforce_train(reinforce_config(4), resume_from=tmp_path / "checkpoint.pt")
        for key, value in continuous.policy.state_dict().items():
            assert torch.equal(value, resumed.policy.state_dict()[key]), key
        assert continuous.history[2:] == resumed.history[2:]
        assert continuous.total_env_steps == resumed.total_env_steps

    def test_ppo(self, tmp_path: Path) -> None:
        continuous = ppo_train(ppo_config(4))
        ppo_train(ppo_config(2, checkpoint_every=2), out_dir=tmp_path)
        resumed = ppo_train(ppo_config(4), resume_from=tmp_path / "checkpoint.pt")
        for key, value in continuous.model.state_dict().items():
            assert torch.equal(value, resumed.model.state_dict()[key]), key
        assert continuous.history[2:] == resumed.history[2:]
        assert continuous.total_env_steps == resumed.total_env_steps

    def test_resume_preserves_run_id(self, tmp_path: Path) -> None:
        first = ppo_train(ppo_config(2, checkpoint_every=2), out_dir=tmp_path)
        resumed = ppo_train(ppo_config(4), resume_from=tmp_path / "checkpoint.pt")
        assert resumed.run_id == first.run_id


FRESH_PROCESS_SNIPPET = """
import json, sys
from pathlib import Path
import torch

algo, config_json, checkpoint, out = sys.argv[1], sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4])
config_dict = json.loads(config_json)
if algo == "reinforce":
    from rlcore.agents.reinforce.train import ReinforceConfig as Config, train
else:
    from rlcore.agents.ppo.train import PpoConfig as Config, train
result = train(Config(**config_dict), resume_from=checkpoint)
model = result.policy if algo == "reinforce" else result.model
torch.save({"state_dict": model.state_dict(), "history_tail": result.history[2:]}, out)
"""


class TestFreshProcessResume:
    """The strict form of the gate: restore happens in a new interpreter."""

    def run_fresh(
        self, algo: str, config_dict: dict[str, object], tmp_path: Path
    ) -> dict[str, Any]:
        out = tmp_path / "fresh_result.pt"
        subprocess.run(
            [
                sys.executable,
                "-c",
                FRESH_PROCESS_SNIPPET,
                algo,
                json.dumps(config_dict),
                str(tmp_path / "checkpoint.pt"),
                str(out),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return cast("dict[str, Any]", torch.load(out, weights_only=False))

    def test_reinforce_fresh_process(self, tmp_path: Path) -> None:
        continuous = reinforce_train(reinforce_config(4))
        reinforce_train(reinforce_config(2, checkpoint_every=2), out_dir=tmp_path)
        fresh = self.run_fresh("reinforce", dataclasses.asdict(reinforce_config(4)), tmp_path)
        for key, value in continuous.policy.state_dict().items():
            assert torch.equal(value, fresh["state_dict"][key]), key
        assert continuous.history[2:] == fresh["history_tail"]

    def test_ppo_fresh_process(self, tmp_path: Path) -> None:
        continuous = ppo_train(ppo_config(4))
        ppo_train(ppo_config(2, checkpoint_every=2), out_dir=tmp_path)
        fresh = self.run_fresh("ppo", dataclasses.asdict(ppo_config(4)), tmp_path)
        for key, value in continuous.model.state_dict().items():
            assert torch.equal(value, fresh["state_dict"][key]), key
        assert continuous.history[2:] == fresh["history_tail"]
