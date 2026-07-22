"""Tests for run-output writing and metadata (M3 extraction)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

from rlcore.experiments import git_commit, run_metadata, write_run_outputs

if TYPE_CHECKING:
    from pathlib import Path


@dataclass
class FakeConfig:
    env_id: str = "CartPole-v1"
    seed: int = 3


class TestWriteRunOutputs:
    def test_writes_all_files(self, tmp_path: Path) -> None:
        history = [{"update": 1.0, "loss": 0.5}, {"update": 2.0, "loss": 0.25}]
        write_run_outputs(tmp_path, FakeConfig(), history, {"final": 1.0})

        assert "env_id: CartPole-v1" in (tmp_path / "config.yaml").read_text()
        lines = (tmp_path / "metrics.jsonl").read_text().strip().splitlines()
        assert [json.loads(line)["update"] for line in lines] == [1.0, 2.0]
        assert json.loads((tmp_path / "result.json").read_text()) == {"final": 1.0}

    def test_creates_missing_directory(self, tmp_path: Path) -> None:
        target = tmp_path / "nested" / "run"
        write_run_outputs(target, FakeConfig(), [], {})
        assert (target / "config.yaml").exists()


class TestMetadata:
    def test_run_metadata_keys(self) -> None:
        meta = run_metadata()
        assert {"rlcore", "python", "torch", "gymnasium", "platform", "commit"} <= set(meta)

    def test_git_commit_shape(self) -> None:
        commit = git_commit()
        # In this repo it is a 40-hex SHA with an optional -dirty suffix;
        # outside a repo the sentinel "unknown" is allowed.
        if commit != "unknown":
            sha = commit.removesuffix("-dirty")
            assert len(sha) == 40
            assert all(c in "0123456789abcdef" for c in sha)
