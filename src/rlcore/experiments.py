"""Run-output writing and run metadata, extracted at M3.

Both trainers wrote identical run directories (resolved config,
per-iteration metrics as JSONL, a final result payload), and both
validation scripts duplicated commit/version stamping. This module owns
those; the full experiment manager (trackers, run registry) remains an M4
deliverable and is out of scope here.
"""

from __future__ import annotations

import json
import platform
import subprocess
from typing import TYPE_CHECKING, Any

import gymnasium as gym
import torch
from omegaconf import OmegaConf

import rlcore

if TYPE_CHECKING:
    from pathlib import Path


def write_run_outputs(
    out_dir: Path,
    config: object,
    history: list[dict[str, float]],
    result_payload: dict[str, Any],
) -> None:
    """Write a run directory: ``config.yaml``, ``metrics.jsonl``, ``result.json``.

    Args:
        out_dir: Target directory (created if missing).
        config: The run's config dataclass instance (serialized via
            OmegaConf, matching what Hydra runs record).
        history: Per-iteration metric dicts, written one JSON object per
            line.
        result_payload: Final-result fields, algorithm-specific by design.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(OmegaConf.to_yaml(OmegaConf.structured(config)))
    with (out_dir / "metrics.jsonl").open("w") as stream:
        for entry in history:
            stream.write(json.dumps(entry) + "\n")
    (out_dir / "result.json").write_text(json.dumps(result_payload, indent=2))


def git_commit() -> str:
    """Return the current commit SHA (with ``-dirty`` if the tree has changes).

    A report stamped with a dirty SHA is not reproducible from that commit;
    the suffix makes that visible instead of silently claiming a clean state.
    """
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"
    return f"{sha}-dirty" if status else sha


def run_metadata() -> dict[str, str]:
    """Version and provenance stamp for validation reports and run records."""
    return {
        "rlcore": rlcore.__version__,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "gymnasium": gym.__version__,
        "platform": platform.platform(),
        "commit": git_commit(),
    }
