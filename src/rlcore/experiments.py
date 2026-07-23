"""Run identity, run-directory writing, and provenance metadata.

M4A defines the standard run-directory layout, written entirely locally
(no external account or credentials involved):

```
run_dir/
├── run.json          # identity: run_id, timestamps, algo/env/seed, metadata
├── config.yaml       # resolved config
├── metrics.jsonl     # one JSON object per training iteration
├── result.json       # final summary payload (algorithm-specific fields)
├── final_model.pt    # final weights artifact (state_dict + small header)
├── summary.md        # human-readable summary        (rlcore.reporting)
└── plots/
    └── learning_curve.png                            (rlcore.reporting)
```

``final_model.pt`` is a *final artifact* for evaluation and sharing — it is
not a resumable training checkpoint; that is M4B's separate contract.
"""

from __future__ import annotations

import json
import platform
import subprocess
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import gymnasium as gym
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

import rlcore

if TYPE_CHECKING:
    from pathlib import Path

FINAL_MODEL_FORMAT = "rlcore-final-model"
FINAL_MODEL_VERSION = 1


def new_run_id(algo: str, env_id: str, seed: int) -> str:
    """Mint a unique, human-scannable run identifier.

    Layout: ``{UTC timestamp}-{algo}-{env}-s{seed}-{6 hex}`` — sortable by
    creation time, greppable by algo/env/seed, unique via the random suffix.
    """
    stamp = datetime.now(tz=UTC).strftime("%Y%m%d-%H%M%S")
    env_slug = env_id.lower().replace("/", "-")
    return f"{stamp}-{algo}-{env_slug}-s{seed}-{uuid.uuid4().hex[:6]}"


def _git_state() -> tuple[str, bool]:
    """Return ``(sha, dirty)``; ``("unknown", False)`` outside a git repo."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown", False
    return sha, bool(status)


def git_commit() -> str:
    """Return the current commit SHA (with ``-dirty`` if the tree has changes).

    A report stamped with a dirty SHA is not reproducible from that commit;
    the suffix makes that visible instead of silently claiming a clean state.
    """
    sha, dirty = _git_state()
    if sha == "unknown":
        return sha
    return f"{sha}-dirty" if dirty else sha


def run_metadata() -> dict[str, str]:
    """Version, device, OS, and provenance stamp for run records and reports."""
    sha, dirty = _git_state()
    return {
        "rlcore": rlcore.__version__,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "gymnasium": gym.__version__,
        "numpy": np.__version__,
        "hydra": hydra.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "device": "cpu",  # training device; revisit at the GPU milestone (M7)
        "cuda_available": str(torch.cuda.is_available()),
        "commit": f"{sha}-dirty" if dirty and sha != "unknown" else sha,
        "git_dirty": str(dirty),
    }


def write_run_record(out_dir: Path, *, run_id: str, algo: str, env_id: str, seed: int) -> None:
    """Write ``run.json``: the run's identity plus full environment metadata."""
    out_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "run_id": run_id,
        "created_at": datetime.now(tz=UTC).isoformat(),
        "algo": algo,
        "env_id": env_id,
        "seed": seed,
        "metadata": run_metadata(),
    }
    (out_dir / "run.json").write_text(json.dumps(record, indent=2))


def write_run_outputs(
    out_dir: Path,
    config: object,
    history: list[dict[str, float]],
    result_payload: dict[str, Any],
) -> None:
    """Write ``config.yaml``, ``metrics.jsonl``, and ``result.json``.

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


def save_final_model(out_dir: Path, *, algo: str, state_dict: dict[str, torch.Tensor]) -> None:
    """Save the final-weights artifact (``final_model.pt``).

    A small versioned header wraps the state dict so loaders can fail
    loudly on the wrong file kind instead of part-loading.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": FINAL_MODEL_FORMAT,
            "version": FINAL_MODEL_VERSION,
            "algo": algo,
            "state_dict": state_dict,
        },
        out_dir / "final_model.pt",
    )


def load_final_model(path: Path, *, expected_algo: str) -> dict[str, torch.Tensor]:
    """Load a ``final_model.pt`` artifact, validating header and algorithm.

    Raises:
        ValueError: If the file is not a final-model artifact, is from a
            newer format version, or belongs to a different algorithm.
    """
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format") != FINAL_MODEL_FORMAT:
        raise ValueError(f"{path} is not an rlcore final-model artifact.")
    if payload["version"] > FINAL_MODEL_VERSION:
        raise ValueError(
            f"{path} has final-model version {payload['version']}; this rlcore "
            f"supports up to {FINAL_MODEL_VERSION}. Upgrade rlcore to load it."
        )
    if payload["algo"] != expected_algo:
        raise ValueError(
            f"{path} contains a {payload['algo']!r} model, expected {expected_algo!r}."
        )
    state_dict: dict[str, torch.Tensor] = payload["state_dict"]
    return state_dict
