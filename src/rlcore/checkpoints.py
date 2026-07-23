"""Resumable training checkpoints: atomic writes, versioned schema, validation.

The M4B contract, scoped to the state REINFORCE and PPO actually have:
model, optimizer, progress counters, full RNG state, resolved config, run
identity, and collector/episode state. A checkpoint is an *envelope*
(``format``/``version``/``algo``) wrapping an algorithm-assembled payload;
the envelope is validated on load with distinct, actionable errors —
never a silent partial load.

**Environment-state boundary (documented, enforced):** exact mid-episode
continuation requires the simulator's internal state, which Gymnasium does
not expose generically. v1 therefore checkpoints environments by
*pickling* them — classic-control envs (CartPole included) pickle cleanly,
carrying both their RNG streams and their physical state, which is what
makes resume bit-exact even mid-episode. Environments that cannot be
pickled fail **at save time** with a clear error: for them, exact resume
is out of scope in v1 (a documented boundary, not a silent degradation).

**Trust boundary:** checkpoints contain pickled objects, so loading uses
full unpickling (``weights_only=False``). Load only checkpoints you (or
your run infrastructure) produced — the same rule as for any pickle file.
"""

from __future__ import annotations

import os
import pickle
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from pathlib import Path

    import gymnasium as gym

CHECKPOINT_FORMAT = "rlcore-checkpoint"
CHECKPOINT_VERSION = 1


def pickle_env(env: gym.Env[Any, Any]) -> bytes:
    """Serialize an env for checkpointing, failing loudly at the boundary.

    Raises:
        ValueError: If the env cannot be pickled — with the v1 boundary
            spelled out rather than deferring the failure to resume time.
    """
    try:
        return pickle.dumps(env)
    except Exception as error:
        raise ValueError(
            f"Checkpointing requires a picklable environment; {type(env).__name__} failed to "
            f"pickle ({error}). rlcore v1 restores environments by unpickling, which is what "
            "makes resume exact (including mid-episode state and RNG streams); for "
            "non-picklable environments exact resume is not supported — see "
            "docs/design/m4b-checkpoint-resume.md."
        ) from error


def unpickle_env(data: bytes) -> gym.Env[Any, Any]:
    """Restore an env pickled by :func:`pickle_env` (trusted input only)."""
    env: gym.Env[Any, Any] = pickle.loads(data)
    return env


def save_checkpoint(path: Path, *, algo: str, payload: dict[str, Any]) -> None:
    """Atomically write a checkpoint envelope.

    Write-to-temp + fsync + rename: a crash mid-write leaves the previous
    checkpoint intact, never a truncated file under the final name.

    Args:
        path: Final checkpoint path.
        algo: Algorithm name recorded in the envelope.
        payload: Algorithm-assembled state (see the trainers'
            checkpoint-payload builders).
    """
    envelope = {
        "format": CHECKPOINT_FORMAT,
        "version": CHECKPOINT_VERSION,
        "algo": algo,
        "payload": payload,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.parent / f"{path.name}.tmp"
    with tmp_path.open("wb") as stream:
        torch.save(envelope, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp_path, path)


def load_checkpoint(path: Path, *, expected_algo: str) -> dict[str, Any]:
    """Load and validate a checkpoint envelope; return its payload.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the file is unreadable, is not an rlcore checkpoint,
            has a newer schema version than this rlcore supports, or was
            written by a different algorithm — each with a distinct message.
    """
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint {path} does not exist.")
    try:
        envelope = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as error:
        raise ValueError(f"Checkpoint {path} is unreadable or corrupted: {error}") from error
    if not isinstance(envelope, dict) or envelope.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"{path} is not an rlcore checkpoint.")
    if envelope["version"] > CHECKPOINT_VERSION:
        raise ValueError(
            f"{path} has checkpoint schema version {envelope['version']}; this rlcore supports "
            f"up to {CHECKPOINT_VERSION}. Upgrade rlcore to load it."
        )
    if envelope["algo"] != expected_algo:
        raise ValueError(
            f"{path} was written by algorithm {envelope['algo']!r}, expected {expected_algo!r}."
        )
    payload: dict[str, Any] = envelope["payload"]
    return payload


def check_resume_config(
    stored: dict[str, Any],
    requested: dict[str, Any],
    *,
    budget_field: str,
) -> None:
    """Validate that a resume request matches the checkpoint's config.

    Every field must match except the *operational* ones: ``budget_field``
    (``updates``/``total_updates`` — extending the budget is the core
    resume use case) and ``checkpoint_every`` (a scheduling knob with no
    effect on training trajectories). Changing anything else silently
    invalidates the run's identity and comparability, so it is refused.

    Raises:
        ValueError: Listing exactly which fields differ.
    """

    def fields_equal(a: object, b: object) -> bool:
        # NaN sentinels (e.g. SAC's target_entropy="auto") must compare equal.
        if isinstance(a, float) and isinstance(b, float) and a != a and b != b:
            return True
        return bool(a == b)

    mutable = {budget_field, "checkpoint_every"}
    mismatched = sorted(
        key
        for key in stored.keys() | requested.keys()
        if key not in mutable and not fields_equal(stored.get(key), requested.get(key))
    )
    if mismatched:
        details = ", ".join(
            f"{key}: checkpoint={stored.get(key)!r} requested={requested.get(key)!r}"
            for key in mismatched
        )
        raise ValueError(
            f"Resume config differs from the checkpoint's beyond {sorted(mutable)}: {details}. "
            "Only the training budget and checkpoint schedule may change on resume."
        )
