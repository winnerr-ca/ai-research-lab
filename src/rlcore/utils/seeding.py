"""Deterministic seeding: one entry point for all sources of randomness.

Every run on the platform is reproducible from ``(config, git SHA, seed)``
(see ``docs/ARCHITECTURE.md`` §3). This module provides the single seeding
entry point, :func:`seed_everything`, which seeds the global RNGs (for
third-party code that draws from them) and returns an :class:`Rng` bundle of
explicitly seeded generators for our own code to draw from.

Scope of the guarantee, stated narrowly: :func:`seed_everything` establishes
controlled random *streams* — it does not by itself make training runs
deterministic. Full training determinism additionally depends on
deterministic PyTorch operations (``torch.use_deterministic_algorithms``,
kernel/reduction behavior), hardware, environment (simulator) behavior,
threading, and pinned software versions. The platform's tested bar is
same-machine CPU determinism with the locked dependency set; GPU determinism
is best-effort per PyTorch's own guarantees.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

# Aliases for annotating the Rng fields named after their libraries: inside the
# class body the field names shadow the module names.
from numpy.random import Generator as _NumpyGenerator
from torch import Generator as _TorchGenerator

__all__ = ["Rng", "restore_rng", "rng_state", "seed_everything"]

#: Seeds must satisfy ``0 <= seed < 2**32`` (NumPy's legacy-seed constraint,
#: the strictest of the seeded libraries).
MAX_SEED: int = 2**32


@dataclass(frozen=True, slots=True)
class Rng:
    """A bundle of explicitly seeded random number generators.

    Prefer drawing from these named streams over the global RNGs: components
    sharing a global stream become order-dependent, so an unrelated code
    change (or a reordered import) silently alters every downstream draw.
    The globals are seeded too, but only as a safety net for third-party code.

    Attributes:
        seed: The root seed this bundle was created from.
        torch: A seeded CPU :class:`torch.Generator`.
        numpy: A seeded :class:`numpy.random.Generator`.
        python: A seeded :class:`random.Random` instance.
    """

    seed: int
    torch: _TorchGenerator
    numpy: _NumpyGenerator
    python: random.Random

    def child_seed(self, index: int) -> int:
        """Derive a stable, well-mixed child seed, e.g. for the ``index``-th env.

        Derivation depends only on ``(self.seed, index)`` — not on call order
        and not on any generator's current state — so components can be
        re-seeded independently and in any order. Mixing goes through
        :class:`numpy.random.SeedSequence`, which is designed to provide
        reproducible, well-separated child streams from a root seed and a
        spawn key.

        Args:
            index: Non-negative stream index.

        Returns:
            A child seed in ``[0, 2**32)``.

        Raises:
            ValueError: If ``index`` is negative.
        """
        if index < 0:
            raise ValueError(f"Child-seed index must be non-negative, got {index}.")
        sequence = np.random.SeedSequence(entropy=self.seed, spawn_key=(index,))
        return int(sequence.generate_state(1)[0])


def rng_state(rng: Rng) -> dict[str, Any]:
    """Capture the full RNG state: the bundle's streams and the globals.

    The returned dict is what checkpoints store; restore with
    :func:`restore_rng`. Includes Python's global ``random`` state, NumPy's
    legacy global state, and torch's global CPU state alongside the
    bundle's explicit generators — resuming with only the explicit streams
    would silently desynchronize any code drawing from the globals.
    """
    return {
        "seed": rng.seed,
        "torch_generator": rng.torch.get_state(),
        "numpy_bit_generator": rng.numpy.bit_generator.state,
        "python_random": rng.python.getstate(),
        "global_torch": torch.get_rng_state(),
        "global_numpy": np.random.get_state(),
        "global_python": random.getstate(),
    }


def restore_rng(state: dict[str, Any]) -> Rng:
    """Restore global RNGs and rebuild the bundle from :func:`rng_state`.

    The inverse of seeding-plus-training: after this call, every stream
    continues exactly where the capture left it.
    """
    torch.set_rng_state(state["global_torch"])
    np.random.set_state(state["global_numpy"])
    random.setstate(state["global_python"])

    torch_generator = torch.Generator()
    torch_generator.set_state(state["torch_generator"])
    numpy_generator = np.random.default_rng()
    numpy_generator.bit_generator.state = state["numpy_bit_generator"]
    python_random = random.Random()
    python_random.setstate(state["python_random"])
    return Rng(
        seed=int(state["seed"]),
        torch=torch_generator,
        numpy=numpy_generator,
        python=python_random,
    )


def seed_everything(seed: int) -> Rng:
    """Seed all global RNGs and return an explicitly seeded generator bundle.

    Seeds Python's ``random`` module, NumPy's legacy global generator, and
    torch (CPU and all visible CUDA devices), then builds a fresh :class:`Rng`
    whose generators are independent of the globals.

    Args:
        seed: Root seed, ``0 <= seed < 2**32``.

    Returns:
        An :class:`Rng` bundle seeded from ``seed``.

    Raises:
        ValueError: If ``seed`` is outside the valid range.
    """
    if not 0 <= seed < MAX_SEED:
        raise ValueError(f"Seed must satisfy 0 <= seed < 2**32, got {seed}.")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch_generator = torch.Generator()
    torch_generator.manual_seed(seed)
    return Rng(
        seed=seed,
        torch=torch_generator,
        numpy=np.random.default_rng(seed),
        python=random.Random(seed),
    )
