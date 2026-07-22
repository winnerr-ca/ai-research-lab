"""Deterministic seeding: one entry point for all sources of randomness.

Every run on the platform is reproducible from ``(config, git SHA, seed)``
(see ``docs/ARCHITECTURE.md`` §3). This module provides the single seeding
entry point, :func:`seed_everything`, which seeds the global RNGs (for
third-party code that draws from them) and returns an :class:`Rng` bundle of
explicitly seeded generators for our own code to draw from.

GPU determinism is best-effort per PyTorch's own guarantees; the tested
reproducibility bar is CPU (``torch.use_deterministic_algorithms`` is a
documented opt-in, not set here).
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch

# Aliases for annotating the Rng fields named after their libraries: inside the
# class body the field names shadow the module names.
from numpy.random import Generator as _NumpyGenerator
from torch import Generator as _TorchGenerator

__all__ = ["Rng", "seed_everything"]

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
        :class:`numpy.random.SeedSequence`, so nearby root seeds or indices do
        not produce correlated streams (as naive ``seed + index`` schemes do).

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
