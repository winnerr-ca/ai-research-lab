"""Unit tests for rlcore.utils.seeding: global and bundled determinism."""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from rlcore.types import Batch
from rlcore.utils.seeding import Rng, seed_everything


def global_draws() -> tuple[torch.Tensor, np.ndarray, float]:
    return torch.rand(4), np.random.rand(4), random.random()


def bundle_draws(rng: Rng) -> tuple[torch.Tensor, np.ndarray, float]:
    return (
        torch.rand(4, generator=rng.torch),
        rng.numpy.random(4),
        rng.python.random(),
    )


class TestSeedEverything:
    def test_global_rngs_are_deterministic(self) -> None:
        seed_everything(123)
        first = global_draws()
        seed_everything(123)
        second = global_draws()
        assert torch.equal(first[0], second[0])
        assert np.array_equal(first[1], second[1])
        assert first[2] == second[2]

    def test_bundle_is_deterministic(self) -> None:
        first = bundle_draws(seed_everything(123))
        second = bundle_draws(seed_everything(123))
        assert torch.equal(first[0], second[0])
        assert np.array_equal(first[1], second[1])
        assert first[2] == second[2]

    def test_different_seeds_differ(self) -> None:
        first = bundle_draws(seed_everything(123))
        second = bundle_draws(seed_everything(124))
        assert not torch.equal(first[0], second[0])
        assert not np.array_equal(first[1], second[1])
        assert first[2] != second[2]

    def test_bundle_is_independent_of_global_stream(self) -> None:
        rng = seed_everything(123)
        torch.rand(100)  # perturb the globals
        np.random.rand(100)
        random.random()
        perturbed = bundle_draws(rng)
        unperturbed = bundle_draws(seed_everything(123))
        assert torch.equal(perturbed[0], unperturbed[0])
        assert np.array_equal(perturbed[1], unperturbed[1])
        assert perturbed[2] == unperturbed[2]

    @pytest.mark.parametrize("bad_seed", [-1, 2**32])
    def test_out_of_range_seed_raises(self, bad_seed: int) -> None:
        with pytest.raises(ValueError, match="0 <= seed"):
            seed_everything(bad_seed)

    def test_boundary_seeds_accepted(self) -> None:
        seed_everything(0)
        seed_everything(2**32 - 1)


class TestChildSeed:
    def test_stable_across_bundles(self) -> None:
        assert seed_everything(9).child_seed(3) == seed_everything(9).child_seed(3)

    def test_independent_of_draw_order(self) -> None:
        rng = seed_everything(9)
        before = rng.child_seed(3)
        torch.rand(10, generator=rng.torch)
        assert rng.child_seed(3) == before

    def test_distinct_per_index_and_root(self) -> None:
        rng = seed_everything(9)
        assert rng.child_seed(0) != rng.child_seed(1)
        assert rng.child_seed(0) != seed_everything(10).child_seed(0)

    def test_range_is_valid_seed_range(self) -> None:
        rng = seed_everything(9)
        assert all(0 <= rng.child_seed(i) < 2**32 for i in range(32))

    def test_negative_index_raises(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            seed_everything(9).child_seed(-1)


def test_batch_contents_identical_across_seeded_contexts() -> None:
    """Phase 0 acceptance criterion 3: seeded contexts produce identical batches."""

    def build() -> Batch:
        rng = seed_everything(42)
        return Batch(
            obs=torch.rand(16, 4, generator=rng.torch),
            action=torch.randint(0, 2, (16,), generator=rng.torch),
            reward=torch.as_tensor(rng.numpy.random(16), dtype=torch.float32),
        )

    first, second = build(), build()
    for key, value in first.items():
        assert torch.equal(second[key], value)
