"""Unit tests for rlcore.types: Batch and PolicyOutput."""

from __future__ import annotations

import pickle

import pytest
import torch

from rlcore.types import CANONICAL_KEYS, Batch, PolicyOutput


def make_batch(n: int = 8) -> Batch:
    return Batch(
        obs=torch.arange(n * 4, dtype=torch.float32).reshape(n, 4),
        action=torch.arange(n, dtype=torch.int64),
        reward=torch.linspace(0.0, 1.0, n),
    )


class TestConstruction:
    def test_from_kwargs(self) -> None:
        batch = make_batch(8)
        assert len(batch) == 8
        assert set(batch.keys()) == {"obs", "action", "reward"}

    def test_from_mapping(self) -> None:
        batch = Batch({"obs": torch.zeros(3, 2), "reward": torch.zeros(3)})
        assert len(batch) == 3

    def test_mapping_and_kwargs_merge(self) -> None:
        batch = Batch({"obs": torch.zeros(3, 2)}, reward=torch.zeros(3))
        assert set(batch.keys()) == {"obs", "reward"}

    def test_duplicate_key_across_mapping_and_kwargs_raises(self) -> None:
        with pytest.raises(ValueError, match="both"):
            Batch({"obs": torch.zeros(3, 2)}, obs=torch.zeros(3, 2))

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one field"):
            Batch()

    def test_mismatched_lengths_raises(self) -> None:
        with pytest.raises(ValueError, match="leading-dimension size"):
            Batch(obs=torch.zeros(4, 2), reward=torch.zeros(5))

    def test_non_tensor_raises(self) -> None:
        with pytest.raises(TypeError, match=r"torch\.Tensor"):
            Batch(obs=[1, 2, 3])  # type: ignore[arg-type]

    def test_scalar_field_raises(self) -> None:
        with pytest.raises(ValueError, match="leading batch dimension"):
            Batch(obs=torch.zeros(3, 2), reward=torch.tensor(1.0))

    def test_reserved_name_raises(self) -> None:
        with pytest.raises(ValueError, match="collides"):
            Batch(to=torch.zeros(3))

    def test_private_name_raises(self) -> None:
        with pytest.raises(ValueError, match="collides"):
            Batch({"_fields": torch.zeros(3)})

    def test_non_identifier_name_raises(self) -> None:
        with pytest.raises(ValueError, match="identifier"):
            Batch({"bad key": torch.zeros(3)})

    def test_mixed_devices_raises(self) -> None:
        with pytest.raises(ValueError, match="one device"):
            Batch(obs=torch.zeros(3, 2), reward=torch.zeros(3, device="meta"))

    def test_heterogeneous_dtypes_allowed(self) -> None:
        batch = Batch(action=torch.zeros(4, dtype=torch.int64), done=torch.zeros(4).bool())
        assert batch.action.dtype == torch.int64
        assert batch.done.dtype == torch.bool


class TestAccess:
    def test_attribute_access(self) -> None:
        batch = make_batch()
        assert torch.equal(batch.obs, batch["obs"])

    def test_missing_attribute_raises_with_available_fields(self) -> None:
        with pytest.raises(AttributeError, match="available fields"):
            _ = make_batch().advantage

    def test_missing_key_raises(self) -> None:
        with pytest.raises(KeyError, match="available fields"):
            _ = make_batch()["advantage"]

    def test_contains(self) -> None:
        batch = make_batch()
        assert "obs" in batch
        assert "advantage" not in batch

    def test_batch_size_and_device(self) -> None:
        batch = make_batch(5)
        assert batch.batch_size == 5
        assert batch.device == torch.device("cpu")

    def test_as_dict_is_shallow_copy(self) -> None:
        batch = make_batch()
        out = batch.as_dict()
        out["extra"] = torch.zeros(8)
        assert "extra" not in batch
        assert out["obs"] is batch.obs  # tensors are shared, not copied

    def test_repr_mentions_fields_and_length(self) -> None:
        text = repr(make_batch(8))
        assert "len=8" in text
        assert "obs" in text


class TestIndexing:
    def test_int_returns_length_one_batch(self) -> None:
        batch = make_batch(8)
        row = batch[3]
        assert isinstance(row, Batch)
        assert len(row) == 1
        assert torch.equal(row.obs[0], batch.obs[3])

    def test_negative_int(self) -> None:
        batch = make_batch(8)
        assert torch.equal(batch[-1].obs[0], batch.obs[7])
        assert torch.equal(batch[-3].obs[0], batch.obs[5])

    def test_int_out_of_range_raises(self) -> None:
        with pytest.raises(IndexError):
            _ = make_batch(8)[8]
        with pytest.raises(IndexError):
            _ = make_batch(8)[-9]

    def test_slice(self) -> None:
        batch = make_batch(8)
        part = batch[2:5]
        assert len(part) == 3
        assert torch.equal(part.obs, batch.obs[2:5])

    def test_index_tensor(self) -> None:
        batch = make_batch(8)
        idx = torch.tensor([0, 7, 3])
        part = batch[idx]
        assert len(part) == 3
        assert torch.equal(part.action, batch.action[idx])

    def test_index_list(self) -> None:
        batch = make_batch(8)
        part = batch[[1, 2]]
        assert len(part) == 2

    def test_boolean_mask(self) -> None:
        batch = make_batch(8)
        mask = batch.action % 2 == 0
        part = batch[mask]
        assert len(part) == 4
        assert bool((part.action % 2 == 0).all())

    def test_direct_iteration_refused(self) -> None:
        with pytest.raises(TypeError, match="not directly iterable"):
            iter(make_batch())


class TestDerivation:
    def test_to_same_device_preserves_values(self) -> None:
        batch = make_batch()
        moved = batch.to("cpu")
        assert torch.equal(moved.obs, batch.obs)

    def test_with_fields_adds_and_replaces(self) -> None:
        batch = make_batch(8)
        derived = batch.with_fields(advantage=torch.ones(8), reward=torch.zeros(8))
        assert torch.equal(derived.advantage, torch.ones(8))
        assert bool((derived.reward == 0).all())
        assert "advantage" not in batch  # original unchanged
        assert bool((batch.reward > 0).any())

    def test_with_fields_validates_length(self) -> None:
        with pytest.raises(ValueError, match="leading-dimension size"):
            make_batch(8).with_fields(advantage=torch.ones(7))

    def test_immutability(self) -> None:
        batch = make_batch()
        with pytest.raises(AttributeError, match="immutable"):
            batch.obs = torch.zeros(8, 4)
        with pytest.raises(AttributeError, match="immutable"):
            del batch.obs


class TestMinibatches:
    def test_partitions_exactly_without_shuffle(self) -> None:
        batch = make_batch(8)
        parts = list(batch.minibatches(4, shuffle=False))
        assert [len(p) for p in parts] == [4, 4]
        assert torch.equal(torch.cat([p.action for p in parts]), batch.action)

    def test_covers_every_index_once_when_shuffled(self) -> None:
        batch = make_batch(10)
        parts = list(batch.minibatches(3, generator=torch.Generator().manual_seed(0)))
        seen = torch.cat([p.action for p in parts])
        assert [len(p) for p in parts] == [3, 3, 3, 1]
        assert torch.equal(torch.sort(seen).values, batch.action)

    def test_shuffle_is_deterministic_given_generator(self) -> None:
        batch = make_batch(10)
        first = [p.action for p in batch.minibatches(3, generator=torch.Generator().manual_seed(7))]
        second = [
            p.action for p in batch.minibatches(3, generator=torch.Generator().manual_seed(7))
        ]
        for a, b in zip(first, second, strict=True):
            assert torch.equal(a, b)

    def test_drop_last(self) -> None:
        batch = make_batch(10)
        parts = list(batch.minibatches(3, shuffle=False, drop_last=True))
        assert [len(p) for p in parts] == [3, 3, 3]

    def test_invalid_size_raises(self) -> None:
        with pytest.raises(ValueError, match=">= 1"):
            list(make_batch().minibatches(0))


class TestSerialization:
    def test_pickle_round_trip(self) -> None:
        batch = make_batch(6)
        restored = pickle.loads(pickle.dumps(batch))
        assert isinstance(restored, Batch)
        assert set(restored.keys()) == set(batch.keys())
        for key, value in batch.items():
            assert torch.equal(restored[key], value)


class TestPolicyOutput:
    def test_defaults(self) -> None:
        out = PolicyOutput(action=torch.tensor([1, 0]))
        assert not out.extras

    def test_extras(self) -> None:
        out = PolicyOutput(action=torch.tensor([1]), extras={"logprob": torch.tensor([-0.5])})
        assert "logprob" in out.extras


def test_canonical_keys_cover_transition_fields() -> None:
    assert {"obs", "action", "reward", "terminated", "truncated", "next_obs"} <= CANONICAL_KEYS
