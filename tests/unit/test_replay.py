"""Tests for the shared replay buffer: overwrite, sampling, state roundtrip."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from rlcore.replay import ReplayBuffer


def make_buffer(capacity: int = 4) -> ReplayBuffer:
    return ReplayBuffer(capacity, obs_shape=(2,))


def fill(buffer: ReplayBuffer, count: int, offset: int = 0) -> None:
    for i in range(count):
        value = float(i + offset)
        buffer.add(np.full(2, value), i + offset, value, np.full(2, value + 0.5), False)


class TestAddAndOverwrite:
    def test_size_grows_to_capacity(self) -> None:
        buffer = make_buffer(4)
        fill(buffer, 3)
        assert len(buffer) == 3
        fill(buffer, 3, offset=3)
        assert len(buffer) == 4

    def test_oldest_transitions_are_overwritten(self) -> None:
        buffer = make_buffer(3)
        fill(buffer, 5)  # rewards 0..4; oldest (0, 1) overwritten
        state = buffer.state()
        stored_rewards = set(state["reward"][: state["size"]].tolist())
        assert stored_rewards == {2.0, 3.0, 4.0}

    def test_terminated_flag_stored(self) -> None:
        buffer = make_buffer(2)
        buffer.add(np.zeros(2), 0, 1.0, np.ones(2), True)
        batch = buffer.sample(4, generator=torch.Generator().manual_seed(0))
        assert bool(batch.terminated.all())

    def test_continuous_actions_supported(self) -> None:
        buffer = ReplayBuffer(4, obs_shape=(3,), action_shape=(2,), action_dtype=np.float32)
        buffer.add(np.zeros(3), np.array([0.5, -0.5], dtype=np.float32), 1.0, np.ones(3), False)
        batch = buffer.sample(2, generator=torch.Generator().manual_seed(0))
        assert batch.action.shape == (2, 2)
        assert batch.action.dtype == torch.float32


class TestSampling:
    def test_deterministic_given_generator(self) -> None:
        buffer = make_buffer(8)
        fill(buffer, 8)
        first = buffer.sample(16, generator=torch.Generator().manual_seed(3))
        second = buffer.sample(16, generator=torch.Generator().manual_seed(3))
        for key, value in first.items():
            assert torch.equal(value, second[key]), key

    def test_batch_fields_and_shapes(self) -> None:
        buffer = make_buffer(8)
        fill(buffer, 5)
        batch = buffer.sample(6, generator=torch.Generator().manual_seed(0))
        assert set(batch.keys()) == {"obs", "action", "reward", "next_obs", "terminated"}
        assert batch.obs.shape == (6, 2)
        assert batch.action.dtype == torch.int64
        assert batch.terminated.dtype == torch.bool

    def test_sample_only_covers_stored_transitions(self) -> None:
        buffer = make_buffer(16)
        fill(buffer, 3)  # rewards 0, 1, 2
        batch = buffer.sample(64, generator=torch.Generator().manual_seed(0))
        assert set(batch.reward.tolist()) <= {0.0, 1.0, 2.0}

    def test_empty_buffer_rejected(self) -> None:
        with pytest.raises(ValueError, match="empty replay buffer"):
            make_buffer().sample(1)

    def test_bad_batch_size_rejected(self) -> None:
        buffer = make_buffer()
        fill(buffer, 1)
        with pytest.raises(ValueError, match="batch_size"):
            buffer.sample(0)


class TestStateRoundtrip:
    def test_save_restore_preserves_contents_and_cursor(self) -> None:
        buffer = make_buffer(4)
        fill(buffer, 6)  # wrapped: pos=2, size=4
        state = buffer.state()

        restored = make_buffer(4)
        restored.restore_state(state)
        assert len(restored) == len(buffer)
        first = buffer.sample(8, generator=torch.Generator().manual_seed(7))
        second = restored.sample(8, generator=torch.Generator().manual_seed(7))
        for key, value in first.items():
            assert torch.equal(value, second[key]), key

        # Cursor position must survive: the next add overwrites the same slot.
        buffer.add(np.full(2, 99.0), 99, 99.0, np.full(2, 99.0), False)
        restored.add(np.full(2, 99.0), 99, 99.0, np.full(2, 99.0), False)
        assert np.array_equal(buffer.state()["reward"], restored.state()["reward"])

    def test_capacity_mismatch_rejected(self) -> None:
        buffer = make_buffer(4)
        fill(buffer, 2)
        other = make_buffer(8)
        with pytest.raises(ValueError, match="capacity"):
            other.restore_state(buffer.state())

    def test_state_is_a_copy(self) -> None:
        buffer = make_buffer(4)
        fill(buffer, 2)
        state = buffer.state()
        buffer.add(np.full(2, 50.0), 5, 50.0, np.full(2, 50.0), False)
        assert 50.0 not in state["reward"].tolist()
