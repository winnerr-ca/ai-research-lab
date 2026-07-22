"""Tests for env-pair creation and the seed-stream layout (M3 extraction)."""

from __future__ import annotations

import numpy as np
import pytest

from rlcore.envs import make_discrete_env_pair
from rlcore.utils.seeding import seed_everything


class TestMakeDiscreteEnvPair:
    def test_dimensions_for_cartpole(self) -> None:
        pair = make_discrete_env_pair("CartPole-v1", seed_everything(0))
        assert pair.obs_dim == 4
        assert pair.n_actions == 2
        pair.close()

    def test_seeding_is_deterministic(self) -> None:
        """Same root seed -> identical env RNG streams for train and eval."""

        def next_obs_pair(seed: int) -> tuple[np.ndarray, np.ndarray]:
            pair = make_discrete_env_pair("CartPole-v1", seed_everything(seed))
            train_obs, _ = pair.train_env.reset()  # continues the seeded stream
            eval_obs, _ = pair.eval_env.reset()
            pair.close()
            return train_obs, eval_obs

        first_train, first_eval = next_obs_pair(7)
        second_train, second_eval = next_obs_pair(7)
        assert np.array_equal(first_train, second_train)
        assert np.array_equal(first_eval, second_eval)

    def test_train_and_eval_streams_differ(self) -> None:
        pair = make_discrete_env_pair("CartPole-v1", seed_everything(7))
        train_obs, _ = pair.train_env.reset()
        eval_obs, _ = pair.eval_env.reset()
        pair.close()
        assert not np.array_equal(train_obs, eval_obs)

    def test_rejects_continuous_action_space(self) -> None:
        with pytest.raises(ValueError, match="Discrete action space"):
            make_discrete_env_pair("Pendulum-v1", seed_everything(0))
