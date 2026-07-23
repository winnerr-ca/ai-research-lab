"""Unit tests for vectorized collection: exact per-env boundary handling."""

from __future__ import annotations

import functools

import numpy as np
import pytest
import torch
from gymnasium.vector import AutoresetMode, SyncVectorEnv

from rlcore._testing import ScriptedEnv
from rlcore.agents.ppo.gae import compute_gae
from rlcore.agents.ppo.model import ActorCritic
from rlcore.agents.ppo.vector_rollout import (
    VectorRolloutCollector,
    make_vector_train_env,
)
from rlcore.utils.seeding import seed_everything


def _scripted_vector_env() -> SyncVectorEnv:
    """Env 0 terminates every 2 steps; env 1 truncates every 3 steps."""
    return SyncVectorEnv(
        [
            functools.partial(ScriptedEnv, [1.0, 2.0], terminate=True),
            functools.partial(ScriptedEnv, [5.0, 6.0, 7.0], terminate=False),
        ],
        autoreset_mode=AutoresetMode.SAME_STEP,
    )


@pytest.fixture
def model() -> ActorCritic:
    seed_everything(0)
    return ActorCritic(2, 2)


def _obs(step_index: float) -> torch.Tensor:
    return torch.full((2,), step_index, dtype=torch.float32)


def test_boundary_next_values_exact(model: ActorCritic) -> None:
    env = _scripted_vector_env()
    env.reset(seed=[0, 1])
    collector = VectorRolloutCollector(env)
    rollout = collector.collect(model, 4)

    terminateds = rollout.per_env["done"] & rollout.batch.terminated.reshape(4, 2)
    # Env 0 terminates at rows 1 and 3; env 1 truncates at row 2.
    assert rollout.batch.terminated.reshape(4, 2)[:, 0].tolist() == [False, True, False, True]
    assert rollout.batch.truncated.reshape(4, 2)[:, 1].tolist() == [False, False, True, False]
    assert terminateds[:, 0].tolist() == [False, True, False, True]

    next_values = rollout.per_env["next_value"]
    values = rollout.per_env["value"]
    with torch.no_grad():
        v_final_env1 = model.value(_obs(3.0).unsqueeze(0)).item()
        v_tail_env1 = model.value(_obs(1.0).unsqueeze(0)).item()

    # Terminated rows: bootstrap suppressed exactly.
    assert next_values[1, 0].item() == 0.0
    assert next_values[3, 0].item() == 0.0
    # Truncated row: bootstrap from the true final obs (step index 3),
    # not the reset obs (step index 0) that SAME_STEP returns.
    assert next_values[2, 1].item() == pytest.approx(v_final_env1)
    # Non-boundary rows: V(s_{t+1}) from the stored next row.
    assert next_values[0, 0].item() == pytest.approx(values[1, 0].item())
    assert next_values[0, 1].item() == pytest.approx(values[1, 1].item())
    assert next_values[1, 1].item() == pytest.approx(values[2, 1].item())
    assert next_values[2, 0].item() == pytest.approx(values[3, 0].item())
    # Horizon tail (no boundary): V(current obs) — env 1 restarted at row 3,
    # so its current obs after the rollout has step index 1.
    assert next_values[3, 1].item() == pytest.approx(v_tail_env1)
    env.close()


def test_same_step_reset_obs_stored_after_done(model: ActorCritic) -> None:
    env = _scripted_vector_env()
    env.reset(seed=[0, 1])
    collector = VectorRolloutCollector(env)
    rollout = collector.collect(model, 4)
    obs = rollout.batch.obs.reshape(4, 2, 2)
    # Env 0: 0,1 | reset 0,1 ; env 1: 0,1,2 | reset 0.
    assert obs[:, 0, 0].tolist() == [0.0, 1.0, 0.0, 1.0]
    assert obs[:, 1, 0].tolist() == [0.0, 1.0, 2.0, 0.0]
    env.close()


def test_completed_returns_pooled_across_envs(model: ActorCritic) -> None:
    env = _scripted_vector_env()
    env.reset(seed=[0, 1])
    collector = VectorRolloutCollector(env)
    rollout = collector.collect(model, 4)
    # Env 0 finished two episodes (3.0 each); env 1 one episode (18.0).
    assert sorted(rollout.completed_returns) == [3.0, 3.0, 18.0]
    env.close()


def test_collect_is_stateful_across_calls(model: ActorCritic) -> None:
    env = _scripted_vector_env()
    env.reset(seed=[0, 1])
    collector = VectorRolloutCollector(env)
    collector.collect(model, 2)
    rollout = collector.collect(model, 2)
    obs = rollout.batch.obs.reshape(2, 2, 2)
    # Env 0 restarted after row 1 of the first call: continues 0, 1.
    assert obs[:, 0, 0].tolist() == [0.0, 1.0]
    # Env 1 continues its first episode: rows 2 then reset 0.
    assert obs[:, 1, 0].tolist() == [2.0, 0.0]
    env.close()


def test_flatten_order_matches_per_env_columns(model: ActorCritic) -> None:
    env = _scripted_vector_env()
    env.reset(seed=[0, 1])
    collector = VectorRolloutCollector(env)
    rollout = collector.collect(model, 3)
    rewards = rollout.per_env["reward"]
    assert rollout.batch.reward.tolist() == rewards.reshape(-1).tolist()
    # Row-major flatten: step-major, env-minor.
    assert rollout.batch.reward[0].item() == rewards[0, 0].item()
    assert rollout.batch.reward[1].item() == rewards[0, 1].item()
    env.close()


def test_per_column_gae_never_crosses_columns(model: ActorCritic) -> None:
    env = _scripted_vector_env()
    env.reset(seed=[0, 1])
    collector = VectorRolloutCollector(env)
    rollout = collector.collect(model, 4)
    per_env = rollout.per_env
    # Column-wise GAE equals single-env GAE applied to that column alone.
    for col in range(2):
        adv_col, targets_col = compute_gae(
            per_env["reward"][:, col],
            per_env["value"][:, col],
            per_env["next_value"][:, col],
            per_env["done"][:, col],
            gamma=0.99,
            lam=0.95,
        )
        assert adv_col.shape == (4,)
        assert torch.isfinite(adv_col).all()
        assert torch.isfinite(targets_col).all()
        # The recursion must break at boundaries: at a done row the advantage
        # is exactly the one-step TD error.
        done_rows = per_env["done"][:, col].nonzero().flatten().tolist()
        for t in done_rows:
            delta = (
                per_env["reward"][t, col]
                + 0.99 * per_env["next_value"][t, col]
                - per_env["value"][t, col]
            )
            assert adv_col[t].item() == pytest.approx(delta.item(), abs=1e-6)
    env.close()


def test_make_vector_train_env_rejects_single_env() -> None:
    rng = seed_everything(0)
    with pytest.raises(ValueError, match="n_envs"):
        make_vector_train_env("CartPole-v1", 1, rng)


def test_child_seeding_is_deterministic_and_distinct() -> None:
    rng_a = seed_everything(7)
    env_a = make_vector_train_env("CartPole-v1", 3, rng_a)
    rng_b = seed_everything(7)
    env_b = make_vector_train_env("CartPole-v1", 3, rng_b)
    reset_a: tuple[np.ndarray, dict[str, object]] = env_a.reset(
        seed=[rng_a.child_seed(10 + i) for i in range(3)]
    )
    reset_b: tuple[np.ndarray, dict[str, object]] = env_b.reset(
        seed=[rng_b.child_seed(10 + i) for i in range(3)]
    )
    obs_a, obs_b = reset_a[0], reset_b[0]
    assert np.allclose(obs_a, obs_b)
    # Distinct child seeds: the three envs do not start identically.
    assert not (np.allclose(obs_a[0], obs_a[1]) and np.allclose(obs_a[1], obs_a[2]))
    env_a.close()
    env_b.close()


def test_state_restore_roundtrip(model: ActorCritic) -> None:
    env = _scripted_vector_env()
    env.reset(seed=[0, 1])
    collector = VectorRolloutCollector(env)
    collector.collect(model, 2)
    state = collector.state()
    env2 = _scripted_vector_env()
    env2.reset(seed=[0, 1])
    restored = VectorRolloutCollector(env2)
    restored.restore_state(state)
    assert np.allclose(restored.state()["obs"], state["obs"])
    assert np.allclose(restored.state()["episode_returns"], state["episode_returns"])
    env.close()
    env2.close()


def test_collect_rejects_bad_n_steps(model: ActorCritic) -> None:
    env = _scripted_vector_env()
    env.reset(seed=[0, 1])
    collector = VectorRolloutCollector(env)
    with pytest.raises(ValueError, match="n_steps"):
        collector.collect(model, 0)
    env.close()
