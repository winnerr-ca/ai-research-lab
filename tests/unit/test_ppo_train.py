"""Tests for the PPO update: minibatch accounting, gradients, determinism."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
import torch

from rlcore._testing import ScriptedEnv
from rlcore.agents.ppo.gae import compute_gae
from rlcore.agents.ppo.model import ActorCritic
from rlcore.agents.ppo.rollout import RolloutCollector
from rlcore.agents.ppo.train import PpoConfig, ppo_update, train

if TYPE_CHECKING:
    from pathlib import Path

    from rlcore.types import Batch


def make_model() -> ActorCritic:
    torch.manual_seed(0)
    return ActorCritic(obs_dim=2, n_actions=2, hidden_sizes=(8,))


def make_data(model: ActorCritic, steps: int = 8) -> Batch:
    collector = RolloutCollector(ScriptedEnv([1.0] * 64))
    batch = collector.collect(model, steps, generator=torch.Generator().manual_seed(0)).batch
    advantages, value_targets = compute_gae(
        batch.reward, batch.value, batch.next_value, batch.done, gamma=0.99, lam=0.95
    )
    return batch.with_fields(advantage=advantages, value_target=value_targets)


def run_update(
    model: ActorCritic, data: Batch, *, target_kl: float | None = None
) -> dict[str, float]:
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    return ppo_update(
        model,
        optimizer,
        data,
        n_epochs=2,
        minibatch_size=4,
        clip_range=0.2,
        vf_coef=0.5,
        ent_coef=0.0,
        max_grad_norm=0.5,
        normalize_advantages=True,
        target_kl=target_kl,
        generator=torch.Generator().manual_seed(0),
    )


def tiny_config(seed: int) -> PpoConfig:
    return PpoConfig(
        seed=seed,
        total_updates=2,
        n_steps=64,
        minibatch_size=32,
        n_epochs=2,
        eval_every=0,
        eval_episodes=2,
        stop_return=None,
    )


class TestPpoUpdate:
    def test_minibatch_accounting(self) -> None:
        model = make_model()
        metrics = run_update(model, make_data(model))
        # 8 samples / minibatch 4 = 2 minibatches per epoch, 2 epochs.
        assert metrics["n_minibatches"] == 4.0
        assert metrics["kl_stopped_epoch"] == -1.0

    def test_parameters_change_and_gradients_finite(self) -> None:
        model = make_model()
        before = [p.detach().clone() for p in model.parameters()]
        metrics = run_update(model, make_data(model))
        assert any(
            not torch.equal(b, p.detach()) for b, p in zip(before, model.parameters(), strict=True)
        )
        for param in model.parameters():
            assert param.grad is not None
            assert bool(torch.isfinite(param.grad).all())
        for key, value in metrics.items():
            assert torch.isfinite(torch.tensor(value)), key

    def test_gradient_norm_clipping_is_applied(self) -> None:
        """The gradients the optimizer applies are norm-bounded by max_grad_norm.

        The reported ``grad_norm`` metric is the PRE-clip norm (what
        ``clip_grad_norm_`` returns), so it may exceed the bound; the
        gradients left on the parameters after the update must not.
        """
        model = make_model()
        data = make_data(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        max_grad_norm = 1e-3
        metrics = ppo_update(
            model,
            optimizer,
            data,
            n_epochs=1,
            minibatch_size=len(data),
            clip_range=0.2,
            vf_coef=0.5,
            ent_coef=0.0,
            max_grad_norm=max_grad_norm,
            normalize_advantages=True,
            target_kl=None,
            generator=torch.Generator().manual_seed(0),
        )
        squares = [
            param.grad.pow(2).sum() for param in model.parameters() if param.grad is not None
        ]
        applied_norm = torch.sqrt(torch.stack(squares).sum())
        assert float(applied_norm) <= max_grad_norm * (1.0 + 1e-4)
        assert metrics["grad_norm"] >= float(applied_norm)  # metric is pre-clip

    def test_target_kl_stops_epochs(self) -> None:
        """A forced KL breach stops the epochs after exactly one minibatch.

        Behavior log-probs shifted far from the model's own guarantee the
        first minibatch's approximate KL exceeds any tiny target.
        """
        model = make_model()
        data = make_data(model)
        shifted = data.with_fields(logprob=data.logprob - 2.0)
        metrics = run_update(model, shifted, target_kl=1e-6)
        assert metrics["n_minibatches"] == 1.0
        assert metrics["kl_stopped_epoch"] == 0.0

    def test_metric_keys(self) -> None:
        model = make_model()
        metrics = run_update(model, make_data(model))
        assert {
            "policy_loss",
            "value_loss",
            "entropy",
            "approx_kl",
            "clip_fraction",
            "grad_norm",
            "n_minibatches",
            "kl_stopped_epoch",
        } <= set(metrics)


class TestSeedControlledTraining:
    def test_same_seed_reproduces_parameters(self) -> None:
        first = train(tiny_config(seed=11))
        second = train(tiny_config(seed=11))
        for key, value in first.model.state_dict().items():
            assert torch.equal(value, second.model.state_dict()[key]), key

    def test_different_seeds_diverge(self) -> None:
        first = train(tiny_config(seed=11))
        second = train(tiny_config(seed=12))
        assert any(
            not torch.equal(value, second.model.state_dict()[key])
            for key, value in first.model.state_dict().items()
        )


class TestTrainOutputs:
    def test_history_and_counters(self) -> None:
        result = train(tiny_config(seed=5))
        assert len(result.history) == 2
        assert result.total_env_steps == 2 * 64
        assert result.total_env_steps == int(result.history[-1]["env_steps"])
        assert not result.stopped_early

    def test_writes_run_directory(self, tmp_path: Path) -> None:
        result = train(tiny_config(seed=5), out_dir=tmp_path)
        assert (tmp_path / "config.yaml").exists()
        lines = (tmp_path / "metrics.jsonl").read_text().strip().splitlines()
        assert len(lines) == len(result.history)
        payload = json.loads((tmp_path / "result.json").read_text())
        assert payload["total_env_steps"] == result.total_env_steps

    def test_rejects_wrong_spaces(self) -> None:
        with pytest.raises(ValueError, match="Discrete action space"):
            train(PpoConfig(env_id="Pendulum-v1", total_updates=1))
