"""Tests for SAC: squashed log-prob, twin targets, Polyak, losses, resume."""

from __future__ import annotations

import copy
import json
import math
import subprocess
import sys
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest
import torch

from rlcore._testing import ContinuousScriptedEnv
from rlcore.agents.sac.loss import (
    actor_loss,
    alpha_loss,
    critic_loss,
    critic_targets,
    polyak_update,
)
from rlcore.agents.sac.model import SquashedGaussianActor
from rlcore.agents.sac.train import SacConfig, train
from rlcore.evaluation import evaluate_policy

if TYPE_CHECKING:
    from pathlib import Path


def make_actor(obs_dim: int = 3, act_dim: int = 2, seed: int = 0) -> SquashedGaussianActor:
    torch.manual_seed(seed)
    return SquashedGaussianActor(
        obs_dim,
        act_dim,
        action_low=torch.tensor([-2.0, -1.0]),
        action_high=torch.tensor([2.0, 3.0]),
        hidden_sizes=(16,),
    )


def tiny_config(total_steps: int, checkpoint_every: int = 0, auto_alpha: bool = True) -> SacConfig:
    return SacConfig(
        seed=5,
        total_steps=total_steps,
        buffer_capacity=500,
        warmup_steps=32,
        batch_size=16,
        hidden_sizes=[16],
        auto_alpha=auto_alpha,
        log_every=50,
        eval_every=100,
        eval_episodes=2,
        stop_return=None,
        checkpoint_every=checkpoint_every,
    )


class TestActor:
    def test_actions_respect_bounds(self) -> None:
        actor = make_actor()
        obs = torch.randn(256, 3) * 10.0  # extreme inputs push tanh to limits
        action, _ = actor.sample(obs, generator=torch.Generator().manual_seed(0))
        low = torch.tensor([-2.0, -1.0])
        high = torch.tensor([2.0, 3.0])
        assert bool((action >= low - 1e-6).all())
        assert bool((action <= high + 1e-6).all())

    def test_log_prob_matches_torch_distributions(self) -> None:
        """Analytic log-prob vs an independent TransformedDistribution stack."""
        actor = make_actor()
        obs = torch.randn(64, 3)
        action, log_prob = actor.sample(obs, generator=torch.Generator().manual_seed(1))

        mu, log_std = actor._mu_log_std(obs)
        base = torch.distributions.Normal(mu, log_std.exp())
        transforms = [
            torch.distributions.transforms.TanhTransform(cache_size=1),
            torch.distributions.transforms.AffineTransform(
                loc=actor.action_bias, scale=actor.action_scale
            ),
        ]
        dist = torch.distributions.TransformedDistribution(base, transforms)
        clamped = action.detach().clamp(-4.9999, 4.9999)
        reference = dist.log_prob(clamped).sum(dim=-1)  # type: ignore[no-untyped-call]
        assert torch.allclose(log_prob.detach(), reference, atol=1e-4)

    def test_log_prob_finite_near_action_limits(self) -> None:
        actor = make_actor()
        with torch.no_grad():  # force an extreme, confident policy
            final = actor.net[-1]
            assert isinstance(final, torch.nn.Linear)
            final.bias[:2] = 50.0  # huge mu -> tanh saturated
            final.bias[2:] = -10.0  # tiny std
        obs = torch.randn(32, 3)
        _action, log_prob = actor.sample(obs, generator=torch.Generator().manual_seed(0))
        assert bool(torch.isfinite(log_prob).all())
        loss = log_prob.mean()
        loss.backward()  # type: ignore[no-untyped-call]
        for param in actor.parameters():
            assert param.grad is None or bool(torch.isfinite(param.grad).all())

    def test_reparameterized_sample_is_differentiable(self) -> None:
        actor = make_actor()
        action, _ = actor.sample(torch.randn(4, 3), generator=torch.Generator().manual_seed(0))
        assert action.requires_grad

    def test_deterministic_action_within_bounds_no_rng(self) -> None:
        actor = make_actor()
        generator = torch.Generator().manual_seed(0)
        state_before = generator.get_state()
        action = actor.deterministic_action(torch.randn(8, 3))
        assert torch.equal(generator.get_state(), state_before)
        assert bool((action >= torch.tensor([-2.0, -1.0])).all())
        assert bool((action <= torch.tensor([2.0, 3.0])).all())

    def test_rejects_bad_bounds(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            SquashedGaussianActor(
                2,
                1,
                action_low=torch.tensor([-torch.inf]),
                action_high=torch.tensor([1.0]),
            )
        with pytest.raises(ValueError, match="exceed"):
            SquashedGaussianActor(
                2, 1, action_low=torch.tensor([1.0]), action_high=torch.tensor([1.0])
            )


class TestCriticTargets:
    def test_hand_computed_twin_min_soft_target(self) -> None:
        targets = critic_targets(
            rewards=torch.tensor([1.0, 1.0]),
            terminated=torch.tensor([False, True]),
            next_q_min=torch.tensor([4.0, 4.0]),
            next_log_prob=torch.tensor([-2.0, -2.0]),
            gamma=0.5,
            alpha=0.1,
        )
        # non-terminal: 1 + 0.5*(4 - 0.1*(-2)) = 1 + 0.5*4.2 = 3.1
        # terminal: bootstrap suppressed -> 1.0
        assert torch.allclose(targets, torch.tensor([3.1, 1.0]))

    def test_truncation_bootstraps(self) -> None:
        targets = critic_targets(
            rewards=torch.tensor([0.0]),
            terminated=torch.tensor([False]),  # truncated transitions carry False
            next_q_min=torch.tensor([10.0]),
            next_log_prob=torch.tensor([0.0]),
            gamma=1.0,
            alpha=0.0,
        )
        assert targets.item() == 10.0

    def test_rejects_grad_inputs(self) -> None:
        with pytest.raises(ValueError, match="no_grad"):
            critic_targets(
                torch.ones(1),
                torch.tensor([False]),
                torch.ones(1, requires_grad=True),
                torch.zeros(1),
                gamma=0.9,
                alpha=0.1,
            )


class TestLosses:
    def test_critic_loss_hand_computed(self) -> None:
        loss = critic_loss(torch.tensor([1.0]), torch.tensor([3.0]), torch.tensor([2.0]))
        assert loss.item() == pytest.approx(1.0 + 1.0)

    def test_actor_loss_hand_computed(self) -> None:
        loss = actor_loss(torch.tensor([2.0, 4.0]), torch.tensor([-1.0, -3.0]), alpha=0.5)
        # mean(0.5*[-1,-3] - [2,4]) = mean([-2.5, -5.5]) = -4.0
        assert loss.item() == pytest.approx(-4.0)

    def test_alpha_loss_gradient_direction(self) -> None:
        """Entropy below target -> gradient increases log_alpha, and vice versa."""
        target_entropy = -2.0

        log_alpha = torch.zeros((), requires_grad=True)
        low_entropy_log_prob = torch.full((8,), 5.0)  # entropy far below target
        alpha_loss(log_alpha, low_entropy_log_prob, target_entropy=target_entropy).backward()  # type: ignore[no-untyped-call]
        assert log_alpha.grad is not None
        assert log_alpha.grad.item() < 0  # descending increases log_alpha

        log_alpha2 = torch.zeros((), requires_grad=True)
        high_entropy_log_prob = torch.full((8,), -50.0)
        alpha_loss(log_alpha2, high_entropy_log_prob, target_entropy=target_entropy).backward()  # type: ignore[no-untyped-call]
        assert log_alpha2.grad is not None
        assert log_alpha2.grad.item() > 0


class TestPolyak:
    def test_hand_computed_interpolation(self) -> None:
        source = torch.nn.Linear(2, 2, bias=False)
        target = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            source.weight.fill_(1.0)
            target.weight.fill_(0.0)
        polyak_update(source, target, tau=0.25)
        assert torch.allclose(target.weight, torch.full((2, 2), 0.25))

    def test_tau_one_copies(self) -> None:
        source, target = torch.nn.Linear(3, 1), torch.nn.Linear(3, 1)
        polyak_update(source, target, tau=1.0)
        for a, b in zip(source.parameters(), target.parameters(), strict=True):
            assert torch.equal(a, b)

    def test_matches_sb3(self) -> None:
        sb3_utils = pytest.importorskip("stable_baselines3.common.utils")
        torch.manual_seed(0)
        source, ours, theirs = (torch.nn.Linear(4, 4) for _ in range(3))
        theirs.load_state_dict(ours.state_dict())
        polyak_update(source, ours, tau=0.05)
        sb3_utils.polyak_update(source.parameters(), theirs.parameters(), tau=0.05)
        for a, b in zip(ours.parameters(), theirs.parameters(), strict=True):
            assert torch.allclose(a, b, atol=1e-7)

    def test_rejects_bad_tau(self) -> None:
        with pytest.raises(ValueError, match="tau"):
            polyak_update(torch.nn.Linear(1, 1), torch.nn.Linear(1, 1), tau=0.0)


class TestTraining:
    def test_seed_reproducibility_and_finite_losses(self) -> None:
        first = train(tiny_config(150))
        second = train(tiny_config(150))
        for key, value in first.actor.state_dict().items():
            assert torch.equal(value, second.actor.state_dict()[key]), key
        losses = [h["critic_loss"] for h in first.history if not math.isnan(h["critic_loss"])]
        assert losses and all(math.isfinite(v) for v in losses)

    def test_fixed_alpha_mode(self) -> None:
        result = train(tiny_config(120, auto_alpha=False))
        alphas = [h["alpha"] for h in result.history if not math.isnan(h["alpha"])]
        assert alphas
        assert all(a == pytest.approx(0.2) for a in alphas)
        assert all(math.isnan(h["alpha_loss"]) for h in result.history)

    def test_auto_alpha_changes(self) -> None:
        result = train(tiny_config(200))
        alphas = [h["alpha"] for h in result.history if not math.isnan(h["alpha"])]
        assert len(set(alphas)) > 1

    def test_resume_equals_continuous(self, tmp_path: Path) -> None:
        continuous = train(tiny_config(200))
        train(tiny_config(100, checkpoint_every=100), out_dir=tmp_path)
        resumed = train(tiny_config(200), resume_from=tmp_path / "checkpoint.pt")
        for key, value in continuous.actor.state_dict().items():
            assert torch.equal(value, resumed.actor.state_dict()[key]), key

    def test_fresh_process_resume(self, tmp_path: Path) -> None:
        continuous = train(tiny_config(200))
        train(tiny_config(100, checkpoint_every=100), out_dir=tmp_path)
        snippet = (
            "import json, sys, torch\n"
            "from pathlib import Path\n"
            "from rlcore.agents.sac.train import SacConfig, train\n"
            "cfg = SacConfig(**json.loads(sys.argv[1]))\n"
            "result = train(cfg, resume_from=Path(sys.argv[2]))\n"
            "torch.save(result.actor.state_dict(), sys.argv[3])\n"
        )
        import dataclasses

        out = tmp_path / "fresh.pt"
        subprocess.run(
            [
                sys.executable,
                "-c",
                snippet,
                json.dumps(dataclasses.asdict(tiny_config(200))),
                str(tmp_path / "checkpoint.pt"),
                str(out),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        fresh = cast("dict[str, torch.Tensor]", torch.load(out, weights_only=True))
        for key, value in continuous.actor.state_dict().items():
            assert torch.equal(value, fresh[key]), key

    def test_evaluation_side_effect_freedom(self) -> None:
        result = train(tiny_config(100))
        actor = result.actor
        before = copy.deepcopy(actor.state_dict())
        env = ContinuousScriptedEnv([1.0, 1.0], obs_dim=3, act_dim=1)  # Pendulum obs_dim
        evaluate_policy(env, lambda obs: actor.deterministic_action(obs), episodes=2)
        for key, value in actor.state_dict().items():
            assert torch.equal(value, before[key]), key


class TestContinuousEvaluation:
    def test_continuous_actions_reach_env(self) -> None:
        env = ContinuousScriptedEnv([1.0, 2.0], act_dim=2)
        stats = evaluate_policy(
            env,
            lambda obs: torch.tensor([[0.5, -0.5]]),
            episodes=1,
        )
        assert stats.episode_returns == (3.0,)
        assert np.allclose(env.received_actions[0], [0.5, -0.5])
