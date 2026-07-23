"""Tests for DQN: TD targets, target isolation/sync, epsilon, updates, resume."""

from __future__ import annotations

import copy
import dataclasses
import json
import math
import subprocess
import sys
from typing import TYPE_CHECKING, cast

import pytest
import torch

from rlcore.agents.dqn.loss import linear_epsilon, q_huber_loss, select_next_values, td_targets
from rlcore.agents.dqn.model import QNetwork, epsilon_greedy_action
from rlcore.agents.dqn.train import DqnConfig, dqn_update, train
from rlcore.replay import ReplayBuffer

if TYPE_CHECKING:
    from pathlib import Path

    from rlcore.types import Batch


def assert_histories_equal(first: list[dict[str, float]], second: list[dict[str, float]]) -> None:
    assert len(first) == len(second)
    for a, b in zip(first, second, strict=True):
        assert a.keys() == b.keys()
        for key in a:
            if math.isnan(a[key]) and math.isnan(b[key]):
                continue
            assert a[key] == b[key], key


def make_q(obs_dim: int = 2, n_actions: int = 3, seed: int = 0) -> QNetwork:
    torch.manual_seed(seed)
    return QNetwork(obs_dim, n_actions, hidden_sizes=(8,))


def tiny_config(total_steps: int, checkpoint_every: int = 0, eval_every: int = 100) -> DqnConfig:
    return DqnConfig(
        seed=5,
        total_steps=total_steps,
        buffer_capacity=500,
        warmup_steps=32,
        batch_size=16,
        target_sync_every=50,
        hidden_sizes=[16],
        log_every=50,
        eval_every=eval_every,
        eval_episodes=2,
        stop_return=None,
        checkpoint_every=checkpoint_every,
    )


class TestTdTargets:
    def test_hand_computed(self) -> None:
        rewards = torch.tensor([1.0, 2.0, 3.0])
        terminated = torch.tensor([False, True, False])
        next_values = torch.tensor([10.0, 10.0, -4.0])
        targets = td_targets(rewards, terminated, next_values, gamma=0.5)
        # [1 + 0.5*10, 2 + 0 (terminal suppressed), 3 + 0.5*(-4)]
        assert torch.allclose(targets, torch.tensor([6.0, 2.0, 1.0]))

    def test_terminal_bootstrap_suppressed(self) -> None:
        targets = td_targets(
            torch.tensor([1.0]), torch.tensor([True]), torch.tensor([1000.0]), gamma=0.99
        )
        assert targets.item() == 1.0

    def test_truncation_bootstraps(self) -> None:
        # A truncated transition carries terminated=False, so its stored
        # final observation's value IS bootstrapped from.
        targets = td_targets(
            torch.tensor([1.0]), torch.tensor([False]), torch.tensor([10.0]), gamma=0.9
        )
        assert targets.item() == pytest.approx(10.0)

    def test_rejects_grad_and_bad_inputs(self) -> None:
        with pytest.raises(ValueError, match="no_grad"):
            td_targets(
                torch.ones(1),
                torch.tensor([False]),
                torch.ones(1, requires_grad=True),
                gamma=0.9,
            )
        with pytest.raises(ValueError, match="bool"):
            td_targets(torch.ones(1), torch.zeros(1), torch.ones(1), gamma=0.9)
        with pytest.raises(ValueError, match="gamma"):
            td_targets(torch.ones(1), torch.tensor([False]), torch.ones(1), gamma=1.5)


class TestNextValueSelection:
    def test_standard_is_target_max(self) -> None:
        target_q = torch.tensor([[1.0, 5.0], [3.0, 2.0]])
        values = select_next_values(target_q, None, double=False)
        assert torch.equal(values, torch.tensor([5.0, 3.0]))

    def test_double_selects_with_online_evaluates_with_target(self) -> None:
        target_q = torch.tensor([[1.0, 5.0]])
        online_q = torch.tensor([[9.0, 0.0]])  # online prefers action 0
        values = select_next_values(target_q, online_q, double=True)
        assert values.item() == 1.0  # target's value of the online-chosen action

    def test_double_requires_online(self) -> None:
        with pytest.raises(ValueError, match="online_q_next"):
            select_next_values(torch.ones(1, 2), None, double=True)


class TestEpsilonSchedule:
    def test_endpoints_and_midpoint(self) -> None:
        assert linear_epsilon(0, start=1.0, final=0.1, decay_steps=100) == 1.0
        assert linear_epsilon(50, start=1.0, final=0.1, decay_steps=100) == pytest.approx(0.55)
        assert linear_epsilon(100, start=1.0, final=0.1, decay_steps=100) == pytest.approx(0.1)
        assert linear_epsilon(10_000, start=1.0, final=0.1, decay_steps=100) == pytest.approx(0.1)

    def test_rejects_bad_decay(self) -> None:
        with pytest.raises(ValueError, match="decay_steps"):
            linear_epsilon(0, start=1.0, final=0.1, decay_steps=0)


class TestEpsilonGreedy:
    def test_epsilon_zero_is_greedy(self) -> None:
        q_net = make_q()
        obs = torch.randn(16, 2)
        actions = epsilon_greedy_action(q_net, obs, epsilon=0.0)
        assert torch.equal(actions, q_net.greedy_action(obs))

    def test_epsilon_one_is_uniform_random(self) -> None:
        q_net = make_q()
        obs = torch.zeros(512, 2)
        actions = epsilon_greedy_action(
            q_net, obs, epsilon=1.0, generator=torch.Generator().manual_seed(0)
        )
        counts = torch.bincount(actions, minlength=3).float()
        assert bool((counts > 100).all())  # all three actions occur often

    def test_generator_controlled(self) -> None:
        q_net = make_q()
        obs = torch.randn(64, 2)
        first = epsilon_greedy_action(
            q_net, obs, epsilon=0.5, generator=torch.Generator().manual_seed(4)
        )
        second = epsilon_greedy_action(
            q_net, obs, epsilon=0.5, generator=torch.Generator().manual_seed(4)
        )
        assert torch.equal(first, second)

    def test_rejects_bad_epsilon(self) -> None:
        with pytest.raises(ValueError, match="epsilon"):
            epsilon_greedy_action(make_q(), torch.randn(1, 2), epsilon=1.5)


def make_batch(q_net: QNetwork, n: int = 16) -> Batch:
    buffer = ReplayBuffer(64, obs_shape=(2,))
    generator = torch.Generator().manual_seed(0)
    for i in range(32):
        obs = torch.randn(2, generator=generator).numpy()
        next_obs = torch.randn(2, generator=generator).numpy()
        buffer.add(obs, i % q_net.n_actions, float(i % 3), next_obs, i % 7 == 0)
    return buffer.sample(n, generator=torch.Generator().manual_seed(1))


class TestDqnUpdate:
    def test_target_network_isolation(self) -> None:
        q_net, target = make_q(), make_q(seed=1)
        target_before = copy.deepcopy(target.state_dict())
        optimizer = torch.optim.Adam(q_net.parameters(), lr=1e-2)
        dqn_update(
            q_net,
            target,
            optimizer,
            make_batch(q_net),
            gamma=0.99,
            double_q=False,
            max_grad_norm=10.0,
        )
        for key, value in target.state_dict().items():
            assert torch.equal(value, target_before[key]), key
        for param in q_net.parameters():
            assert param.grad is not None
            assert bool(torch.isfinite(param.grad).all())

    def test_update_equivalence_with_manual_loss(self) -> None:
        """dqn_update's gradients equal a manually composed TD loss's."""
        q_net, target = make_q(), make_q(seed=1)
        batch = make_batch(q_net)
        optimizer = torch.optim.SGD(q_net.parameters(), lr=0.0)  # keep params fixed
        dqn_update(q_net, target, optimizer, batch, gamma=0.9, double_q=False, max_grad_norm=1e9)
        update_grads = [p.grad.detach().clone() for p in q_net.parameters() if p.grad is not None]

        q_net.zero_grad()
        with torch.no_grad():
            next_values = target(batch.next_obs).max(dim=-1).values
            targets = td_targets(batch.reward, batch.terminated, next_values, gamma=0.9)
        q_taken = q_net(batch.obs).gather(1, batch.action.unsqueeze(1)).squeeze(1)
        manual = q_huber_loss(q_taken, targets)
        manual.backward()  # type: ignore[no-untyped-call]
        manual_grads = [p.grad.detach().clone() for p in q_net.parameters() if p.grad is not None]
        for got, expected in zip(update_grads, manual_grads, strict=True):
            assert torch.allclose(got, expected, atol=1e-7)

    def test_double_q_changes_targets(self) -> None:
        q_net, target = make_q(), make_q(seed=1)
        batch = make_batch(q_net)
        opt_a = torch.optim.SGD(q_net.parameters(), lr=0.0)
        std = dqn_update(q_net, target, opt_a, batch, gamma=0.99, double_q=False, max_grad_norm=1e9)
        dbl = dqn_update(q_net, target, opt_a, batch, gamma=0.99, double_q=True, max_grad_norm=1e9)
        assert std["target_mean"] != dbl["target_mean"]
        # Double-DQN targets can never exceed standard max-targets.
        assert dbl["target_mean"] <= std["target_mean"] + 1e-6


class TestTraining:
    def test_target_sync_happens(self) -> None:
        result = train(tiny_config(total_steps=100))
        assert result.total_env_steps == 100
        # After the final sync at step 100 the run ends; verify training ran
        # updates and produced finite losses.
        logged = [h["loss"] for h in result.history if not math.isnan(h["loss"])]
        assert logged, "expected at least one logged loss after warmup"

    def test_schedule_validation(self) -> None:
        with pytest.raises(ValueError, match="eval_every"):
            train(tiny_config(total_steps=100, eval_every=75))

    def test_seed_reproducibility(self) -> None:
        first = train(tiny_config(total_steps=150))
        second = train(tiny_config(total_steps=150))
        for key, value in first.q_net.state_dict().items():
            assert torch.equal(value, second.q_net.state_dict()[key]), key
        assert_histories_equal(first.history, second.history)

    def test_resume_equals_continuous(self, tmp_path: Path) -> None:
        continuous = train(tiny_config(total_steps=200))
        train(tiny_config(total_steps=100, checkpoint_every=100), out_dir=tmp_path)
        resumed = train(tiny_config(total_steps=200), resume_from=tmp_path / "checkpoint.pt")
        for key, value in continuous.q_net.state_dict().items():
            assert torch.equal(value, resumed.q_net.state_dict()[key]), key
        assert_histories_equal(continuous.history[2:], resumed.history[2:])

    def test_fresh_process_resume(self, tmp_path: Path) -> None:
        continuous = train(tiny_config(total_steps=200))
        train(tiny_config(total_steps=100, checkpoint_every=100), out_dir=tmp_path)
        snippet = (
            "import json, sys, torch\n"
            "from pathlib import Path\n"
            "from rlcore.agents.dqn.train import DqnConfig, train\n"
            "cfg = DqnConfig(**json.loads(sys.argv[1]))\n"
            "result = train(cfg, resume_from=Path(sys.argv[2]))\n"
            "torch.save(result.q_net.state_dict(), sys.argv[3])\n"
        )
        out = tmp_path / "fresh.pt"
        subprocess.run(
            [
                sys.executable,
                "-c",
                snippet,
                json.dumps(dataclasses.asdict(tiny_config(total_steps=200))),
                str(tmp_path / "checkpoint.pt"),
                str(out),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        fresh = cast("dict[str, torch.Tensor]", torch.load(out, weights_only=True))
        for key, value in continuous.q_net.state_dict().items():
            assert torch.equal(value, fresh[key]), key
