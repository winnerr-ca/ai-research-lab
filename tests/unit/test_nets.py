"""Tests for shared network helpers (M3 extraction)."""

from __future__ import annotations

import math

import torch
from torch import nn

from rlcore.nets import build_mlp, entropy_from_log_probs, gather_log_prob


class TestBuildMlp:
    def test_structure(self) -> None:
        net = build_mlp(4, (8, 8), 3, final_gain=0.01)
        kinds = [type(m) for m in net]
        assert kinds == [nn.Linear, nn.Tanh, nn.Linear, nn.Tanh, nn.Linear]
        assert net[0].in_features == 4
        assert net[-1].out_features == 3

    def test_empty_hidden_is_single_linear(self) -> None:
        net = build_mlp(4, (), 2, final_gain=1.0)
        assert [type(m) for m in net] == [nn.Linear]

    def test_hidden_rows_are_orthonormal_scaled(self) -> None:
        torch.manual_seed(0)
        first = build_mlp(16, (8,), 2, final_gain=0.01)[0]
        assert isinstance(first, nn.Linear)
        weight = first.weight  # [8, 16]: rows orthogonal with gain sqrt(2)
        gram = weight @ weight.T
        expected = 2.0 * torch.eye(8)
        assert torch.allclose(gram, expected, atol=1e-5)

    def test_final_gain_scales_output_layer(self) -> None:
        torch.manual_seed(0)
        small = build_mlp(8, (), 8, final_gain=0.01)[0]
        torch.manual_seed(0)
        unit = build_mlp(8, (), 8, final_gain=1.0)[0]
        assert isinstance(small, nn.Linear) and isinstance(unit, nn.Linear)
        assert torch.allclose(small.weight * 100.0, unit.weight, atol=1e-5)

    def test_biases_zero(self) -> None:
        net = build_mlp(4, (8,), 2, final_gain=0.01)
        for module in net:
            if isinstance(module, nn.Linear):
                assert bool((module.bias == 0).all())

    def test_seeded_construction_is_deterministic(self) -> None:
        torch.manual_seed(3)
        first = build_mlp(4, (8, 8), 2, final_gain=0.01)
        torch.manual_seed(3)
        second = build_mlp(4, (8, 8), 2, final_gain=0.01)
        for a, b in zip(first.parameters(), second.parameters(), strict=True):
            assert torch.equal(a, b)


class TestCategoricalHelpers:
    def test_gather_log_prob_matches_manual(self) -> None:
        log_probs = torch.randn(5, 3).log_softmax(dim=-1)
        action = torch.tensor([0, 2, 1, 2, 0])
        expected = torch.stack([log_probs[i, a] for i, a in enumerate(action)])
        assert torch.allclose(gather_log_prob(log_probs, action), expected, atol=1e-7)

    def test_entropy_of_uniform_is_log_n(self) -> None:
        log_probs = torch.zeros(4, 5).log_softmax(dim=-1)
        assert torch.allclose(
            entropy_from_log_probs(log_probs), torch.full((4,), math.log(5.0)), atol=1e-6
        )

    def test_entropy_of_deterministic_is_zero(self) -> None:
        logits = torch.full((2, 3), -1000.0)
        logits[:, 0] = 1000.0
        entropy = entropy_from_log_probs(logits.log_softmax(dim=-1))
        assert torch.allclose(entropy, torch.zeros(2), atol=1e-5)

    def test_helpers_preserve_grad(self) -> None:
        logits = torch.randn(3, 4, requires_grad=True)
        log_probs = logits.log_softmax(dim=-1)
        out = gather_log_prob(log_probs, torch.tensor([0, 1, 2]))
        assert out.requires_grad
        assert entropy_from_log_probs(log_probs).requires_grad
