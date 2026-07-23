"""Unit tests for device resolution and safe CUDA fallback."""

from __future__ import annotations

import logging

import pytest
import torch

from rlcore.utils.device import resolve_device


def test_cpu_is_cpu() -> None:
    assert resolve_device("cpu") == torch.device("cpu")


def test_auto_matches_availability() -> None:
    expected = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    assert resolve_device("auto") == expected


def test_cuda_falls_back_with_warning_when_unavailable(
    caplog: pytest.LogCaptureFixture,
) -> None:
    if torch.cuda.is_available():  # pragma: no cover - GPU CI only
        pytest.skip("CUDA available; fallback path not reachable.")
    with caplog.at_level(logging.WARNING, logger="rlcore.utils.device"):
        device = resolve_device("cuda")
    assert device == torch.device("cpu")
    assert any("falling back to CPU" in message for message in caplog.messages)


def test_cuda_index_spec_accepted() -> None:
    device = resolve_device("cuda:1")
    assert device.type in ("cuda", "cpu")


def test_unknown_spec_rejected() -> None:
    with pytest.raises(ValueError, match="device spec"):
        resolve_device("tpu")
