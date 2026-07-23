"""Device resolution with safe CUDA fallback.

Policy: ``"auto"`` selects CUDA when genuinely available, otherwise CPU;
requesting ``"cuda"`` on a machine without it falls back to CPU with a
logged warning rather than crashing — a queued cloud job that lands on the
wrong node type should degrade, not die. CPU remains the platform's tested
reproducibility bar (ARCHITECTURE §3.4); CUDA determinism is best-effort
per PyTorch's own guarantees.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


def resolve_device(spec: str) -> torch.device:
    """Resolve a device spec (``"auto"``/``"cpu"``/``"cuda"``/``"cuda:N"``).

    Args:
        spec: Requested device.

    Returns:
        The resolved device; CUDA requests fall back to CPU (with a
        warning) when CUDA is unavailable.

    Raises:
        ValueError: For specs other than auto/cpu/cuda[:N].
    """
    if spec == "auto":
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if spec == "cpu":
        return torch.device("cpu")
    if spec == "cuda" or spec.startswith("cuda:"):
        if torch.cuda.is_available():
            return torch.device(spec)
        logger.warning("CUDA requested (%s) but unavailable; falling back to CPU.", spec)
        return torch.device("cpu")
    raise ValueError(f"Unknown device spec {spec!r}; expected 'auto', 'cpu', or 'cuda[:N]'.")
