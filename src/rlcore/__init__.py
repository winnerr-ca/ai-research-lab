"""rlcore: a modular reinforcement learning research platform.

Legible algorithm code on shared, tested infrastructure — see
``docs/ARCHITECTURE.md`` for the design and ``docs/ROADMAP.md`` for the
build plan.
"""

from rlcore.types import Batch, PolicyOutput

__all__ = ["Batch", "PolicyOutput", "__version__"]

__version__ = "0.0.1"
