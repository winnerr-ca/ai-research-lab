"""Local research dashboard (``rlcore-lab``).

A localhost-only web interface over the platform's existing artifacts:
browse and compare recorded runs, launch new training runs (including
user-added algorithm packages under ``rlcore.agents``), watch launched
jobs, and list research-store entities. The dashboard adds no numbers of
its own — every figure it shows comes from run-directory files or from
:mod:`rlcore.stats`, and every launched run writes a standard run
directory attributable to a commit, config, and seed.
"""

from rlcore.lab.data import compare_runs, discover_runs, load_metrics, resolve_run_dir
from rlcore.lab.launch import JobManager, algo_fields, discover_algos, validate_launch

__all__ = [
    "JobManager",
    "algo_fields",
    "compare_runs",
    "discover_algos",
    "discover_runs",
    "load_metrics",
    "resolve_run_dir",
    "validate_launch",
]
