"""Algorithm discovery and job launching for ``rlcore-lab``.

Algorithms are discovered, not hard-coded: any package under
``rlcore.agents`` whose ``train`` module holds exactly one ``*Config``
dataclass and a ``train`` callable is launchable — so a user-added
variant package appears in the dashboard without touching this module.

Launches follow the benchmark runner's isolation pattern
(:mod:`rlcore.benchmark`): one subprocess per run, output captured to
files inside the run directory, failures recorded rather than raised.
Every launched run writes the standard run directory, so anything
started from the dashboard is as attributable as a CLI run.
"""

from __future__ import annotations

import dataclasses
import importlib
import json
import logging
import pkgutil
import re
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import rlcore.agents

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

logger = logging.getLogger(__name__)

_ALGO_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")

#: Stderr tail length kept in failed-job records (same bound as benchmarks).
_STDERR_TAIL = 2000

# Mirrors the benchmark child snippet, generalized: the algorithm's train
# module is imported dynamically so user-added agent packages launch too.
_CHILD_SNIPPET = """
import dataclasses, importlib, json, sys
from pathlib import Path
algo, overrides, out_dir = sys.argv[1], json.loads(sys.argv[2]), Path(sys.argv[3])
mod = importlib.import_module(f"rlcore.agents.{algo}.train")
configs = [
    obj
    for name in dir(mod)
    if name.endswith("Config")
    and dataclasses.is_dataclass(obj := getattr(mod, name))
    and isinstance(obj, type)
    and obj.__module__ == mod.__name__
]
if len(configs) != 1:
    raise SystemExit(f"expected one *Config dataclass in {mod.__name__}, found {len(configs)}")
mod.train(configs[0](**overrides), out_dir=out_dir)
"""


@dataclass(frozen=True)
class AlgoSpec:
    """One launchable algorithm package under ``rlcore.agents``.

    Attributes:
        name: Package name (also the ``algo=`` identifier).
        config_cls: The package's single ``*Config`` dataclass.
        train_fn: The package's ``train`` callable (used for discovery
            validation; launches run in a subprocess, not through this
            reference).
    """

    name: str
    config_cls: type[Any]
    train_fn: Callable[..., Any]


def _spec_for(name: str) -> AlgoSpec | None:
    """Build an :class:`AlgoSpec` for one agents subpackage, or ``None``."""
    try:
        module = importlib.import_module(f"rlcore.agents.{name}.train")
    except ImportError as error:
        logger.warning("agents package %r has no importable train module: %s", name, error)
        return None
    configs = [
        obj
        for attr in dir(module)
        if attr.endswith("Config")
        and dataclasses.is_dataclass(obj := getattr(module, attr))
        and isinstance(obj, type)
        and obj.__module__ == module.__name__
    ]
    train_fn = getattr(module, "train", None)
    if len(configs) != 1 or not callable(train_fn):
        logger.warning(
            "agents package %r skipped: needs exactly one *Config dataclass and a "
            "train callable (found %d configs).",
            name,
            len(configs),
        )
        return None
    return AlgoSpec(name=name, config_cls=configs[0], train_fn=train_fn)


def discover_algos() -> dict[str, AlgoSpec]:
    """Discover launchable algorithm packages under ``rlcore.agents``."""
    specs: dict[str, AlgoSpec] = {}
    for info in pkgutil.iter_modules(rlcore.agents.__path__):
        if not info.ispkg or not _ALGO_NAME_RE.match(info.name):
            continue
        spec = _spec_for(info.name)
        if spec is not None:
            specs[info.name] = spec
    return specs


def _json_safe(value: Any) -> Any:  # noqa: ANN401 - defaults are heterogeneous by design
    """Return ``value`` if strictly JSON-serializable, else its ``repr``.

    ``allow_nan=False`` matters: Python would happily emit ``NaN`` (e.g.
    SAC's auto ``target_entropy`` sentinel), which browsers reject as
    invalid JSON.
    """
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError):
        return repr(value)
    return value


def algo_fields(spec: AlgoSpec) -> list[dict[str, Any]]:
    """Describe a config dataclass's fields for the launch form.

    Returns:
        One ``{"name", "type", "default"}`` dict per field; fields without
        a default report ``"default": None`` and ``"required": True``.
    """
    described: list[dict[str, Any]] = []
    for field in dataclasses.fields(spec.config_cls):
        entry: dict[str, Any] = {"name": field.name, "type": str(field.type)}
        if field.default is not dataclasses.MISSING:
            entry["default"] = _json_safe(field.default)
        elif field.default_factory is not dataclasses.MISSING:
            entry["default"] = _json_safe(field.default_factory())
        else:
            entry["default"] = None
            entry["required"] = True
        described.append(entry)
    return described


def validate_launch(algos: dict[str, AlgoSpec], algo: str, overrides: dict[str, Any]) -> AlgoSpec:
    """Validate a launch request against the discovered algorithms.

    Args:
        algos: Result of :func:`discover_algos`.
        algo: Requested algorithm name.
        overrides: Config-field overrides from the client.

    Returns:
        The validated spec.

    Raises:
        ValueError: On unknown algorithms or override keys — the launch
            endpoint must never pass unvetted input to a subprocess.
    """
    if not _ALGO_NAME_RE.match(algo) or algo not in algos:
        raise ValueError(f"unknown algorithm {algo!r}; discovered: {sorted(algos)}")
    spec = algos[algo]
    known = {field.name for field in dataclasses.fields(spec.config_cls)}
    unknown = sorted(set(overrides) - known)
    if unknown:
        raise ValueError(f"unknown override keys for {algo!r}: {unknown}")
    return spec


@dataclass
class _Job:
    """Internal record for one launched subprocess."""

    job_id: str
    algo: str
    overrides: dict[str, Any]
    run_dir: Path
    process: subprocess.Popen[bytes]
    started: float
    started_at: str


class JobManager:
    """Launches and tracks training subprocesses for the dashboard.

    Jobs live for the server process's lifetime; the durable record is
    the run directory each job writes (plus ``lab-launch.json`` /
    ``launch-stderr.log`` written here).
    """

    def __init__(self, out_root: Path) -> None:
        """Create a manager writing run directories under ``out_root``."""
        self._out_root = out_root
        self._jobs: dict[str, _Job] = {}
        self._lock = threading.Lock()

    def launch(self, spec: AlgoSpec, overrides: dict[str, Any]) -> dict[str, Any]:
        """Start one training run in a subprocess; return its job snapshot.

        The run directory is created up front and records the launch
        request (``lab-launch.json``) so even an immediately-crashing
        job leaves a trace.
        """
        job_id = uuid.uuid4().hex[:12]
        seed = overrides.get("seed", "d")
        stamp = datetime.now(tz=UTC).strftime("%Y%m%d-%H%M%S")
        run_dir = self._out_root / f"{stamp}-{spec.name}-s{seed}-{job_id}"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "lab-launch.json").write_text(
            json.dumps({"algo": spec.name, "overrides": overrides}, indent=2)
        )
        stderr_path = run_dir / "launch-stderr.log"
        with stderr_path.open("wb") as stderr_file:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    _CHILD_SNIPPET,
                    spec.name,
                    json.dumps(overrides),
                    str(run_dir),
                ],
                stdout=subprocess.DEVNULL,
                stderr=stderr_file,
            )
        job = _Job(
            job_id=job_id,
            algo=spec.name,
            overrides=overrides,
            run_dir=run_dir,
            process=process,
            started=time.perf_counter(),
            started_at=datetime.now(tz=UTC).isoformat(),
        )
        with self._lock:
            self._jobs[job_id] = job
        logger.info("launched job %s: %s -> %s", job_id, spec.name, run_dir)
        return self._snapshot(job)

    def _snapshot(self, job: _Job) -> dict[str, Any]:
        """Point-in-time status for one job (never raises)."""
        returncode = job.process.poll()
        snapshot: dict[str, Any] = {
            "job_id": job.job_id,
            "algo": job.algo,
            "overrides": job.overrides,
            "run_dir": str(job.run_dir),
            "started_at": job.started_at,
            "wall_time_s": round(time.perf_counter() - job.started, 1),
        }
        if returncode is None:
            snapshot["status"] = "running"
        elif returncode == 0:
            snapshot["status"] = "ok"
        else:
            snapshot["status"] = "failed"
            snapshot["returncode"] = returncode
            stderr_path = job.run_dir / "launch-stderr.log"
            if stderr_path.is_file():
                snapshot["error"] = stderr_path.read_text(errors="replace").strip()[-_STDERR_TAIL:]
        return snapshot

    def jobs(self) -> list[dict[str, Any]]:
        """Snapshots for every job this server has launched, newest first."""
        with self._lock:
            jobs = list(self._jobs.values())
        return [self._snapshot(job) for job in reversed(jobs)]

    def wait(self, job_id: str, timeout_s: float) -> dict[str, Any]:
        """Block until a job exits (or ``timeout_s`` elapses); return its snapshot.

        Raises:
            KeyError: For unknown job ids.
        """
        with self._lock:
            job = self._jobs[job_id]
        try:
            job.process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            logger.warning("job %s still running after %.1fs", job_id, timeout_s)
        return self._snapshot(job)
