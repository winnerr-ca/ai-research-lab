"""Optional experiment-tracker adapters behind a small protocol.

Local tracking (the M4A run directory) is always the source of record and
never requires any of this module. Trackers are strictly additive mirrors:

- ``track=none`` (default): no tracker, no optional imports — verified by
  test.
- ``track=wandb``: Weights & Biases adapter. Defaults to **offline mode**
  unless the user has set ``WANDB_MODE`` themselves, so no account or
  credential is ever required for normal operation; runs sync later with
  ``wandb sync`` if and when the user chooses to log in.
- ``track=mlflow``: MLflow adapter. Defaults to a local SQLite backend
  (``sqlite:///mlflow.db``, MLflow's recommended local store) — again no
  server, account, or credential required.

Both dependencies are optional (``pip install rlcore[track]``); adapters
import them lazily and fail at construction with an actionable message if
missing — before any training time is spent.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, cast

if TYPE_CHECKING:
    from collections.abc import Mapping


class Tracker(Protocol):
    """Lifecycle a trainer drives; implementations mirror metrics elsewhere."""

    def start(
        self,
        *,
        run_id: str,
        algo: str,
        env_id: str,
        seed: int,
        config: Mapping[str, Any],
    ) -> None:
        """Open the backend run with the platform's run identity."""
        ...

    def log_metrics(self, step: int, metrics: Mapping[str, float]) -> None:
        """Log one training iteration's scalar metrics."""
        ...

    def log_summary(self, summary: Mapping[str, Any]) -> None:
        """Record final-result fields on the backend run."""
        ...

    def finish(self) -> None:
        """Close the backend run."""
        ...


class WandbTracker:
    """Weights & Biases adapter (offline by default; no credentials needed).

    Args:
        project: W&B project name.
        mode: ``"offline"``/``"online"``/``"disabled"``; ``None`` respects a
            user-set ``WANDB_MODE`` env var and otherwise forces offline.
        dir: Optional directory for the local wandb files.

    Raises:
        ImportError: If ``wandb`` is not installed (install
            ``rlcore[track]``).
    """

    def __init__(
        self, project: str = "rlcore", mode: str | None = None, dir: str | None = None
    ) -> None:
        """Import wandb lazily and store settings; the run opens in start()."""
        try:
            import wandb
        except ImportError as error:
            raise ImportError(
                "track=wandb requires the optional wandb dependency: "
                "install with `pip install rlcore[track]` (or `uv sync --extra track`)."
            ) from error
        import os

        self._wandb = wandb
        self._project = project
        self._mode = mode if mode is not None else os.environ.get("WANDB_MODE", "offline")
        self._dir = dir
        self._run: Any = None

    def start(
        self, *, run_id: str, algo: str, env_id: str, seed: int, config: Mapping[str, Any]
    ) -> None:
        """Open the W&B run named after the platform run id."""
        self._run = self._wandb.init(
            project=self._project,
            name=run_id,
            mode=cast("Any", self._mode),
            dir=self._dir,
            config={"algo": algo, "env_id": env_id, "seed": seed, **dict(config)},
        )

    def log_metrics(self, step: int, metrics: Mapping[str, float]) -> None:
        """Log one iteration's metrics at ``step``."""
        self._run.log(dict(metrics), step=step)

    def log_summary(self, summary: Mapping[str, Any]) -> None:
        """Record final-result fields in the run summary."""
        for key, value in summary.items():
            self._run.summary[key] = value

    def finish(self) -> None:
        """Close the W&B run."""
        if self._run is not None:
            self._run.finish()


class MlflowTracker:
    """MLflow adapter (local file store by default; no credentials needed).

    Args:
        experiment: MLflow experiment name.
        tracking_uri: Tracking URI; the default is a local, serverless
            SQLite store in the working directory (MLflow's recommended
            local backend; its legacy ``./mlruns`` file store is in
            maintenance mode).

    Raises:
        ImportError: If ``mlflow`` is not installed (install
            ``rlcore[track]``).
    """

    def __init__(
        self, experiment: str = "rlcore", tracking_uri: str = "sqlite:///mlflow.db"
    ) -> None:
        """Import mlflow lazily and store settings; the run opens in start()."""
        try:
            import mlflow
        except ImportError as error:
            raise ImportError(
                "track=mlflow requires the optional mlflow dependency: "
                "install with `pip install rlcore[track]` (or `uv sync --extra track`)."
            ) from error
        self._mlflow = mlflow
        self._mlflow.set_tracking_uri(tracking_uri)
        self._experiment = experiment

    def start(
        self, *, run_id: str, algo: str, env_id: str, seed: int, config: Mapping[str, Any]
    ) -> None:
        """Open the MLflow run named after the platform run id."""
        self._mlflow.set_experiment(self._experiment)
        self._mlflow.start_run(run_name=run_id)
        params: dict[str, Any] = {"algo": algo, "env_id": env_id, "seed": seed}
        params.update(config)
        self._mlflow.log_params(params)

    def log_metrics(self, step: int, metrics: Mapping[str, float]) -> None:
        """Log one iteration's metrics at ``step`` (non-finite values skipped)."""
        finite = {k: float(v) for k, v in metrics.items() if v == v}  # drop NaN
        self._mlflow.log_metrics(finite, step=step)

    def log_summary(self, summary: Mapping[str, Any]) -> None:
        """Record final scalar fields as metrics; others as tags."""
        for key, value in summary.items():
            if isinstance(value, int | float) and value == value:
                self._mlflow.log_metric(f"final_{key}", float(value))
            else:
                self._mlflow.set_tag(f"final_{key}", str(value))

    def finish(self) -> None:
        """Close the MLflow run."""
        self._mlflow.end_run()


def make_tracker(kind: str, *, project: str = "rlcore") -> Tracker | None:
    """Build the tracker named by ``kind``; ``"none"`` returns ``None``.

    ``"none"`` performs no optional imports at all, so local-only operation
    never touches (or requires) wandb/mlflow.

    Raises:
        ValueError: For an unknown tracker kind.
        ImportError: If the chosen backend's package is not installed.
    """
    if kind == "none":
        return None
    if kind == "wandb":
        return WandbTracker(project=project)
    if kind == "mlflow":
        return MlflowTracker(experiment=project)
    raise ValueError(f"Unknown tracker {kind!r}; expected 'none', 'wandb', or 'mlflow'.")
