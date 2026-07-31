"""Tests for the lab HTTP server: routes, validation, and page serving."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

import pytest

from rlcore.lab.launch import JobManager
from rlcore.lab.server import LabContext, LabServer

if TYPE_CHECKING:
    from pathlib import Path


def make_run(root: Path, name: str) -> None:
    run_dir = root / name
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "run_id": name,
                "algo": "ppo",
                "env_id": "CartPole-v1",
                "seed": 0,
                "created_at": "2026-07-01T00:00:00+00:00",
            }
        )
    )
    (run_dir / "result.json").write_text(
        json.dumps({"final_eval_return_mean": 400.0, "total_env_steps": 1000})
    )
    # Second line carries NaN the way DQN's trainer really writes it
    # (Python json emits bare NaN); the server must map it to null.
    (run_dir / "metrics.jsonl").write_text(
        json.dumps({"update": 1.0, "eval_return_mean": 5.0})
        + "\n"
        + json.dumps({"update": 2.0, "loss": float("nan")})
        + "\n"
    )


@pytest.fixture
def server(tmp_path: Path) -> Iterator[str]:
    runs_root = tmp_path / "outputs"
    make_run(runs_root, "demo")
    context = LabContext(
        runs_root=runs_root,
        out_root=runs_root / "lab",
        store_root=tmp_path / "research-store",
        jobs=JobManager(runs_root / "lab"),
    )
    lab = LabServer(("127.0.0.1", 0), context)
    thread = threading.Thread(target=lab.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{lab.server_address[1]}"
    finally:
        lab.shutdown()
        lab.server_close()
        thread.join(timeout=10)


def get_json(url: str) -> Any:  # noqa: ANN401 - JSON payload shapes vary by endpoint
    with urllib.request.urlopen(url) as response:
        raw = response.read()

    # Strict parse: browsers reject NaN/Infinity, so the server must never
    # emit them (a real regression: SAC's NaN target_entropy default).
    def _reject(constant: str) -> None:
        raise AssertionError(f"server emitted non-JSON constant {constant!r}")

    return json.loads(raw, parse_constant=_reject)


class TestRoutes:
    def test_serves_page(self, server: str) -> None:
        with urllib.request.urlopen(server + "/") as response:
            body = response.read().decode()
        assert response.headers["Content-Type"].startswith("text/html")
        assert "rlcore lab" in body

    def test_runs_metrics_compare(self, server: str) -> None:
        runs = get_json(server + "/api/runs")
        assert [run["run_id"] for run in runs] == ["demo"]
        metrics = get_json(server + "/api/metrics?dir=demo")
        assert metrics == [
            {"update": 1.0, "eval_return_mean": 5.0},
            {"update": 2.0, "loss": None},
        ]
        compared = get_json(server + "/api/compare?dirs=demo")
        assert compared["groups"][0]["stats"]["mean"] == 400.0

    def test_algos_listed_with_fields(self, server: str) -> None:
        algos = get_json(server + "/api/algos")
        names = {algo["name"] for algo in algos}
        assert {"reinforce", "ppo", "dqn", "sac"} <= names

    def test_research_absent_store(self, server: str) -> None:
        listing = get_json(server + "/api/research")
        assert listing["present"] is False

    def test_jobs_empty(self, server: str) -> None:
        assert get_json(server + "/api/jobs") == []

    def test_unknown_path_404(self, server: str) -> None:
        with pytest.raises(urllib.error.HTTPError) as excinfo:
            get_json(server + "/api/nope")
        assert excinfo.value.code == 404


class TestValidationResponses:
    def test_path_traversal_rejected(self, server: str) -> None:
        with pytest.raises(urllib.error.HTTPError) as excinfo:
            get_json(server + "/api/metrics?dir=../../etc")
        assert excinfo.value.code == 400

    def test_launch_rejects_unknown_algo(self, server: str) -> None:
        request = urllib.request.Request(
            server + "/api/launch",
            data=json.dumps({"algo": "a3c", "overrides": {}}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as excinfo:
            urllib.request.urlopen(request)
        assert excinfo.value.code == 400
        assert "unknown algorithm" in json.loads(excinfo.value.read())["error"]

    def test_stop_rejects_unknown_job(self, server: str) -> None:
        request = urllib.request.Request(
            server + "/api/stop",
            data=json.dumps({"job_id": "ghost"}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as excinfo:
            urllib.request.urlopen(request)
        assert excinfo.value.code == 400
        assert "unknown job id" in json.loads(excinfo.value.read())["error"]

    def test_launch_rejects_bad_body(self, server: str) -> None:
        request = urllib.request.Request(server + "/api/launch", data=b"[1, 2]", method="POST")
        with pytest.raises(urllib.error.HTTPError) as excinfo:
            urllib.request.urlopen(request)
        assert excinfo.value.code == 400
