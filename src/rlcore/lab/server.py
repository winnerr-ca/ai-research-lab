"""Localhost HTTP server for the ``rlcore-lab`` dashboard.

Deliberately small and dependency-free: Python's standard-library HTTP
server, JSON endpoints over :mod:`rlcore.lab.data` and
:mod:`rlcore.lab.launch`, and a single static page. Binds to loopback by
default because the launch endpoint executes training subprocesses;
gated research actions (publish, delete, conclude, paid providers)
remain unreachable from HTTP exactly as they are from the research CLI.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path
from typing import Any, cast
from urllib.parse import parse_qs, urlparse

from rlcore.lab.data import compare_runs, discover_runs, load_metrics, resolve_run_dir
from rlcore.lab.launch import JobManager, algo_fields, discover_algos, validate_launch
from rlcore.lab.replay import record_trajectory
from rlcore.research.store import ResearchStore

logger = logging.getLogger(__name__)

_MAX_BODY_BYTES = 1_000_000


@dataclass
class LabContext:
    """Shared state for request handlers.

    Attributes:
        runs_root: Directory scanned for run directories.
        out_root: Directory new dashboard-launched runs are written under.
        store_root: Research-store root (listing only; may not exist).
        jobs: The job manager owning launched subprocesses.
    """

    runs_root: Path
    out_root: Path
    store_root: Path
    jobs: JobManager


class LabServer(ThreadingHTTPServer):
    """Threading HTTP server carrying the dashboard context."""

    daemon_threads = True

    def __init__(self, address: tuple[str, int], context: LabContext) -> None:
        """Bind to ``address`` and serve with the given context."""
        super().__init__(address, _LabHandler)
        self.context = context


def _research_listing(store_root: Path) -> dict[str, Any]:
    """List research-store entity ids grouped by kind (read-only).

    Entity ids are ``<kind>-<hex>`` by construction
    (:func:`rlcore.research.entities.new_entity_id`), so kinds come from
    the ids without loading each entity.
    """
    if not store_root.is_dir():
        return {"present": False, "store": str(store_root), "kinds": {}}
    store = ResearchStore(store_root)
    kinds: dict[str, list[str]] = {}
    for entity_id in store.list_ids():
        kind = entity_id.rsplit("-", 1)[0]
        kinds.setdefault(kind, []).append(entity_id)
    return {"present": True, "store": str(store_root), "kinds": kinds}


def _finite(value: Any) -> Any:  # noqa: ANN401 - operates on arbitrary JSON trees
    """Recursively replace non-finite floats with ``None`` for strict JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _finite(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_finite(item) for item in value]
    return value


class _LabHandler(BaseHTTPRequestHandler):
    """Routes dashboard requests; all responses are JSON except the page."""

    @property
    def _ctx(self) -> LabContext:
        return cast("LabServer", self.server).context

    def log_message(self, format: str, *args: Any) -> None:  # noqa: ANN401 - stdlib API
        """Route request logging through the module logger."""
        logger.debug("%s - %s", self.address_string(), format % args)

    def _send_json(self, payload: Any, status: int = 200) -> None:  # noqa: ANN401
        try:
            body = json.dumps(payload, allow_nan=False).encode()
        except ValueError:
            # NaN/Infinity leak in from real artifacts (e.g. DQN metrics
            # before the first loss); browsers reject them, so map to null.
            body = json.dumps(_finite(payload)).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_page(self) -> None:
        body = resources.files("rlcore.lab").joinpath("static/index.html").read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        """Serve the page and the read-only JSON API."""
        url = urlparse(self.path)
        query = parse_qs(url.query)
        try:
            if url.path in ("/", "/index.html"):
                self._send_page()
            elif url.path == "/api/algos":
                self._send_json(
                    [
                        {"name": name, "fields": algo_fields(spec)}
                        for name, spec in sorted(discover_algos().items())
                    ]
                )
            elif url.path == "/api/runs":
                self._send_json(discover_runs(self._ctx.runs_root))
            elif url.path == "/api/metrics":
                rel = query.get("dir", [""])[0]
                run_dir = resolve_run_dir(self._ctx.runs_root, rel)
                self._send_json(load_metrics(run_dir))
            elif url.path == "/api/replay":
                rel = query.get("dir", [""])[0]
                run_dir = resolve_run_dir(self._ctx.runs_root, rel)
                try:
                    self._send_json(record_trajectory(run_dir))
                except FileNotFoundError as error:
                    self._send_json({"error": f"run has no final model yet: {error}"}, status=400)
            elif url.path == "/api/compare":
                rels = [rel for rel in query.get("dirs", [""])[0].split("|") if rel]
                self._send_json(compare_runs(self._ctx.runs_root, rels))
            elif url.path == "/api/jobs":
                self._send_json(self._ctx.jobs.jobs())
            elif url.path == "/api/research":
                self._send_json(_research_listing(self._ctx.store_root))
            else:
                self._send_json({"error": f"unknown path {url.path!r}"}, status=404)
        except ValueError as error:
            self._send_json({"error": str(error)}, status=400)
        except Exception:
            logger.exception("GET %s failed", self.path)
            self._send_json({"error": "internal error; see server log"}, status=500)

    def do_POST(self) -> None:
        """Handle ``/api/launch``: validate, then start a training subprocess."""
        if urlparse(self.path).path != "/api/launch":
            self._send_json({"error": f"unknown path {self.path!r}"}, status=404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= _MAX_BODY_BYTES:
                raise ValueError("request body missing or too large.")
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError("request body must be a JSON object.")
            algo = str(payload.get("algo", ""))
            overrides = payload.get("overrides", {})
            if not isinstance(overrides, dict):
                raise ValueError("overrides must be a JSON object.")
            spec = validate_launch(discover_algos(), algo, overrides)
            self._send_json(self._ctx.jobs.launch(spec, overrides))
        except (ValueError, json.JSONDecodeError) as error:
            self._send_json({"error": str(error)}, status=400)
        except Exception:
            logger.exception("POST %s failed", self.path)
            self._send_json({"error": "internal error; see server log"}, status=500)


def lab_main() -> None:
    """Argparse entry point for ``rlcore-lab``."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="rlcore local research dashboard.")
    parser.add_argument("--host", default="127.0.0.1", help="bind address (loopback default)")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--runs-root", type=Path, default=Path("outputs"), help="directory scanned for runs"
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=None,
        help="where dashboard-launched runs go (default: <runs-root>/lab)",
    )
    parser.add_argument(
        "--store", type=Path, default=Path("research-store"), help="research-store root"
    )
    args = parser.parse_args()
    if args.host not in ("127.0.0.1", "localhost", "::1"):
        logger.warning(
            "binding to %s exposes the launch endpoint (which starts training "
            "subprocesses) beyond this machine — loopback is the intended use.",
            args.host,
        )
    out_root = args.out_root if args.out_root is not None else args.runs_root / "lab"
    context = LabContext(
        runs_root=args.runs_root,
        out_root=out_root,
        store_root=args.store,
        jobs=JobManager(out_root),
    )
    # Warm algorithm discovery off the request path: the first import of the
    # agents' train modules pulls in torch, which can take long enough on a
    # small machine that the page's first /api/algos call would race it.
    threading.Thread(target=discover_algos, daemon=True).start()
    server = LabServer((args.host, args.port), context)
    logger.info(
        "rlcore-lab on http://%s:%d  (runs: %s | launches: %s | store: %s)",
        args.host,
        args.port,
        args.runs_root,
        out_root,
        args.store,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("shutting down.")
    finally:
        server.server_close()
