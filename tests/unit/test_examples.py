"""Tutorial scripts are tested artifacts: each runs green at smoke scale."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"

CASES = [
    ("01_train_and_evaluate.py", ["--updates", "2"]),
    ("02_resume_training.py", ["--steps", "2000"]),
    ("03_compare_with_statistics.py", ["--updates", "2", "--seeds", "2"]),
    ("04_research_workflow.py", ["--updates", "2"]),
]


@pytest.mark.parametrize(("script", "args"), CASES, ids=[c[0] for c in CASES])
def test_example_runs_green(script: str, args: list[str], tmp_path: Path) -> None:
    out_dir = tmp_path / "example-out"
    extra = [] if script.startswith("03") else ["--out-dir", str(out_dir)]
    completed = subprocess.run(
        [sys.executable, str(EXAMPLES / script), *args, *extra],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr[-2000:]
