from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
PORTABLE_EXAMPLES = sorted(
    path
    for path in (ROOT / "examples").glob("*.py")
    if not path.name.startswith("_") and path.name != "triton_fp8_inference.py"
)

pytestmark = [
    pytest.mark.hub,
    pytest.mark.slow,
    pytest.mark.skipif(
        os.environ.get("RUN_HUB_TESTS") != "1",
        reason="set RUN_HUB_TESTS=1 to execute published-model examples",
    ),
]


@pytest.mark.parametrize("script", PORTABLE_EXAMPLES, ids=lambda path: path.stem)
def test_portable_example_runs_and_checks_its_result(script, tmp_path):
    env = {**os.environ, "RT_EXAMPLE_EPOCHS": "1"}
    completed = subprocess.run(
        [sys.executable, str(script)],
        cwd=tmp_path,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert completed.stdout.strip()
