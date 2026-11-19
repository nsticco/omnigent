"""Installed Claude qualification against a local mock provider only."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("mode", ["native", "all", "none"])
@pytest.mark.parametrize("execution", ["normal", "timeout", "cancel"])
def test_installed_claude_authored_hooks(tmp_path: Path, mode: str, execution: str) -> None:
    cli = shutil.which("claude")
    if cli is None:
        pytest.skip("installed Claude CLI required")
    if execution == "cancel" and sys.platform != "linux":
        pytest.skip("managed-terminal descendant checks use Linux process state")
    if execution == "cancel" and shutil.which("tmux") is None:
        pytest.skip("installed tmux required for managed-terminal cancellation")
    root = Path(__file__).resolve().parents[3]
    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("_authored_hooks_probe.py")),
            json.dumps(
                [
                    str(root),
                    cli,
                    str(tmp_path / "run"),
                    mode,
                    ["--permission-mode", "bypassPermissions"],
                    {"PROJECT_SIBLING": "process-sibling"},
                    {"env": {"PROJECT_LABEL": "settings-value", "PROJECT_ONLY": "settings-only"}},
                    execution == "timeout",
                    execution == "cancel",
                ]
            ),
        ],
        env={**os.environ, "PYTHONPATH": str(root)},
        text=True,
        capture_output=True,
        timeout=65,
    )
    assert result.returncode == 0, result.stdout + result.stderr[-3000:]
