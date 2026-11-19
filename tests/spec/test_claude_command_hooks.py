"""Authored commands survive trusted bundle transport, never untrusted uploads."""

from pathlib import Path

import pytest
import yaml

from omnigent.chat import _bundle_agent
from omnigent.errors import OmnigentError
from omnigent.server.bundles import validate_agent_bundle
from omnigent.spec import load
from omnigent.spec.omnigent import agent_def_to_agent_spec, agent_spec_to_agent_def

HOOKS = {
    "PostToolUse": [
        {
            "matcher": "Bash",
            "hooks": [{"type": "command", "command": "printf authored", "timeout": 2}],
        }
    ]
}


@pytest.mark.parametrize("directory", [False, True])
def test_hooks_roundtrip_and_upload_boundary(tmp_path: Path, directory: bool) -> None:
    executor = {"harness": "claude-native", "model": "claude-test", "claude_command_hooks": HOOKS}
    source = tmp_path / "source"
    source.mkdir()
    if directory:
        (source / "config.yaml").write_text(
            yaml.safe_dump(
                {
                    "spec_version": 1,
                    "name": "hooks",
                    "executor": {
                        "type": "omnigent",
                        "model": executor.pop("model"),
                        "config": executor,
                    },
                }
            )
        )
    else:
        source = source / "agent.yaml"
        source.write_text(
            yaml.safe_dump({"name": "hooks", "prompt": "Run the task.", "executor": executor})
        )
    bundle = _bundle_agent(source)
    spec = load(bundle, dest=tmp_path / "uploaded")
    for candidate in (spec, agent_def_to_agent_spec(agent_spec_to_agent_def(spec))):
        assert candidate.executor.config["claude_command_hooks"] == HOOKS
    with pytest.raises(OmnigentError, match="operator-authored"):
        validate_agent_bundle(bundle)
    assert (
        validate_agent_bundle(bundle, enforce_handler_allowlist=False).executor.config[
            "claude_command_hooks"
        ]
        == HOOKS
    )
