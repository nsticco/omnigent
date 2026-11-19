"""Operator-authored commands compose with, rather than replace, bridge hooks."""

import copy
import json
from pathlib import Path

import pytest

from omnigent.harnesses.claude_native.bridge import augment_claude_args


@pytest.mark.parametrize("skills_filter", ["all", "none"])
def test_authored_hooks_are_additive_and_frozen(
    tmp_path: Path, skills_filter: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("omnigent.harnesses.claude_native.bridge._TRUSTED_PARENT", tmp_path)
    monkeypatch.setattr("omnigent.harnesses.claude_native.bridge._BRIDGE_ROOT", tmp_path)
    authored = {
        "PostToolUse": [
            {
                "matcher": "Bash",
                "hooks": [
                    {"type": "command", "command": "printf authored", "timeout": 2},
                ],
            }
        ]
    }
    original = copy.deepcopy(authored)
    baseline = augment_claude_args((), bridge_dir=tmp_path, skills_filter=skills_filter)
    baseline_settings = json.loads(Path(baseline[baseline.index("--settings") + 1]).read_text())
    args = augment_claude_args(
        (), bridge_dir=tmp_path, skills_filter=skills_filter, authored_hooks=authored
    )
    settings = json.loads(Path(args[args.index("--settings") + 1]).read_text())
    assert args.count("--settings") == 1
    assert authored == original
    for event, hooks in baseline_settings["hooks"].items():
        assert settings["hooks"][event][: len(hooks)] == hooks
    assert settings["hooks"]["PostToolUse"][-1] == authored["PostToolUse"][0]
    assert settings["permissions"] == baseline_settings["permissions"]


@pytest.mark.parametrize(
    "authored",
    [
        {"PreToolUse": []},
        {"PostToolUse": [{"matcher": ".*", "hooks": [{"type": "command", "command": "true"}]}]},
        {"PostToolUse": [{"matcher": "Bash", "hooks": [{"type": "prompt", "prompt": "x"}]}]},
        {
            "PostToolUse": [
                {
                    "matcher": "Bash",
                    "hooks": [{"type": "command", "command": "true", "async": True}],
                }
            ]
        },
        {
            "PostToolUse": [
                {
                    "matcher": "Bash",
                    "hooks": [{"type": "command", "command": "true", "timeout": True}],
                }
            ]
        },
    ],
)
def test_unsupported_authored_hooks_fail_closed(tmp_path: Path, authored: object) -> None:
    with pytest.raises(ValueError):
        augment_claude_args((), bridge_dir=tmp_path, authored_hooks=authored)
