"""Admission checks for persisted operator-authored Claude command hooks."""

from __future__ import annotations

from collections.abc import Collection
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omnigent.spec import AgentSpec

from omnigent.errors import ErrorCode, OmnigentError
from omnigent.host.frames import CAP_CLAUDE_COMMAND_HOOKS


def require_hook_support(
    spec: AgentSpec | None,
    capabilities: Collection[str],
    *,
    component: str,
    operator_authored: bool,
    harness: str | None = None,
) -> None:
    if spec is None:
        return
    hooks = spec.executor.config.get("claude_command_hooks")
    if hooks is not None:
        from omnigent.harness_aliases import canonicalize_harness
        from omnigent.harnesses.claude_native.authored_hooks import parse_authored_hooks

        parse_authored_hooks(hooks)
        selected_harness = canonicalize_harness(harness or spec.executor.config.get("harness"))
        if not operator_authored or selected_harness != "claude-native":
            raise OmnigentError(
                "Command hooks require an operator-authored Claude native agent",
                code=ErrorCode.INVALID_INPUT,
            )
        if CAP_CLAUDE_COMMAND_HOOKS not in capabilities:
            raise OmnigentError(
                f"The {component} does not support Claude command hooks",
                code=ErrorCode.RUNNER_CAPABILITY_MISMATCH,
            )
    for child in spec.sub_agents:
        require_hook_support(
            child, capabilities, component=component, operator_authored=operator_authored
        )
