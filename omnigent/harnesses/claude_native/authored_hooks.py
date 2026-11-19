"""Bounded operator-authored hooks transported into Claude's invocation settings."""

from dataclasses import dataclass


@dataclass(frozen=True)
class CommandHook:
    command: str
    timeout: int | float | None = None

    def settings(self) -> dict[str, object]:
        result: dict[str, object] = {"type": "command", "command": self.command}
        if self.timeout is not None:
            result["timeout"] = self.timeout
        return result


def parse_authored_hooks(value: object) -> tuple[tuple[CommandHook, ...], ...]:
    """Validate and freeze synchronous PostToolUse Bash commands only."""
    if value is None:
        return ()
    if not isinstance(value, dict) or set(value) != {"PostToolUse"}:
        raise ValueError("claude_command_hooks supports only PostToolUse")
    groups = value["PostToolUse"]
    if not isinstance(groups, list) or not 1 <= len(groups) <= 16:
        raise ValueError("PostToolUse must contain 1 to 16 matcher groups")
    result = []
    for group in groups:
        if (
            not isinstance(group, dict)
            or set(group) != {"matcher", "hooks"}
            or group["matcher"] != "Bash"
        ):
            raise ValueError("claude_command_hooks requires the exact Bash matcher")
        hooks = group["hooks"]
        if not isinstance(hooks, list) or not 1 <= len(hooks) <= 16:
            raise ValueError("each matcher must contain 1 to 16 command hooks")
        commands = []
        for hook in hooks:
            if (
                not isinstance(hook, dict)
                or set(hook) - {"type", "command", "timeout"}
                or hook.get("type") != "command"
            ):
                raise ValueError("only synchronous command hooks are supported")
            command = hook.get("command")
            if (
                not isinstance(command, str)
                or not command.strip()
                or len(command) > 8192
                or "\x00" in command
            ):
                raise ValueError("hook command must be a nonempty bounded string")
            timeout = hook.get("timeout")
            if "timeout" in hook and (
                isinstance(timeout, bool)
                or not isinstance(timeout, (int, float))
                or not 0 < timeout <= 600
            ):
                raise ValueError("hook timeout must be positive and at most 600 seconds")
            commands.append(CommandHook(command, timeout))
        result.append(tuple(commands))
    return tuple(result)


def append_authored_hooks(settings: dict, value: object) -> None:
    """Append validated authored hooks without altering bridge-owned entries."""
    groups = parse_authored_hooks(value)
    if groups:
        settings["hooks"].setdefault("PostToolUse", []).extend(
            {"matcher": "Bash", "hooks": [hook.settings() for hook in group]} for group in groups
        )
