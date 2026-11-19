"""Reject known-unavailable harnesses before creating a session."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import TYPE_CHECKING

from omnigent.entities import Conversation
from omnigent.errors import ErrorCode, OmnigentError
from omnigent.harness_aliases import native_terminal_name
from omnigent.harness_availability import HARNESS_TMUX_UNAVAILABLE, harness_launch_availability
from omnigent.host.frames import CAP_CODEX_PLAN_AT_LAUNCH
from omnigent.stores import ConversationStore
from omnigent.stores.conversation_store import CODEX_NATIVE_COLLABORATION_MODE_LABEL_KEY
from omnigent.stores.host_store import HostStore
from omnigent.util.tmux_compat import MIN_TMUX_VERSION_HINT

if TYPE_CHECKING:
    from fastapi import Request

    from omnigent.entities import Agent
    from omnigent.server.host_registry import HostRegistry
    from omnigent.stores.permission_store import PermissionStore


async def validate_create_harness_readiness(
    *,
    harness: str | None,
    host_id: str | None,
    parent_session_id: str | None,
    inherited_runner_id: str | None,
    user_id: str | None,
    conversation_store: ConversationStore,
    host_store: HostStore | None,
    parent: Conversation | None = None,
) -> None:
    """Check resolved placement after parent authorization and before persistence.

    Children on an existing runner use their ancestor's host. Legacy hosts
    without readiness reports remain unknown; host launch checks still apply.
    """
    from omnigent.server.routes._host_launch import resolve_host_owner

    if host_store is None or not harness or harness == "auto":
        return
    if inherited_runner_id is not None:
        from omnigent.runner.routing import routing_host_id

        if parent is None and parent_session_id:
            parent = await asyncio.to_thread(
                conversation_store.get_conversation, parent_session_id
            )
        host_id = (
            await asyncio.to_thread(
                routing_host_id, parent, conversation_store, max_ancestor_reads=16
            )
            if parent is not None
            else None
        )
        host = await asyncio.to_thread(host_store.get_host, host_id) if host_id else None
    elif host_id:
        host = await asyncio.to_thread(
            resolve_host_owner, user_id=user_id, host_id=host_id, host_store=host_store
        )
    else:
        host = None
    if host is None:
        return
    # Sharing a parent session does not grant access to its host telemetry.
    if user_id is not None and host.user_id != user_id:
        return
    available, reason = harness_launch_availability(harness, host.configured_harnesses)
    if available is not False:
        return
    if reason == HARNESS_TMUX_UNAVAILABLE:
        raise OmnigentError(
            f"Harness {harness!r} needs tmux {MIN_TMUX_VERSION_HINT} or newer on the target "
            "host. Install or upgrade tmux there (macOS: `brew install tmux`), then check "
            "the harness again. If Omnigent still can't find it, quit and reopen the "
            "Omnigent app, or choose another harness.",
            code=ErrorCode.HARNESS_NOT_CONFIGURED,
        )
    raise OmnigentError(
        f"Harness {harness!r} is not configured on the target host ({reason}). "
        "Install or configure the harness on that host, or choose an available harness.",
        code=ErrorCode.HARNESS_NOT_CONFIGURED,
    )


def require_codex_plan_launch_support(
    *,
    harness: str | None,
    host_id: str | None,
    labels: Mapping[str, str],
    host_registry: HostRegistry | None,
) -> None:
    """Refuse a Plan-mode Codex create on a connected host whose runner would start in Default.

    Call only after host ownership is authorized so the 409 never reveals another
    user's host build.
    """
    if (
        host_id is None
        or host_registry is None
        or native_terminal_name(harness) != "codex"
        or labels.get(CODEX_NATIVE_COLLABORATION_MODE_LABEL_KEY) != "plan"
    ):
        return
    if not host_registry.host_may_support(host_id, CAP_CODEX_PLAN_AT_LAUNCH):
        raise OmnigentError(
            "This host is too old to start a Codex session in Plan mode. Update "
            "omnigent on the host and reconnect it, or turn Plan mode off, then try again.",
            code=ErrorCode.CONFLICT,
        )


def require_process_home_host(host_id: str | None, host_registry: HostRegistry | None) -> None:
    """Require positively advertised support, including after a session host rebind."""
    from omnigent.host.frames import CAP_SESSION_PROCESS_HOME

    connection = host_registry.get(host_id) if host_id and host_registry is not None else None
    if connection is None or CAP_SESSION_PROCESS_HOME not in connection.hello.capabilities:
        raise OmnigentError(
            "The selected host does not support session process_home", code=ErrorCode.CONFLICT
        )


async def validate_process_home_selection(
    process_home: str | None,
    *,
    request: Request,
    user_id: str | None,
    permission_store: PermissionStore | None,
    agent: Agent,
    harness: str | None,
    host_id: str | None,
    host_type: str,
    child: bool = False,
) -> None:
    """Admit an operator-only private HOME without probing the server filesystem."""
    if process_home is None:
        return
    from omnigent.inner.datamodel import OSEnvSpec
    from omnigent.server.auth import local_single_user_enabled
    from omnigent.server.routes._session_create_validation import _authorize_host_for_workspace

    try:
        OSEnvSpec(process_home=process_home)
    except ValueError as exc:
        raise OmnigentError(str(exc), code=ErrorCode.INVALID_INPUT) from exc
    if permission_store is not None and not local_single_user_enabled():
        if user_id is None or not await asyncio.to_thread(permission_store.is_admin, user_id):
            raise OmnigentError(
                "process_home requires operator authority", code=ErrorCode.FORBIDDEN
            )
    if (
        not agent.operator_authored
        or native_terminal_name(harness) != "claude"
        or child
        or host_type != "external"
        or not host_id
    ):
        raise OmnigentError(
            "process_home requires an operator-authored Claude agent on an explicit external host",
            code=ErrorCode.INVALID_INPUT,
        )
    await _authorize_host_for_workspace(
        user_id=user_id,
        host_id=host_id,
        host_store=getattr(request.app.state, "host_store", None),
        host_registry=getattr(request.app.state, "host_registry", None),
    )
    require_process_home_host(host_id, getattr(request.app.state, "host_registry", None))
