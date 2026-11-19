"""Private HOME is explicit, durable operator launch configuration."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from omnigent.errors import OmnigentError
from omnigent.host.frames import CAP_SESSION_PROCESS_HOME
from omnigent.server.routes._session_harness_readiness import validate_process_home_selection


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case",
    [
        "valid",
        "relative",
        "user_agent",
        "non_admin",
        "old_host",
        "offline",
        "child",
        "other_harness",
    ],
)
async def test_process_home_admission(case, monkeypatch):
    monkeypatch.setenv("OMNIGENT_LOCAL_SINGLE_USER", "0")
    registry = Mock()
    registry.get.return_value = (
        None
        if case == "offline"
        else SimpleNamespace(
            hello=SimpleNamespace(
                capabilities=[] if case == "old_host" else [CAP_SESSION_PROCESS_HOME]
            )
        )
    )
    permissions = Mock()
    permissions.is_admin.return_value = case != "non_admin"
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(host_registry=registry)))
    args = {
        "request": request,
        "user_id": "operator@example.test",
        "permission_store": permissions,
        "agent": SimpleNamespace(operator_authored=case != "user_agent"),
        "harness": "codex-native" if case == "other_harness" else "claude-native",
        "host_id": "host",
        "host_type": "external",
        "child": case == "child",
    }
    path = "relative" if case == "relative" else "/not/on/server/private"
    if case == "valid":
        await validate_process_home_selection(path, **args)
    else:
        with pytest.raises(OmnigentError):
            await validate_process_home_selection(path, **args)


@pytest.mark.asyncio
async def test_process_home_http_create_read_fork_and_patch(
    client, app, db_uri, tmp_path, monkeypatch
):
    from omnigent.host.frames import HOST_CAPABILITIES, HostHelloFrame
    from omnigent.server.host_registry import HostRegistry
    from omnigent.server.routes._sessions import orchestration
    from omnigent.stores.agent_store.sqlalchemy_store import SqlAlchemyAgentStore
    from omnigent.stores.artifact_store.local import LocalArtifactStore
    from tests.server.helpers import build_agent_bundle

    agent_id = "a" * 32
    bundle_key = agent_id + "/bundle"
    LocalArtifactStore(str(tmp_path / "artifacts")).put(
        bundle_key,
        build_agent_bundle(
            "private-home-agent",
            executor={"type": "omnigent", "config": {"harness": "claude-native"}},
        ),
    )
    SqlAlchemyAgentStore(db_uri).create(agent_id, "private-home-agent", bundle_key)
    registry = HostRegistry()
    host_id = "b" * 32
    registry.register(
        host_id,
        Mock(),
        HostHelloFrame(
            version="test", frame_protocol_version=1, name="test", capabilities=HOST_CAPABILITIES
        ),
        owner=None,
    )
    app.state.host_registry = registry
    # Only the remote host transport is stubbed; admission and persistence stay real.
    monkeypatch.setattr(
        orchestration, "_validate_session_workspace", AsyncMock(return_value="/workspace")
    )
    app.state.host_store = None
    response = await client.post(
        "/v1/sessions",
        json={
            "agent_id": agent_id,
            "host_id": host_id,
            "workspace": "/workspace",
            "process_home": "/private/source",
        },
    )
    assert response.status_code == 201, response.text
    session_id = response.json()["id"]
    assert response.json()["process_home"] == "/private/source"
    assert (await client.get(f"/v1/sessions/{session_id}")).json()[
        "process_home"
    ] == "/private/source"
    changed = await client.patch(
        f"/v1/sessions/{session_id}", json={"process_home": "/private/changed"}
    )
    assert changed.status_code == 422
    from omnigent.stores.host_store import HostStore

    hosts = HostStore(db_uri)
    hosts.upsert_on_connect(host_id, "existing-host", "local")
    app.state.host_store = hosts
    forked = await client.post(
        f"/v1/sessions/{session_id}/fork",
        json={"process_home": "/private/fork", "host_id": host_id},
    )
    assert forked.status_code == 201, forked.text
    assert forked.json()["process_home"] == "/private/fork"
    assert (await client.get(f"/v1/sessions/{session_id}")).json()[
        "process_home"
    ] == "/private/source"


def test_uploaded_bundle_cannot_choose_runner_private_home():
    import json

    from omnigent.server.bundles import validate_agent_bundle
    from tests._helpers.session import bundle_files

    for terminal in (False, True):
        selected = {"process_home": "/private/operator", "sandbox": {"type": "none"}}
        config = {
            "spec_version": 1,
            "name": "unsafe-home",
            "prompt": "hello",
            "executor": {"type": "omnigent", "config": {"harness": "claude-native"}},
        }
        config.update(
            {"terminals": {"shell": {"command": "sh", "os_env": selected}}}
            if terminal
            else {"os_env": selected}
        )
        with pytest.raises(OmnigentError, match="process_home"):
            validate_agent_bundle(bundle_files({"config.yaml": json.dumps(config).encode()}))
