"""Hook authority and runtime support follow the saved agent across relaunch."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from omnigent.errors import OmnigentError
from omnigent.host.frames import CAP_CLAUDE_COMMAND_HOOKS, CAP_MCP_REGISTRY
from omnigent.server.mcp_compatibility import require_session_runtime_support
from tests.spec.test_claude_command_hooks import HOOKS


@pytest.mark.parametrize(
    "capabilities", [[], [CAP_MCP_REGISTRY], [CAP_MCP_REGISTRY, CAP_CLAUDE_COMMAND_HOOKS]]
)
@pytest.mark.parametrize("trusted", [False, True])
@pytest.mark.parametrize("harness", [None, "codex-native"])
def test_saved_agent_hook_admission(capabilities, trusted, harness):
    spec = SimpleNamespace(
        executor=SimpleNamespace(
            config={"harness": "claude-native", "claude_command_hooks": HOOKS}
        ),
        sub_agents=[],
        mcp_servers=[],
    )
    agent_store = Mock()
    agent_store.get.return_value = SimpleNamespace(
        id="agent", bundle_location="saved-bundle", operator_authored=trusted
    )
    cache = Mock()
    cache.load.return_value = SimpleNamespace(spec=spec)
    conversation = SimpleNamespace(agent_id="agent", harness_override=harness, labels={})
    args = {"component": "host", "agent_store": agent_store, "agent_cache": cache}
    if trusted and harness is None and CAP_CLAUDE_COMMAND_HOOKS in capabilities:
        require_session_runtime_support(conversation, capabilities, **args)
        require_session_runtime_support(conversation, capabilities, **args)
    else:
        with pytest.raises(OmnigentError):
            require_session_runtime_support(conversation, capabilities, **args)
    assert spec.executor.config["claude_command_hooks"] == HOOKS


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "supported,override", [(True, None), (False, None), (True, "codex-native")]
)
async def test_http_creation_checks_selected_hook_runtime(
    client, app, db_uri, tmp_path, monkeypatch, supported, override
):
    from unittest.mock import AsyncMock

    from omnigent.host.frames import HOST_CAPABILITIES, HostHelloFrame
    from omnigent.server.host_registry import HostRegistry
    from omnigent.server.routes._sessions import orchestration
    from omnigent.stores.agent_store.sqlalchemy_store import SqlAlchemyAgentStore
    from omnigent.stores.artifact_store.local import LocalArtifactStore
    from tests.server.helpers import build_agent_bundle

    agent_id, host_id = "a" * 32, "b" * 32
    key = agent_id + "/bundle"
    LocalArtifactStore(str(tmp_path / "artifacts")).put(
        key,
        build_agent_bundle(
            "authored-hooks",
            executor={
                "type": "omnigent",
                "config": {
                    "harness": "claude-native",
                    "claude_command_hooks": HOOKS,
                },
            },
        ),
    )
    SqlAlchemyAgentStore(db_uri).create(agent_id, "authored-hooks", key)
    registry = HostRegistry()
    capabilities = [
        item for item in HOST_CAPABILITIES if supported or item != CAP_CLAUDE_COMMAND_HOOKS
    ]
    registry.register(
        host_id,
        Mock(),
        HostHelloFrame(
            version="test", frame_protocol_version=1, name="test", capabilities=capabilities
        ),
        owner=None,
    )
    app.state.host_registry = registry
    app.state.host_store = None
    monkeypatch.setattr(
        orchestration, "_validate_session_workspace", AsyncMock(return_value="/workspace")
    )
    response = await client.post(
        "/v1/sessions",
        json={
            "agent_id": agent_id,
            "host_id": host_id,
            "workspace": "/workspace",
            "harness_override": override,
        },
    )
    if supported and override is None:
        assert response.status_code == 201, response.text
        assert (await client.get("/v1/sessions/" + response.json()["id"])).status_code == 200
    else:
        assert response.status_code in {400, 503}, response.text
        assert "hook" in response.text.lower()
