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
    conversation = SimpleNamespace(
        agent_id="agent", harness_override=harness, sub_agent_name=None, labels={}
    )
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
    "placement", ["explicit_host", "inherited_runner", "inherited_runner_with_host"]
)
@pytest.mark.parametrize(
    "supported,override", [(True, None), (False, None), (True, "codex-native")]
)
async def test_http_creation_checks_selected_hook_runtime(
    client, app, db_uri, tmp_path, monkeypatch, supported, override, placement
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
    payload = {
        "agent_id": agent_id,
        "host_id": host_id,
        "workspace": "/workspace",
        "harness_override": override,
    }
    if placement.startswith("inherited_runner"):
        from omnigent.stores.conversation_store.sqlalchemy_store import SqlAlchemyConversationStore
        from tests.server.helpers import create_test_agent

        parent_agent = await create_test_agent(client)
        parent = await client.post("/v1/sessions", json={"agent_id": parent_agent["id"]})
        assert parent.status_code == 201, parent.text
        parent_id = parent.json()["id"]
        store = SqlAlchemyConversationStore(db_uri)
        store.set_runner_id(parent_id, "runner_test")
        tunnel = Mock()
        tunnel.get.return_value = SimpleNamespace(hello=SimpleNamespace(capabilities=capabilities))
        app.state.tunnel_registry = tunnel
        if placement == "inherited_runner":
            payload.pop("host_id")
        else:
            registry.get(host_id).hello.capabilities = [
                item
                for item in HOST_CAPABILITIES
                if not supported or item != CAP_CLAUDE_COMMAND_HOOKS
            ]
        payload["parent_session_id"] = parent_id
    response = await client.post("/v1/sessions", json=payload)
    if supported and override is None:
        assert response.status_code == 201, response.text
        assert (await client.get("/v1/sessions/" + response.json()["id"])).status_code == 200
    else:
        assert response.status_code in {400, 503}, response.text
        assert "hook" in response.text.lower()


@pytest.mark.parametrize(
    "selected,harness,supported,allowed",
    [
        ("codex-child", "codex-native", False, True),
        ("codex-child", None, False, True),
        ("hook-child", "claude-native", True, True),
        ("hook-child", "claude-native", False, False),
        ("hook-child", "codex-native", True, False),
        ("missing", None, True, False),
    ],
)
def test_saved_session_uses_selected_child(selected, harness, supported, allowed):
    from omnigent.spec import AgentSpec, ExecutorSpec

    hooked = ExecutorSpec(
        type="omnigent", config={"harness": "claude-native", "claude_command_hooks": HOOKS}
    )
    root = AgentSpec(
        spec_version=1,
        name="root",
        executor=hooked,
        sub_agents=[
            AgentSpec(
                spec_version=1,
                name="codex-child",
                executor=ExecutorSpec(type="omnigent", config={"harness": "codex-native"}),
            ),
            AgentSpec(spec_version=1, name="hook-child", executor=hooked),
        ],
    )
    store, cache = Mock(), Mock()
    store.get.return_value = SimpleNamespace(
        id="agent", bundle_location="saved", operator_authored=True
    )
    cache.load.return_value = SimpleNamespace(spec=root)
    conversation = SimpleNamespace(
        agent_id="agent", sub_agent_name=selected, harness_override=harness, labels={}
    )
    capabilities = [CAP_MCP_REGISTRY] + ([CAP_CLAUDE_COMMAND_HOOKS] if supported else [])

    def validate():
        require_session_runtime_support(
            conversation, capabilities, component="runner", agent_store=store, agent_cache=cache
        )

    if allowed:
        validate()
    else:
        with pytest.raises(OmnigentError):
            validate()
