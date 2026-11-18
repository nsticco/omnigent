"""Account and rate-limit reads on the native Codex forwarder's connection."""

from __future__ import annotations

import asyncio
import copy
from typing import Any

import pytest

from omnigent.harnesses.codex_native.account_usage import (
    CodexAccountUsageReader,
    project_account_usage,
)
from omnigent.harnesses.codex_native.app_server import CodexAppServerResponseError

_ACCOUNT = {
    "account": {
        "type": "chatgpt",
        "email": "person@example.com",
        "planType": "plus",
        "workspaceName": "Example Workspace",
    },
    "userId": "user-1",
    "requiresOpenaiAuth": True,
    "workspaceRouting": {
        "chatgptAccountId": "acct-1111",
        "backendOrigin": "https://chatgpt.example.com",
        "accountRoutingOverride": "NO_CONSTRAINT",
    },
}
_CODEX = {
    "limitId": "codex",
    "limitName": None,
    "normalModelSlug": None,
    "primary": {"usedPercent": 12, "windowDurationMins": 300, "resetsAt": 1795020000},
    "secondary": {"usedPercent": 40, "windowDurationMins": 10080, "resetsAt": 1795500000},
    "credits": {"hasCredits": False, "unlimited": False, "balance": "0"},
    "planType": "plus",
}
_LIMITS = {
    "ordinaryUsageAllowed": None,
    "rateLimits": _CODEX,
    "rateLimitsByLimitId": {"codex": _CODEX},
    "rateLimitResetCredits": {"availableCount": 2},
    "accountId": "acct-1111",
    "rateLimitUpsell": {"banner": "upgrade"},
}


def _project(**changes: Any) -> dict[str, object] | None:
    limits = {**copy.deepcopy(_LIMITS), **changes}
    return project_account_usage(_ACCOUNT, limits, _ACCOUNT, observed_at=1795000000.5)


def test_projection_keeps_only_account_and_window_fields() -> None:
    projection = _project()

    assert projection == {
        "observed_at": 1795000000.5,
        "codex_version": None,
        "account": {
            "type": "chatgpt",
            "email": "person@example.com",
            "plan_type": "plus",
            "workspace_name": "Example Workspace",
        },
        "routing_account_id": "acct-1111",
        "account_id": "acct-1111",
        "rate_limits": {
            "limit_id": "codex",
            "limit_name": None,
            "model": None,
            "primary": {"used_percent": 12, "window_duration_mins": 300, "resets_at": 1795020000},
            "secondary": {
                "used_percent": 40,
                "window_duration_mins": 10080,
                "resets_at": 1795500000,
            },
        },
        "rate_limits_by_limit_id": {"codex": projection["rate_limits"]},
    }


def test_projection_drops_a_read_pair_that_straddled_an_account_switch() -> None:
    switched = {**_ACCOUNT, "userId": "user-2"}

    assert project_account_usage(_ACCOUNT, _LIMITS, switched, observed_at=1.0) is None


def test_projection_keeps_usage_when_an_account_read_failed() -> None:
    projection = project_account_usage(None, _LIMITS, _ACCOUNT, observed_at=1.0)

    assert projection is not None
    assert projection["account"] is None
    assert projection["account_id"] == "acct-1111"


@pytest.mark.parametrize(
    "window",
    [
        {"usedPercent": -1, "windowDurationMins": 10080, "resetsAt": 1},
        {"usedPercent": "lots", "windowDurationMins": 10080, "resetsAt": 1},
        {"usedPercent": 1, "windowDurationMins": 10080.5, "resetsAt": 1},
        "not a window",
    ],
)
def test_projection_rejects_unreadable_windows(window: object) -> None:
    bucket = {**_CODEX, "secondary": window}

    assert _project(rateLimits=bucket, rateLimitsByLimitId=None) is None


class _AccountClient:
    """Fake app-server connection answering account and rate-limit reads."""

    def __init__(self, *, account: dict | None = None) -> None:
        self.account = account or _ACCOUNT
        self.requests: list[tuple[str, dict[str, Any]]] = []
        self.failures: dict[str, list[CodexAppServerResponseError]] = {}

    def fail(self, method: str, *errors: dict[str, object]) -> None:
        self.failures[method] = [CodexAppServerResponseError(error) for error in errors]

    async def request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        self.requests.append((method, params))
        pending = self.failures.get(method)
        if pending:
            raise pending.pop(0)
        return {"result": self.account if method == "account/read" else _LIMITS}

    def methods(self) -> list[str]:
        return [method for method, _params in self.requests]


class _Relay:
    def __init__(self, *, accept: bool = True) -> None:
        self.accept = accept
        self.posts: list[dict[str, object]] = []
        self.posted = asyncio.Event()

    async def __call__(self, projection: dict[str, object]) -> bool:
        self.posts.append(projection)
        self.posted.set()
        return self.accept


async def _run(reader: CodexAccountUsageReader, until: Any, timeout: float = 2.0) -> None:
    task = asyncio.create_task(reader.run())
    try:
        await asyncio.wait_for(until(), timeout)
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


async def test_reader_reads_once_after_connect_without_forcing_renewal() -> None:
    client, relay = _AccountClient(), _Relay()
    reader = CodexAccountUsageReader(client, relay, min_interval=60)  # type: ignore[arg-type]

    await _run(reader, relay.posted.wait)

    assert client.requests == [
        ("account/read", {"refreshToken": False}),
        ("account/rateLimits/read", {"excludeResetCreditDetails": True}),
        ("account/read", {"refreshToken": False}),
    ]
    assert relay.posts == [_project() | {"observed_at": relay.posts[0]["observed_at"]}]


async def test_rate_limit_hints_coalesce_into_one_read_per_interval() -> None:
    client, relay = _AccountClient(), _Relay()
    reader = CodexAccountUsageReader(client, relay, min_interval=0.2)  # type: ignore[arg-type]

    async def burst_then_settle() -> None:
        await relay.posted.wait()
        for _ in range(50):
            reader.hint()
            await asyncio.sleep(0)
        await asyncio.sleep(0.5)

    await _run(reader, burst_then_settle)

    assert len(relay.posts) == 2
    assert client.methods().count("account/rateLimits/read") == 2


async def test_failed_relays_back_off_instead_of_looping() -> None:
    client, relay = _AccountClient(), _Relay(accept=False)
    reader = CodexAccountUsageReader(client, relay, min_interval=0.1)  # type: ignore[arg-type]

    async def hint_repeatedly() -> None:
        for _ in range(30):
            reader.hint()
            await asyncio.sleep(0.02)

    await _run(reader, hint_repeatedly)

    # 0.6s of hints: the first read, then waits of 0.2s and 0.4s.
    assert 2 <= len(relay.posts) <= 3


async def test_a_rejected_token_is_renewed_once_per_session() -> None:
    client, relay = _AccountClient(), _Relay()
    unauthorized = {"code": -32603, "message": "GET /wham/usage failed: 401 Unauthorized"}
    client.fail("account/rateLimits/read", unauthorized, unauthorized, unauthorized)
    reader = CodexAccountUsageReader(client, relay, min_interval=0.05)  # type: ignore[arg-type]

    async def two_reads() -> None:
        while client.methods().count("account/rateLimits/read") < 3:
            reader.hint()
            await asyncio.sleep(0.05)
        reader.hint()
        await relay.posted.wait()

    await _run(reader, two_reads)

    renewals = [params for method, params in client.requests if params.get("refreshToken")]
    assert renewals == [{"refreshToken": True}]
    assert len(relay.posts) == 1


async def test_a_session_without_a_chatgpt_account_stops_reading() -> None:
    client, relay = _AccountClient(account={"account": {"type": "apiKey"}}), _Relay()
    reader = CodexAccountUsageReader(client, relay, min_interval=0.01)  # type: ignore[arg-type]

    task = asyncio.create_task(reader.run())
    reader.hint()
    await asyncio.wait_for(task, 2)

    assert client.methods() == ["account/read"]
    assert relay.posts == []


async def test_a_rejected_optional_field_is_not_sent_again() -> None:
    client, relay = _AccountClient(), _Relay()
    client.fail(
        "account/rateLimits/read",
        {"code": -32600, "message": "unknown field `excludeResetCreditDetails`"},
    )
    reader = CodexAccountUsageReader(client, relay, min_interval=0.01)  # type: ignore[arg-type]

    async def two_posts() -> None:
        while len(relay.posts) < 2:
            reader.hint()
            await asyncio.sleep(0.02)

    await _run(reader, two_posts)

    rate_reads = [params for method, params in client.requests if method.endswith("Limits/read")]
    assert rate_reads[:3] == [{"excludeResetCreditDetails": True}, {}, {}]
