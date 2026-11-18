"""The forwarder relays account usage read on its own app-server connection."""

from __future__ import annotations

import asyncio
import functools
import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from omnigent.harnesses.codex_native import forwarder as fwd
from omnigent.harnesses.codex_native.account_usage import CodexAccountUsageReader
from omnigent.harnesses.codex_native.bridge import CodexNativeBridgeState, write_bridge_state

_SESSION_ID = "conv_parent"
_THREAD = "thread_parent"
_WINDOW = {"usedPercent": 40, "windowDurationMins": 10080, "resetsAt": 1795500000}


class _CodexClient:
    """Fake app-server connection: thread requests and account reads."""

    def __init__(self) -> None:
        self.queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
        self.requests: list[tuple[str, dict[str, Any]]] = []
        self.used = 40

    async def connect(self) -> None:
        return None

    async def request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        self.requests.append((method, params))
        if method == "account/read":
            account = {"type": "chatgpt", "email": "person@example.com", "planType": "plus"}
            return {"result": {"account": account, "requiresOpenaiAuth": True}}
        if method == "account/rateLimits/read":
            window = {**_WINDOW, "usedPercent": self.used}
            bucket = {"limitId": "codex", "primary": None, "secondary": window}
            return {"result": {"rateLimits": bucket, "rateLimitsByLimitId": None}}
        return {"result": {"thread": {"id": params.get("threadId", _THREAD), "turns": []}}}

    def is_subscribed_to_thread(self, _thread_id: str) -> bool:
        return True

    async def respond(self, request_id: int | str, result: dict[str, Any]) -> None:
        return None

    async def close(self) -> None:
        return None

    async def iter_events(self) -> Any:
        while (event := await self.queue.get()) is not None:
            yield event


async def test_rate_limit_notification_triggers_a_coalesced_authoritative_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        fwd,
        "CodexAccountUsageReader",
        functools.partial(CodexAccountUsageReader, min_interval=0.05),
    )
    write_bridge_state(
        tmp_path,
        CodexNativeBridgeState(
            session_id=_SESSION_ID,
            socket_path=str(tmp_path / "app-server.sock"),
            thread_id=_THREAD,
            codex_home=str(tmp_path / "codex-home"),
        ),
    )
    codex = _CodexClient()
    relayed: list[dict[str, Any]] = []
    other_posts: list[dict[str, Any]] = []
    second_read = asyncio.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else {}
        usage = body.get("data", {}).get("codex_account_usage")
        if body.get("type") == "external_session_usage" and usage is not None:
            assert request.url.path == f"/v1/sessions/{_SESSION_ID}/events"
            relayed.append(usage)
            if len(relayed) == 2:
                second_read.set()
        else:
            other_posts.append(body)
        return httpx.Response(200, json={}, request=request)

    forwarder = asyncio.create_task(
        fwd.supervise_forwarder(
            base_url="http://omnigent.test",
            headers={},
            session_id=_SESSION_ID,
            bridge_dir=tmp_path,
            app_server_url=str(tmp_path / "app-server.sock"),
            thread_id=_THREAD,
            client=codex,  # type: ignore[arg-type]
            ap_transport=httpx.MockTransport(handler),
        )
    )
    try:
        async with asyncio.timeout(10):
            while not relayed:
                await asyncio.sleep(0.01)
        codex.used = 55
        # A sparse notification carries partial values; it is only a hint.
        sparse = {"rateLimits": {"limitId": "codex", "primary": None, "secondary": None}}
        for _ in range(20):
            codex.queue.put_nowait({"method": "account/rateLimits/updated", "params": sparse})
        async with asyncio.timeout(10):
            await second_read.wait()
        await asyncio.sleep(0.2)
    finally:
        codex.queue.put_nowait(None)
    async with asyncio.timeout(30):
        await forwarder

    assert [usage["rate_limits"]["secondary"]["used_percent"] for usage in relayed] == [40, 55]
    assert relayed[0]["account"]["email"] == "person@example.com"
    assert codex.requests.count(("account/read", {"refreshToken": False})) == 4
    assert not any("rateLimits" in json.dumps(post) for post in other_posts)
