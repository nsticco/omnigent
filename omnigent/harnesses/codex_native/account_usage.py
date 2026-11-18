"""
Read the Codex account and plan rate limits on the forwarder's connection.

The native Codex forwarder already holds an initialized connection to the
session's app-server. :class:`CodexAccountUsageReader` reads the account and
its quota there with ``account/read`` (``refreshToken: false``),
``account/rateLimits/read`` and a second ``account/read``, so nothing starts
another app-server against the session's login. A change between the two
account reads means the pair straddled a login or workspace switch, and the
reading is dropped.

The reader runs once after the forwarder connects and again when Codex sends
``account/rateLimits/updated``. That notification is sparse, so it is only a
hint: bursts coalesce into one authoritative read no sooner than
``min_interval`` after the previous one, failed reads back off, and the
notification's own values are never relayed. A rejected token is renewed at
most once per session through Codex's own ``refreshToken: true`` flow.

Each successful read is relayed on ``external_session_usage`` as an
allowlisted ``codex_account_usage`` projection; tokens, provider error text
and other response fields never leave this module. It never starts a turn,
redeems a reset, or enables a client capability.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import re
import time
from collections.abc import Awaitable, Callable, Mapping

from omnigent.harnesses.codex_native.app_server import (
    CodexAppServerClient,
    CodexAppServerResponseError,
)

_logger = logging.getLogger(__name__)

ACCOUNT_RATE_LIMITS_UPDATED_METHOD = "account/rateLimits/updated"
# Codex skips the reset-credit inventory lookup with this flag. A protocol
# that rejects the field is asked again without it.
_EXCLUDE_RESET_CREDITS = "excludeResetCreditDetails"
_REQUEST_TIMEOUT_SECONDS = 20.0
_MIN_INTERVAL_SECONDS = 60.0
_MAX_BACKOFF_SECONDS = 3600.0
_MAX_BUCKETS = 16
_BUCKET_ID = re.compile(r"^[A-Za-z0-9_.:-]{1,48}$")
_UNAUTHORIZED = re.compile(r"(?<![0-9])401(?![0-9])|unauthorized|token expired")

PostAccountUsage = Callable[[dict[str, object]], Awaitable[bool]]


def _text(value: object, limit: int = 128) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text or len(text) > limit or not text.isprintable():
        return None
    return text


def _number(value: object, *, integer: bool) -> float | int | None:
    """
    Return a finite, positive-or-zero JSON number, or raise for anything else.

    :param value: Protocol value, e.g. ``40`` or ``None``.
    :param integer: Whether the field must be a whole number.
    :returns: The number, or ``None`` when not reported.
    :raises ValueError: If the value is present but unreadable.
    """
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise ValueError("not a non-negative number")
    if integer:
        if isinstance(value, float) and not value.is_integer():
            raise ValueError("not a whole number")
        return int(value)
    if value != value or value in (float("inf"), float("-inf")):
        raise ValueError("not finite")
    return value


def _window(value: object) -> dict[str, object] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("window is not an object")
    return {
        "used_percent": _number(value.get("usedPercent"), integer=False),
        "window_duration_mins": _number(value.get("windowDurationMins"), integer=True),
        "resets_at": _number(value.get("resetsAt"), integer=True),
    }


def _bucket(value: object) -> dict[str, object] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("bucket is not an object")
    return {
        "limit_id": _text(value.get("limitId"), 48),
        "limit_name": _text(value.get("limitName")),
        "model": _text(value.get("normalModelSlug")),
        "primary": _window(value.get("primary")),
        "secondary": _window(value.get("secondary")),
    }


def _account_identity(result: object) -> dict[str, object] | None:
    """
    Return the account fields that must agree across the read pair.

    :param result: ``account/read`` result, or ``None`` when it failed.
    :returns: Allowlisted account, routing account and user id, or ``None``.
    """
    if not isinstance(result, Mapping):
        return None
    account = result.get("account")
    routing = result.get("workspaceRouting")
    return {
        "account": (
            {
                "type": _text(account.get("type"), 32),
                "email": _text(account.get("email"), 320),
                "plan_type": _text(account.get("planType"), 64),
                "workspace_name": _text(account.get("workspaceName")),
            }
            if isinstance(account, Mapping)
            else None
        ),
        "routing_account_id": (
            _text(routing.get("chatgptAccountId")) if isinstance(routing, Mapping) else None
        ),
        "user_id": _text(result.get("userId")),
    }


def project_account_usage(
    account_before: object,
    rate_limits: Mapping[str, object],
    account_after: object,
    *,
    observed_at: float,
) -> dict[str, object] | None:
    """
    Reduce one read pair to the allowlisted ``codex_account_usage`` shape.

    :param account_before: ``account/read`` result before the quota read,
        or ``None`` when that read failed.
    :param rate_limits: ``account/rateLimits/read`` result.
    :param account_after: ``account/read`` result after the quota read, or
        ``None`` when that read failed.
    :param observed_at: Epoch seconds when the quota was read.
    :returns: The projection, or ``None`` when the account changed between
        the reads or a quota value is unreadable. Account fields are kept
        only when both account reads succeeded and agree.
    """
    before, after = _account_identity(account_before), _account_identity(account_after)
    if before is not None and after is not None and before != after:
        return None
    identity = after if before is not None else None
    try:
        by_limit = rate_limits.get("rateLimitsByLimitId")
        buckets: dict[str, object] | None = None
        if isinstance(by_limit, Mapping):
            buckets = {
                key: _bucket(value)
                for key, value in list(by_limit.items())[:_MAX_BUCKETS]
                if isinstance(key, str) and _BUCKET_ID.fullmatch(key)
            }
        elif by_limit is not None:
            return None
        legacy = _bucket(rate_limits.get("rateLimits"))
    except ValueError:
        return None
    if legacy is None and not buckets:
        return None
    return {
        "observed_at": observed_at,
        "codex_version": None,
        "account": identity["account"] if identity else None,
        "routing_account_id": identity["routing_account_id"] if identity else None,
        "account_id": _text(rate_limits.get("accountId")),
        "rate_limits": legacy,
        "rate_limits_by_limit_id": buckets,
    }


def _unauthorized(exc: CodexAppServerResponseError) -> bool:
    return exc.code == 401 or bool(_UNAUTHORIZED.search((exc.message or "").lower()))


def _rejects_field(exc: CodexAppServerResponseError, field: str) -> bool:
    return exc.code == -32602 or (
        exc.code == -32600 and f"unknown field `{field}`" in (exc.message or "")
    )


class CodexAccountUsageReader:
    """
    Coalesced, bounded account-usage reads on one app-server connection.

    :param client: The forwarder's connected app-server client.
    :param post: Relays one projection; returns whether it was accepted.
    :param min_interval: Seconds between reads, however many hints arrive.
    :param max_backoff: Longest delay after repeated failed reads.
    :param request_timeout: Bound for each app-server request.
    :param clock: Monotonic clock for scheduling.
    :param wall_clock: Epoch clock for ``observed_at``.
    """

    def __init__(
        self,
        client: CodexAppServerClient,
        post: PostAccountUsage,
        *,
        min_interval: float = _MIN_INTERVAL_SECONDS,
        max_backoff: float = _MAX_BACKOFF_SECONDS,
        request_timeout: float = _REQUEST_TIMEOUT_SECONDS,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
    ) -> None:
        self._client = client
        self._post = post
        self._min_interval = min_interval
        self._max_backoff = max_backoff
        self._request_timeout = request_timeout
        self._clock = clock
        self._wall_clock = wall_clock
        self._wake = asyncio.Event()
        self._wake.set()
        self._next_at = 0.0
        self._failures = 0
        self._renewed = False
        self._exclude_reset_credits = True
        self._unsupported = False

    def hint(self) -> None:
        """
        Note that the account's rate limits changed; a read follows when due.

        :returns: None.
        """
        self._wake.set()

    async def run(self) -> None:
        """
        Read now, then once per hint no sooner than the interval or backoff.

        :returns: None. Runs until cancelled or the session has no ChatGPT
            account.
        """
        while not self._unsupported:
            await self._wake.wait()
            delay = self._next_at - self._clock()
            if delay > 0:
                await asyncio.sleep(delay)
            # Hints from here on schedule the next read, not this one.
            self._wake.clear()
            try:
                relayed = await self._read_once()
            except (ConnectionError, OSError, RuntimeError, TimeoutError) as exc:
                _logger.warning("Codex account usage read failed: %s", type(exc).__name__)
                relayed = False
            if relayed:
                self._failures = 0
                self._next_at = self._clock() + self._min_interval
            else:
                self._failures += 1
                backoff = self._min_interval * 2 ** min(self._failures, 16)
                self._next_at = self._clock() + min(self._max_backoff, backoff)

    async def _request(self, method: str, params: dict[str, object]) -> object:
        response = await asyncio.wait_for(
            self._client.request(method, params), timeout=self._request_timeout
        )
        return response.get("result")

    async def _account(self) -> object:
        """
        Read the account, or ``None`` when only identity is unavailable.

        :returns: ``account/read`` result or ``None``.
        :raises CodexAppServerResponseError: If the token was rejected.
        """
        try:
            return await self._request("account/read", {"refreshToken": False})
        except CodexAppServerResponseError as exc:
            if _unauthorized(exc):
                raise
            return None

    async def _rate_limits(self) -> object:
        if self._exclude_reset_credits:
            try:
                return await self._request(
                    "account/rateLimits/read", {_EXCLUDE_RESET_CREDITS: True}
                )
            except CodexAppServerResponseError as exc:
                if not _rejects_field(exc, _EXCLUDE_RESET_CREDITS):
                    raise
                self._exclude_reset_credits = False
        return await self._request("account/rateLimits/read", {})

    async def _read_once(self) -> bool:
        try:
            before = await self._account()
            if isinstance(before, Mapping):
                account = before.get("account")
                if isinstance(account, Mapping) and account.get("type") != "chatgpt":
                    # API-key and other modes have no ChatGPT account usage.
                    self._unsupported = True
                    return False
            limits = await self._rate_limits()
            observed_at = self._wall_clock()
            after = await self._account()
        except CodexAppServerResponseError as exc:
            if not _unauthorized(exc) or self._renewed:
                _logger.warning("Codex account usage read was refused (code=%s)", exc.code)
                return False
            # One native renewal for this session; never a rotation loop.
            self._renewed = True
            with contextlib.suppress(CodexAppServerResponseError):
                await self._request("account/read", {"refreshToken": True})
            return await self._read_once()
        if not isinstance(limits, Mapping):
            return False
        projection = project_account_usage(before, limits, after, observed_at=observed_at)
        if projection is None:
            return False
        return await self._post(projection)


__all__ = [
    "ACCOUNT_RATE_LIMITS_UPDATED_METHOD",
    "CodexAccountUsageReader",
    "project_account_usage",
]
