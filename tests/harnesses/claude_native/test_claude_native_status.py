"""Tests for the Claude Code statusLine wrapper."""

from __future__ import annotations

import io
import json
import subprocess
import sys
from pathlib import Path

import pytest

from omnigent.harnesses.claude_native import status as claude_native_status


def _run(
    *,
    stdin_payload: str,
    bridge_dir: Path,
    chain: str | None = None,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str] | None = None,
) -> int:
    """
    Invoke the wrapper's ``main()`` with a stubbed stdin.

    :param stdin_payload: Raw stdin text the wrapper will read.
    :param bridge_dir: Bridge directory the wrapper writes into.
    :param chain: Optional chained command, e.g. ``"echo claude-hud"``.
    :param monkeypatch: Pytest fixture for patching ``sys.stdin``.
    :param capsys: Pytest stdout/stderr capture fixture, unused here.
    :returns: Process exit code from ``main()``.
    """
    del capsys
    monkeypatch.setattr(sys, "stdin", io.StringIO(stdin_payload))
    argv = ["--bridge-dir", str(bridge_dir)]
    if chain is not None:
        argv.extend(["--chain", chain])
    return claude_native_status.main(argv)


def test_status_wrapper_writes_context_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    The wrapper persists ``context_window_size`` and ``current_usage``.

    These are the two fields the forwarder consumes; missing either
    silently breaks the ring on a real claude-native session, so this
    test pins the exact wire shape Claude Code provides on statusLine
    stdin (modeled on claude-hud's reverse-engineered schema).
    """
    stdin = json.dumps(
        {
            "session_id": "abc",
            "model": {"display_name": "Opus 4.7"},
            "context_window": {
                "context_window_size": 1_000_000,
                "current_usage": {
                    "input_tokens": 6,
                    "cache_creation_input_tokens": 100,
                    "cache_read_input_tokens": 200,
                    "output_tokens": 50,
                },
                "used_percentage": 31,
            },
        }
    )

    rc = _run(stdin_payload=stdin, bridge_dir=tmp_path, monkeypatch=monkeypatch)
    assert rc == 0

    persisted = json.loads((tmp_path / "context.json").read_text(encoding="utf-8"))
    assert persisted["context_window_size"] == 1_000_000
    assert persisted["current_usage"]["input_tokens"] == 6
    assert persisted["current_usage"]["cache_creation_input_tokens"] == 100
    assert persisted["used_percentage"] == 31


def test_status_wrapper_captures_cost(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    The wrapper persists Claude Code's cumulative ``cost.total_cost_usd``.

    Claude Code's statusLine stdin carries a top-level ``cost`` block with its
    own session billing. claude-native never produces a ``response.completed``
    event, so the Omnigent relay's cost accumulation never runs for it — capturing
    this is the only way native session cost reaches ``session_usage``. A
    failure here means native Cost-Ask policies always see $0.
    """
    stdin = json.dumps(
        {
            "session_id": "abc",
            "context_window": {"context_window_size": 1_000_000},
            "cost": {"total_cost_usd": 0.42, "total_duration_ms": 1234},
        }
    )

    rc = _run(stdin_payload=stdin, bridge_dir=tmp_path, monkeypatch=monkeypatch)
    assert rc == 0

    persisted = json.loads((tmp_path / "context.json").read_text(encoding="utf-8"))
    assert persisted["total_cost_usd"] == 0.42


def test_status_wrapper_drops_payload_without_context_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Payloads with no ``context_window`` block leave no file behind.

    Older Claude Code versions (pre v2.x) didn't include ``context_window``
    on statusLine stdin. The wrapper must degrade gracefully so the
    ring keeps the spec default rather than rendering a half-init state.
    """
    stdin = json.dumps({"session_id": "abc", "model": {"display_name": "Opus 4.7"}})

    rc = _run(stdin_payload=stdin, bridge_dir=tmp_path, monkeypatch=monkeypatch)
    assert rc == 0
    assert not (tmp_path / "context.json").exists()


def test_status_wrapper_chains_to_user_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """
    The chained command receives the original stdin and renders its stdout.

    Claude Code only invokes a single statusLine command. Without
    chaining, overriding for context capture would silently hide
    claude-hud / any user-installed status bar.
    """
    captured_calls: list[dict[str, object]] = []

    class _FakeProc:
        returncode = 0
        stdout = "claude-hud line\n"
        stderr = ""

    def fake_run(*args: object, **kwargs: object) -> _FakeProc:
        captured_calls.append({"args": args, "kwargs": kwargs})
        return _FakeProc()

    monkeypatch.setattr(subprocess, "run", fake_run)

    stdin = json.dumps({"context_window": {"context_window_size": 200_000}})
    rc = _run(
        stdin_payload=stdin,
        bridge_dir=tmp_path,
        chain="echo claude-hud",
        monkeypatch=monkeypatch,
    )
    assert rc == 0
    assert len(captured_calls) == 1
    assert captured_calls[0]["args"] == ("echo claude-hud",)
    assert captured_calls[0]["kwargs"]["input"] == stdin
    assert captured_calls[0]["kwargs"]["shell"] is True
    out, _err = capsys.readouterr()
    assert "claude-hud line" in out


def test_status_wrapper_chain_swallows_subprocess_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """
    Chained-command failures don't propagate — context capture stays best-effort.

    The statusLine command runs on every Claude Code render tick. A
    crash there would yank the user's terminal status bar away on
    every tick; the wrapper logs to stderr and returns 0 instead.
    """

    def fake_run(*args: object, **kwargs: object) -> None:
        raise OSError("chain broken")

    monkeypatch.setattr(subprocess, "run", fake_run)
    stdin = json.dumps({"context_window": {"context_window_size": 200_000}})
    rc = _run(
        stdin_payload=stdin,
        bridge_dir=tmp_path,
        chain="bogus",
        monkeypatch=monkeypatch,
    )
    assert rc == 0
    _out, err = capsys.readouterr()
    assert "chain failed" in err


def test_normalize_status_payload_extracts_record() -> None:
    """The normalizer extracts window/usage/cost/model; None without a window."""
    from omnigent.harnesses.claude_native.status import normalize_status_payload

    record = normalize_status_payload(
        {
            "context_window": {
                "context_window_size": 200_000,
                "current_usage": {"input_tokens": 5},
                "used_percentage": 2.5,
            },
            "cost": {"total_cost_usd": 0.42},
            "model": {"id": "claude-opus-4-8", "display_name": "Opus"},
        }
    )
    assert record == {
        "context_window_size": 200_000,
        "current_usage": {"input_tokens": 5},
        "used_percentage": 2.5,
        "total_cost_usd": 0.42,
        "model": "claude-opus-4-8",
    }
    assert normalize_status_payload({"model": "claude-opus-4-8"}) is None


@pytest.mark.parametrize(
    ("version", "kept"),
    [("2.1.299", True), ("", False), (2.1, False), ("9" * 33, False)],
)
def test_normalize_status_payload_keeps_a_short_claude_code_version(
    version: object, kept: bool
) -> None:
    """The statusLine's Claude Code version is kept for delivery diagnostics."""
    from omnigent.harnesses.claude_native.status import normalize_status_payload

    record = normalize_status_payload(
        {"context_window": {"context_window_size": 200_000}, "version": version}
    )
    assert record is not None
    assert record.get("claude_code_version") == (version if kept else None)


def test_sync_raw_status_context_normalizes_and_retries(tmp_path: Path) -> None:
    """The forwarder-side sync normalizes raw captures and tolerates junk.

    Unchanged signatures are no-ops, a malformed raw file leaves the
    signature untouched so the next poll retries, and a rewritten raw
    file re-normalizes.
    """
    import json as _json

    from omnigent.harnesses.claude_native.status import (
        CONTEXT_RAW_FILE,
        sync_raw_status_context,
    )

    raw_path = tmp_path / CONTEXT_RAW_FILE
    payload = {
        "context_window": {"context_window_size": 1000},
        "model": {"id": "claude-opus-4-8"},
    }
    raw_path.write_text(_json.dumps(payload, indent=2), encoding="utf-8")

    sig = sync_raw_status_context(tmp_path, None)
    assert sig is not None
    written = _json.loads((tmp_path / "context.json").read_text("utf-8"))
    assert written == {"context_window_size": 1000, "model": "claude-opus-4-8"}

    # Unchanged raw file: same signature back, nothing rewritten.
    assert sync_raw_status_context(tmp_path, sig) == sig

    # Malformed raw file: signature unchanged so the next poll retries.
    raw_path.write_text("{not json", encoding="utf-8")
    assert sync_raw_status_context(tmp_path, sig) == sig

    # Rewritten raw file: re-normalized under a fresh signature.
    payload["model"] = {"id": "claude-sonnet-4-6"}
    raw_path.write_text(_json.dumps(payload), encoding="utf-8")
    new_sig = sync_raw_status_context(tmp_path, sig)
    assert new_sig != sig
    written = _json.loads((tmp_path / "context.json").read_text("utf-8"))
    assert written["model"] == "claude-sonnet-4-6"


def _quota(five_hour: object = None, seven_day: object = None, **extra: object) -> dict:
    """A statusLine payload carrying only plan ``rate_limits``."""
    rate_limits: dict[str, object] = {}
    if five_hour is not None:
        rate_limits["five_hour"] = five_hour
    if seven_day is not None:
        rate_limits["seven_day"] = seven_day
    return {"session_id": "abc", "version": "2.1.281", "rate_limits": rate_limits, **extra}


def test_weekly_quota_records_without_context_window_and_still_chains(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Plan quota is captured even when ``context_window`` is absent.

    The user's statusLine command still renders from the same stdin.
    """
    stdin = json.dumps(_quota(seven_day={"used_percentage": 42, "resets_at": 1795000000}))

    rc = _run(
        stdin_payload=stdin,
        bridge_dir=tmp_path,
        chain="printf hud-rendered",
        monkeypatch=monkeypatch,
    )

    assert rc == 0
    assert capsys.readouterr().out == "hud-rendered"
    persisted = json.loads((tmp_path / "context.json").read_text(encoding="utf-8"))
    weekly = persisted["rate_limits"]["seven_day"]
    assert (weekly["used_percentage"], weekly["resets_at"]) == (42.0, 1795000000)
    assert isinstance(weekly["observed_at"], float)
    assert persisted["rate_limits"]["claude_code_version"] == "2.1.281"
    assert "five_hour" not in persisted["rate_limits"]
    assert "context_window_size" not in persisted
    assert claude_native_status.read_rate_limits(tmp_path) == persisted["rate_limits"]


def test_cached_quota_re_renders_keep_their_capture_time() -> None:
    """Re-rendered cached quota never advances observation freshness.

    Each window keeps the time its values last changed, so a newer short
    window does not refresh an unchanged weekly window.
    """
    first = claude_native_status.normalize_status_payload(
        _quota(
            five_hour={"used_percentage": 10, "resets_at": 1794000000},
            seven_day={"used_percentage": 40, "resets_at": 1795000000},
        ),
        None,
        100.0,
    )
    again = claude_native_status.normalize_status_payload(
        _quota(
            five_hour={"used_percentage": 10, "resets_at": 1794000000},
            seven_day={"used_percentage": 40, "resets_at": 1795000000},
        ),
        first,
        200.0,
    )
    assert again["rate_limits"] == first["rate_limits"]

    newer = claude_native_status.normalize_status_payload(
        _quota(
            five_hour={"used_percentage": 11, "resets_at": 1794000000},
            seven_day={"used_percentage": 40, "resets_at": 1795000000},
        ),
        again,
        300.0,
    )
    assert newer["rate_limits"]["five_hour"]["observed_at"] == 300.0
    assert newer["rate_limits"]["seven_day"]["observed_at"] == 100.0

    # A render without quota keeps the last reading and its time.
    later = claude_native_status.normalize_status_payload(
        {"context_window": {"context_window_size": 1000}}, newer, 400.0
    )
    assert later["context_window_size"] == 1000
    assert later["rate_limits"] == newer["rate_limits"]


def test_quota_render_keeps_the_previous_context_fields() -> None:
    previous = claude_native_status.normalize_status_payload(
        {"context_window": {"context_window_size": 1000}, "model": {"id": "claude-x"}},
        None,
        1.0,
    )
    record = claude_native_status.normalize_status_payload(
        _quota(seven_day={"used_percentage": 3, "resets_at": None}), previous, 2.0
    )
    assert record["context_window_size"] == 1000
    assert record["model"] == "claude-x"
    assert record["rate_limits"]["seven_day"] == {
        "used_percentage": 3.0,
        "resets_at": None,
        "observed_at": 2.0,
    }


@pytest.mark.parametrize(
    "window",
    [
        {"used_percentage": True, "resets_at": False},
        {"used_percentage": "42", "resets_at": "soon"},
        {"used_percentage": -1, "resets_at": 0},
        {"used_percentage": float("nan"), "resets_at": None},
        "42%",
    ],
)
def test_unreadable_quota_windows_are_not_recorded(window: object) -> None:
    assert (
        claude_native_status.normalize_status_payload(_quota(seven_day=window), None, 1.0) is None
    )


def test_unknown_rate_limit_keys_are_ignored() -> None:
    record = claude_native_status.normalize_status_payload(
        {"rate_limits": {"seven_day_opus": {"used_percentage": 9, "resets_at": 1}}},
        None,
        1.0,
    )
    assert record is None


def test_sync_stamps_changed_quota_with_the_capture_time(tmp_path: Path) -> None:
    """The forwarder stamps new quota with the shim's write time, not its poll."""
    import os

    from omnigent.harnesses.claude_native.status import CONTEXT_RAW_FILE, sync_raw_status_context

    raw_path = tmp_path / CONTEXT_RAW_FILE
    raw_path.write_text(
        json.dumps(_quota(seven_day={"used_percentage": 5, "resets_at": 1795000000})),
        encoding="utf-8",
    )
    os.utime(raw_path, (1_700_000_000, 1_700_000_000))
    sig = sync_raw_status_context(tmp_path, None)
    weekly = claude_native_status.read_rate_limits(tmp_path)["seven_day"]
    assert weekly["observed_at"] == 1_700_000_000.0

    # A later render of the same cached values keeps the original time.
    raw_path.write_text(
        json.dumps(_quota(seven_day={"used_percentage": 5, "resets_at": 1795000000}, cost={})),
        encoding="utf-8",
    )
    os.utime(raw_path, (1_700_000_900, 1_700_000_900))
    assert sync_raw_status_context(tmp_path, sig) != sig
    assert claude_native_status.read_rate_limits(tmp_path)["seven_day"]["observed_at"] == (
        1_700_000_000.0
    )
