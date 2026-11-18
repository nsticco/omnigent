"""Claude Code statusLine wrapper that captures context-window and quota data.

Claude Code only exposes ``context_window`` (the model's real window
size and live usage) and the account's plan ``rate_limits`` to the
``statusLine`` command via stdin. This wrapper reads that stdin, writes
the relevant fields atomically to ``bridge_dir/context.json`` for the
forwarder to consume, then exec's the user's pre-existing statusLine
command (if any) with the same stdin so its terminal output still
renders.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

_CONTEXT_FILE = "context.json"
# Raw statusLine stdin captured by the shell shim the settings now install
# (no Python spawn on Claude's blocking statusLine path). The forwarder
# normalizes it into ``context.json`` via :func:`sync_raw_status_context`.
CONTEXT_RAW_FILE = "context_raw.json"
_MAX_VERSION_CHARS = 32
# Plan rate-limit windows the statusLine reports. Other keys are ignored:
# an unrecognized window is never reported as one of these.
RATE_LIMIT_WINDOWS = ("five_hour", "seven_day")


def normalize_status_payload(
    payload: dict[str, object],
    previous: dict[str, object] | None = None,
    captured_at: float | None = None,
) -> dict[str, object] | None:
    """
    Extract the ``context.json`` record from a raw statusLine payload.

    Context fields and plan quota are read independently, so quota still
    records when ``context_window`` is absent (the previous record's
    context fields are kept) and vice versa.

    :param payload: Decoded statusLine stdin JSON from Claude Code.
    :param previous: The record currently in ``context.json``, if any.
        Its quota windows keep their capture time while their values are
        unchanged, because Claude Code re-renders cached rate limits.
    :param captured_at: Epoch seconds when this payload was captured;
        defaults to now.
    :returns: The record to persist, or ``None`` when the payload carries
        neither a usable ``context_window`` nor quota.
    """
    record = _context_record(payload)
    rate_limits = normalize_rate_limits(
        payload,
        (previous or {}).get("rate_limits"),
        time.time() if captured_at is None else captured_at,
    )
    if record is None:
        if rate_limits is None:
            return None
        record = {key: value for key, value in (previous or {}).items() if key != "rate_limits"}
    if rate_limits is not None:
        record["rate_limits"] = rate_limits
    return record


def normalize_rate_limits(
    payload: dict[str, object],
    previous: object,
    captured_at: float,
) -> dict[str, object] | None:
    """
    Read ``rate_limits.five_hour`` / ``seven_day`` from a statusLine payload.

    ``used_percentage`` is 0-100 and ``resets_at`` is epoch seconds; each is
    independently nullable. Every window carries ``observed_at``: the time
    its values last changed. A re-render of unchanged cached values keeps
    the earlier time, and a window this payload omits keeps its previous
    reading, so freshness only advances with newly observed provider data.

    :param payload: Decoded statusLine stdin JSON from Claude Code.
    :param previous: The previous record's ``rate_limits`` block, if any.
    :param captured_at: Epoch seconds when this payload was captured.
    :returns: ``{"five_hour": {...}, "seven_day": {...},
        "claude_code_version": "2.1.281"}`` with only the windows known so
        far, or ``None`` when no window has been reported.
    """
    known = previous if isinstance(previous, dict) else {}
    raw = payload.get("rate_limits")
    raw = raw if isinstance(raw, dict) else {}
    windows: dict[str, object] = {}
    for key in RATE_LIMIT_WINDOWS:
        before = known.get(key)
        values = _rate_limit_values(raw.get(key))
        if values is None:
            if isinstance(before, dict):
                windows[key] = before
            continue
        unchanged = isinstance(before, dict) and all(
            before.get(field) == value for field, value in values.items()
        )
        observed_at = before.get("observed_at") if unchanged and isinstance(before, dict) else None
        windows[key] = {
            **values,
            "observed_at": observed_at if isinstance(observed_at, (int, float)) else captured_at,
        }
    if not windows:
        return None
    version = payload.get("version")
    if isinstance(version, str) and 0 < len(version) <= _MAX_VERSION_CHARS:
        windows["claude_code_version"] = version
    elif isinstance(known.get("claude_code_version"), str):
        windows["claude_code_version"] = known["claude_code_version"]
    return windows


def _finite_number(value: object) -> float | None:
    """Return ``value`` as a finite float, or ``None`` for anything else."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def _rate_limit_values(window: object) -> dict[str, object] | None:
    """Return a window's valid fields, or ``None`` when it reports none."""
    if not isinstance(window, dict):
        return None
    used = _finite_number(window.get("used_percentage"))
    resets = _finite_number(window.get("resets_at"))
    used = used if used is not None and used >= 0 else None
    resets = resets if resets is not None and resets > 0 else None
    if used is None and resets is None:
        return None
    return {
        "used_percentage": used,
        "resets_at": int(resets) if resets is not None else None,
    }


def read_rate_limits(bridge_dir: Path) -> dict[str, object] | None:
    """
    Read the quota block from ``context.json``, independent of context.

    :param bridge_dir: Bridge directory shared with the statusLine shim.
    :returns: The ``rate_limits`` block, or ``None`` when absent/unreadable.
    """
    rate_limits = (_read_record(bridge_dir) or {}).get("rate_limits")
    return rate_limits if isinstance(rate_limits, dict) else None


def _read_record(bridge_dir: Path) -> dict[str, object] | None:
    try:
        parsed = json.loads((bridge_dir / _CONTEXT_FILE).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _context_record(payload: dict[str, object]) -> dict[str, object] | None:
    """Context, cost, model and version fields, when ``context_window`` is usable."""
    context = payload.get("context_window")
    if not isinstance(context, dict):
        return None
    size = context.get("context_window_size")
    usage = context.get("current_usage")
    if not isinstance(size, int) or size <= 0:
        return None
    record: dict[str, object] = {"context_window_size": size}
    if isinstance(usage, dict):
        record["current_usage"] = usage
    used_pct = context.get("used_percentage")
    if isinstance(used_pct, (int, float)):
        record["used_percentage"] = used_pct
    # Claude Code's statusLine stdin carries a top-level ``cost`` block with
    # its own cumulative session billing; the forwarder reports it because
    # claude-native produces no ``response.completed`` cost events.
    cost = payload.get("cost")
    if isinstance(cost, dict):
        total_cost = cost.get("total_cost_usd")
        if (
            isinstance(total_cost, (int, float))
            and not isinstance(total_cost, bool)
            and total_cost >= 0
        ):
            record["total_cost_usd"] = float(total_cost)
    # The active model, rewritten on every render — including right after an
    # in-pane ``/model`` switch — so gates see the switch before the next turn.
    model = payload.get("model")
    model_id: str | None = None
    if isinstance(model, dict):
        raw_model = model.get("id") or model.get("display_name")
        if isinstance(raw_model, str) and raw_model.strip():
            model_id = raw_model.strip()
    elif isinstance(model, str) and model.strip():
        model_id = model.strip()
    if model_id is not None:
        record["model"] = model_id
    # The Claude Code build behind the pane, so delivery diagnostics can tell versions apart.
    version = payload.get("version")
    if isinstance(version, str) and 0 < len(version) <= _MAX_VERSION_CHARS:
        record["claude_code_version"] = version
    return record


def sync_raw_status_context(
    bridge_dir: Path,
    last_sig: tuple[int, int] | None,
) -> tuple[int, int] | None:
    """
    Normalize the shim's raw statusLine capture into ``context.json``.

    Called from the forwarder's poll loop. Cheap when nothing changed —
    one ``stat`` against the remembered ``(mtime_ns, size)`` signature.

    :param bridge_dir: Bridge directory shared with the statusLine shim.
    :param last_sig: Signature returned by the previous call, or ``None``.
    :returns: The new signature to carry forward (unchanged on a miss or
        an unparseable file, so the next poll retries).
    """
    raw_path = bridge_dir / CONTEXT_RAW_FILE
    try:
        stat = raw_path.stat()
    except OSError:
        return last_sig
    sig = (stat.st_mtime_ns, stat.st_size)
    if sig == last_sig:
        return last_sig
    try:
        payload = json.loads(raw_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return last_sig
    if not isinstance(payload, dict):
        return sig
    # The shim wrote this render at the file's mtime; a quota window that
    # changed is stamped with that capture time, not with this poll.
    record = normalize_status_payload(payload, _read_record(bridge_dir), stat.st_mtime_ns / 1e9)
    if record is not None:
        _write_record_atomic(bridge_dir, record)
    return sig


def main(argv: list[str] | None = None) -> int:
    """
    Capture Claude Code's statusLine stdin and chain to the user's command.

    :param argv: Optional argv override (excluding program name).
    :returns: ``0`` on success, ``0`` after a soft failure (the
        statusLine must never crash Claude Code's TUI loop).
    """
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    raw = sys.stdin.read()
    payload: dict[str, object] | None = None
    try:
        parsed = json.loads(raw) if raw.strip() else None
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        payload = parsed
    if payload is not None:
        _write_context_atomic(Path(args.bridge_dir), payload)
    if args.chain:
        _chain(args.chain, raw)
    return 0


def _parse_args(argv: list[str]) -> argparse.Namespace:
    """
    Parse wrapper arguments.

    :param argv: CLI argv excluding program name.
    :returns: Parsed namespace with ``bridge_dir`` and optional ``chain``.
    """
    parser = argparse.ArgumentParser(prog="python -m omnigent.harnesses.claude_native.status")
    parser.add_argument("--bridge-dir", required=True)
    # The user's original statusLine command (as a shell string). When
    # set we exec it with the same stdin so claude-hud / their custom
    # status bar still renders for omnigent-launched sessions.
    parser.add_argument("--chain", default=None)
    return parser.parse_args(argv)


def _write_context_atomic(bridge_dir: Path, payload: dict[str, object]) -> None:
    """
    Persist the statusLine payload's context fields to ``context.json``.

    Atomic write so the forwarder never observes a half-written file.
    Soft-fails (writes nothing) when the payload carries no usable
    ``context_window``. Retained for older bridge dirs whose settings
    still invoke this module; new settings install a shell shim and the
    forwarder normalizes via :func:`sync_raw_status_context`.

    :param bridge_dir: Bridge directory shared with the forwarder.
    :param payload: Decoded statusLine stdin JSON.
    """
    record = normalize_status_payload(payload, _read_record(bridge_dir))
    if record is None:
        return
    try:
        _write_record_atomic(bridge_dir, record)
    except OSError as exc:
        print(f"omnigent claude status: write failed: {exc}", file=sys.stderr)


def _write_record_atomic(bridge_dir: Path, record: dict[str, object]) -> None:
    """
    Atomically write one normalized record to ``context.json``.

    :param bridge_dir: Bridge directory shared with the forwarder.
    :param record: Normalized record from :func:`normalize_status_payload`.
    :raises OSError: When the temp-file write or replace fails.
    """
    bridge_dir.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".context-", dir=str(bridge_dir))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(record, handle, separators=(",", ":"))
        os.replace(tmp_path, str(bridge_dir / _CONTEXT_FILE))
    except OSError:
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)
        raise


def _chain(command: str, stdin_payload: str) -> None:
    """
    Exec the user's pre-existing statusLine command, piping our stdin.

    :param command: Shell command string from the user's
        ``~/.claude/settings.json`` ``statusLine.command``.
    :param stdin_payload: The original stdin we received from Claude
        Code. Forwarded verbatim so the chained command sees exactly
        what Claude Code sent.
    """
    try:
        proc = subprocess.run(
            command,
            input=stdin_payload,
            shell=True,
            capture_output=True,
            text=True,
            check=False,
            timeout=5.0,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"omnigent claude status: chain failed: {exc}", file=sys.stderr)
        return
    if proc.stdout:
        sys.stdout.write(proc.stdout)
    if proc.stderr:
        sys.stderr.write(proc.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
