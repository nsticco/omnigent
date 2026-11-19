"""Isolated installed-CLI probe; only connects to its local mock provider."""

import asyncio
import json
import os
import pathlib
import shutil
import socket
import subprocess
import sys
import time
import uuid

import httpx


def main() -> None:
    root, cli, temporary, mode, base, projected, selected, timeout, *extra = json.loads(
        sys.argv[1]
    )
    cancel = bool(extra and extra[0])
    p = pathlib.Path(temporary)
    p.mkdir(parents=True)
    os.environ["HOME"] = str(p / "runner")
    pathlib.Path(os.environ["HOME"]).mkdir()
    from omnigent.harnesses.claude_native.bridge import (
        augment_claude_args,
        bridge_dir_for_bridge_id,
        prepare_bridge_dir,
    )

    home = p / "home"
    home.mkdir()
    config = home / ".claude"
    config.mkdir()
    workspace = p / "workspace"
    workspace.mkdir()
    command = {
        "type": "command",
        "command": ("sleep 2; " if timeout else "")
        + "printf PROFILE_HOOK_RAN >> "
        + str(workspace / "hook-marker"),
    }
    if timeout:
        command["timeout"] = 0.1
    if cancel:
        command["command"] = (
            "printf '%s' $$ > "
            + str(workspace / "hook-pid")
            + "; sleep 3 & printf '%s' $! > "
            + str(workspace / "child-pid")
            + "; wait; printf FINISHED > "
            + str(workspace / "hook-marker")
        )
    selected["hooks"] = {"PostToolUse": [{"matcher": "Bash", "hooks": [command]}]}
    authored = selected["hooks"]
    if mode != "native":
        selected.pop("hooks")
    (config / "settings.json").write_text(json.dumps(selected))
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    url = f"http://127.0.0.1:{port}"
    client = httpx.Client(trust_env=False)
    server = subprocess.Popen(
        [
            sys.executable,
            str(pathlib.Path(root) / "tests/server/integration/mock_llm_server.py"),
            str(port),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    bridge = None
    try:
        for _ in range(100):
            try:
                if client.get(url + "/stats").status_code == 200:
                    break
            except httpx.ConnectError:
                pass
            time.sleep(0.1)
        else:
            raise RuntimeError("local mock API did not start")
        args = base
        if mode != "native":
            bridge_id = "env-test-" + uuid.uuid4().hex
            bridge = bridge_dir_for_bridge_id(bridge_id)
            prepare_bridge_dir(bridge_id, workspace=workspace)
            args = augment_claude_args(
                base,
                bridge_dir=bridge,
                python_executable=sys.executable,
                skills_filter=mode,
                authored_hooks=authored,
            )
            generated = json.loads(pathlib.Path(args[args.index("--settings") + 1]).read_text())
            assert "SessionStart" in generated["hooks"]
            assert "Stop" in generated["hooks"]
            assert args.count("--settings") == 1
        body = {
            "command": (
                'printf "label=%s only=%s sibling=%s" '
                '"$PROJECT_LABEL" "$PROJECT_ONLY" "$PROJECT_SIBLING"'
            ),
            "description": "Inspect nonsecret project values",
        }
        client.post(
            url + "/mock/configure",
            json={
                "responses": [
                    {"tool_calls": [{"name": "Bash", "arguments": json.dumps(body)}]},
                    {"text": "done"},
                ]
            },
        ).raise_for_status()
        env = {
            "PATH": os.defpath,
            "HOME": str(home),
            "CLAUDE_CONFIG_DIR": str(config),
            "ANTHROPIC_BASE_URL": url,
            "ANTHROPIC_API_KEY": "mock-key",
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
            "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS": "1",
            "HTTP_PROXY": "http://127.0.0.1:1",
            "HTTPS_PROXY": "http://127.0.0.1:1",
            "ALL_PROXY": "http://127.0.0.1:1",
            "NO_PROXY": "127.0.0.1,localhost",
            "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
        }
        if cancel:
            env["PATH"] = str(pathlib.Path(shutil.which("tmux")).parent) + os.pathsep + os.defpath
        env.update(
            {"PROJECT_LABEL": "process-value", "PROJECT_SIBLING": "process-sibling"}
            if mode == "native"
            else projected
        )
        command_args = [
            cli,
            "-p",
            "Complete the local test",
            "--output-format",
            "stream-json",
            "--verbose",
            *args,
        ]
        if cancel:
            from omnigent.inner.terminal import TerminalInstance

            async def cancel_running_hook():
                private = p / "terminal"
                private.mkdir(mode=0o700)
                terminal = TerminalInstance(
                    name="claude",
                    session_key="main",
                    socket_path=private / "tmux.sock",
                    private_dir=private,
                    command=cli,
                    args=command_args[1:],
                    env=env,
                    inherit_env=False,
                )
                try:
                    await terminal.launch(cwd=workspace)
                    for _ in range(300):
                        if (workspace / "child-pid").exists():
                            break
                        await asyncio.sleep(0.1)
                    else:
                        raise AssertionError("hook did not start in managed terminal")
                    hook_pid = int((workspace / "hook-pid").read_text())
                    child_pid = int((workspace / "child-pid").read_text())

                    def alive(pid):
                        try:
                            state = pathlib.Path(f"/proc/{pid}/stat").read_text()
                        except FileNotFoundError:
                            return False
                        return state.split(")", 1)[1].strip()[0] != "Z"

                    assert alive(hook_pid) and alive(child_pid)
                    await terminal.close()
                    for _ in range(20):
                        if not alive(hook_pid) and not alive(child_pid):
                            break
                        await asyncio.sleep(0.1)
                    assert not alive(hook_pid), "hook shell survived managed terminal close"
                    assert not alive(child_pid), "hook child survived managed terminal close"
                    await asyncio.sleep(3.2)
                    assert not (workspace / "hook-marker").exists(), (
                        "cancelled hook wrote delayed side effect"
                    )
                    assert not terminal.running and not private.exists()
                finally:
                    await terminal.close()

            asyncio.run(cancel_running_hook())
            print(json.dumps({"mode": mode, "managedTerminalCancellationVerified": True}))
            return
        run = subprocess.run(
            command_args,
            cwd=workspace,
            env=env,
            text=True,
            capture_output=True,
            timeout=45,
            check=False,
        )
        assert run.returncode == 0, run.stderr[-1000:]

        def nodes(value):
            if isinstance(value, dict):
                yield value
                for item in value.values():
                    yield from nodes(item)
            elif isinstance(value, list):
                for item in value:
                    yield from nodes(item)

        entries = list(nodes(client.get(url + "/mock/message_requests").json()))
        uses = {
            item["id"]
            for item in entries
            if item.get("type") == "tool_use"
            and item.get("name") == "Bash"
            and item.get("input", {}).get("command") == body["command"]
        }
        results = [
            item
            for item in entries
            if item.get("type") == "tool_result" and item.get("tool_use_id") in uses
        ]
        assert uses and results, "Bash was not actually attempted"
        for result in results:
            assert not result.get("is_error"), result
            assert "sibling=process-sibling" in json.dumps(result), result
        if timeout:
            time.sleep(2.5)
            assert not (workspace / "hook-marker").exists()
        else:
            assert (workspace / "hook-marker").read_text() == "PROFILE_HOOK_RAN"
        print(
            json.dumps(
                {
                    "hookMarker": (workspace / "hook-marker").read_text()
                    if (workspace / "hook-marker").exists()
                    else None,
                    "version": subprocess.check_output(
                        [cli, "--version"], text=True, env=env
                    ).strip(),
                    "mode": mode,
                    "actualToolOutputVerified": True,
                }
            )
        )
    finally:
        server.terminate()
        server.wait(timeout=5)
        if bridge:
            shutil.rmtree(bridge)


if __name__ == "__main__":
    main()
