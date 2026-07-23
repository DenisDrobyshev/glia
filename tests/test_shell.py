from __future__ import annotations

import json
import socket
import threading
import urllib.request
from http.server import ThreadingHTTPServer

from conftest import run

from glia.providers import EchoLLM
from glia.shell import config as config_mod
from glia.shell.backend import build_agent
from glia.shell.config import Config
from glia.shell.server import ShellState, make_handler

# --- config -------------------------------------------------------------------


def test_config_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(config_mod, "config_dir", lambda: tmp_path)
    Config(mode="claude", anthropic_api_key="secret", model="claude-sonnet-5").save()
    loaded = Config.load()
    assert loaded.mode == "claude"
    assert loaded.model == "claude-sonnet-5"
    assert loaded.anthropic_api_key == "secret"


def test_public_config_hides_key():
    pub = Config(anthropic_api_key="secret").public()
    assert "anthropic_api_key" not in pub
    assert pub["has_key"] is True


def test_config_falls_back_on_corrupt_file(tmp_path, monkeypatch):
    monkeypatch.setattr(config_mod, "config_dir", lambda: tmp_path)
    tmp_path.mkdir(exist_ok=True)
    config_mod.config_path().write_text("{ not valid json", "utf-8")
    assert Config.load().mode == "demo"  # defaults, no crash


# --- agent construction -------------------------------------------------------


def test_build_agent_offline_uses_echo_and_streams():
    agent = build_agent(Config(mode="demo"))
    assert isinstance(agent.llm, EchoLLM)
    assert agent.stream is True
    assert "hello" in run(agent.run("hello")).output


def test_build_agent_claude_without_key_falls_back_to_demo():
    assert isinstance(build_agent(Config(mode="claude", anthropic_api_key="")).llm, EchoLLM)


def test_build_agent_claude_with_key_registers_tools():
    agent = build_agent(Config(mode="claude", anthropic_api_key="k"))
    assert type(agent.llm).__name__ == "ClaudeLLM"
    assert len(agent.tools) == 2  # demo tools


def test_build_agent_ollama_uses_ollama_provider():
    agent = build_agent(Config(mode="ollama", ollama_model="deepseek-r1", ollama_host="http://h:1"))
    assert type(agent.llm).__name__ == "OllamaLLM"
    assert agent.llm.model == "deepseek-r1"
    assert agent.llm.host == "http://h:1"


def test_config_persists_ollama_fields(tmp_path, monkeypatch):
    monkeypatch.setattr(config_mod, "config_dir", lambda: tmp_path)
    Config(mode="ollama", ollama_model="qwen2.5", ollama_host="http://x:11434").save()
    loaded = Config.load()
    assert loaded.ollama_model == "qwen2.5"
    assert loaded.ollama_host == "http://x:11434"
    assert loaded.public()["ollama_model"] == "qwen2.5"


# --- live server --------------------------------------------------------------


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _serve(state: ShellState):
    port = _free_port()
    srv = ThreadingHTTPServer(("127.0.0.1", port), make_handler(state))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{port}"


def test_server_serves_ui_and_config(tmp_path, monkeypatch):
    monkeypatch.setattr(config_mod, "config_dir", lambda: tmp_path)
    srv, base = _serve(ShellState())
    try:
        html = urllib.request.urlopen(base + "/", timeout=5).read().decode()
        assert "<title>glia</title>" in html
        cfg = json.loads(urllib.request.urlopen(base + "/api/config", timeout=5).read())
        assert cfg["mode"] == "demo" and cfg["has_key"] is False
    finally:
        srv.shutdown()


def test_server_chat_streams_events(tmp_path, monkeypatch):
    monkeypatch.setattr(config_mod, "config_dir", lambda: tmp_path)
    srv, base = _serve(ShellState())
    try:
        req = urllib.request.Request(
            base + "/api/chat",
            data=json.dumps({"message": "ping"}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        body = urllib.request.urlopen(req, timeout=5).read().decode()
        kinds = [json.loads(line[6:])["kind"] for line in body.splitlines() if line.startswith("data: ")]
        assert "run_started" in kinds
        assert "run_finished" in kinds
        assert "__done__" in kinds
        assert "ping" in body  # the echo reply streamed back
    finally:
        srv.shutdown()


def test_server_new_and_quit(tmp_path, monkeypatch):
    monkeypatch.setattr(config_mod, "config_dir", lambda: tmp_path)
    state = ShellState()
    srv, base = _serve(state)
    try:
        urllib.request.urlopen(
            urllib.request.Request(base + "/api/new", data=b"{}", method="POST"), timeout=5
        )
        urllib.request.urlopen(
            urllib.request.Request(base + "/api/quit", data=b"{}", method="POST"), timeout=5
        )
        assert state.quit_event.is_set()
    finally:
        srv.shutdown()


# --- OpenAI mode + interactive approval ---------------------------------------


def test_free_port_is_valid():
    from glia.shell.app import _free_port

    assert 1024 <= _free_port() <= 65535


def test_open_native_window_returns_false_without_webview(monkeypatch):
    import sys

    from glia.shell import app as app_mod

    monkeypatch.setitem(sys.modules, "webview", None)  # force `import webview` to fail
    assert app_mod._open_native_window("http://127.0.0.1:1") is False


def test_app_main_web_mode_runs_and_quits(tmp_path, monkeypatch):
    import time

    from glia.shell import app as app_mod

    monkeypatch.setattr(config_mod, "config_dir", lambda: tmp_path)
    monkeypatch.setattr(app_mod.webbrowser, "open", lambda url: None)  # don't launch a real browser

    from glia.shell.app import _free_port

    port = _free_port()
    rc: dict = {}
    thread = threading.Thread(target=lambda: rc.__setitem__("code", app_mod.main(["--web", "--port", str(port)])))
    thread.start()

    base = f"http://127.0.0.1:{port}"
    for _ in range(100):  # wait for the server to come up
        try:
            urllib.request.urlopen(base + "/api/config", timeout=2)
            break
        except Exception:  # noqa: BLE001
            time.sleep(0.05)

    urllib.request.urlopen(
        urllib.request.Request(base + "/api/quit", data=b"{}", method="POST"), timeout=5
    )
    thread.join(timeout=10)
    assert rc.get("code") == 0


def test_build_agent_openai_mode():
    agent = build_agent(Config(mode="openai", openai_api_key="sk", openai_model="gpt-4o-mini"))
    assert type(agent.llm).__name__ == "OpenAILLM"
    assert agent.llm.model == "gpt-4o-mini"


def test_config_persists_openai_and_approve(tmp_path, monkeypatch):
    monkeypatch.setattr(config_mod, "config_dir", lambda: tmp_path)
    Config(mode="openai", openai_api_key="sk", openai_model="gpt-4o", approve_tools=True).save()
    loaded = Config.load()
    assert loaded.openai_model == "gpt-4o" and loaded.approve_tools is True
    pub = loaded.public()
    assert "openai_api_key" not in pub and pub["has_openai_key"] is True


def test_approval_policy_resolves_via_other_thread():
    import asyncio

    from glia.approval import ApprovalRequest

    state = ShellState()
    policy = state.approval_policy()

    async def scenario():
        req = ApprovalRequest(step=1, tool_use_id="t1", name="danger", arguments={})
        task = asyncio.ensure_future(policy(req))
        await asyncio.sleep(0)  # let the policy register its future
        assert state.resolve_approval("t1", allow=False, reason="nope") is True
        return await task

    decision = run(scenario())
    assert decision.allow is False and decision.reason == "nope"


def test_resolve_unknown_approval_is_false():
    assert ShellState().resolve_approval("missing", allow=True) is False


def test_server_interactive_approval_deny_round_trip(tmp_path, monkeypatch):
    import time

    from glia import Agent
    from glia import tool as _tool
    from glia.providers import EchoLLM
    from glia.providers import call as tcall
    from glia.shell import server as server_mod

    monkeypatch.setattr(config_mod, "config_dir", lambda: tmp_path)

    @_tool
    async def danger(x: str) -> str:
        return "ran"

    def fake_build(config, *, stream=True, approval=None):
        return Agent(EchoLLM([tcall("danger", {"x": "boom"}), "done"]), tools=[danger],
                     approval=approval, stream=stream)

    monkeypatch.setattr(server_mod, "build_agent", fake_build)

    state = ShellState()
    state.config.approve_tools = True
    srv, base = _serve(state)
    try:
        result: dict = {}

        def do_chat():
            req = urllib.request.Request(
                base + "/api/chat", data=json.dumps({"message": "go"}).encode(),
                headers={"Content-Type": "application/json"}, method="POST",
            )
            result["body"] = urllib.request.urlopen(req, timeout=10).read().decode()

        chat = threading.Thread(target=do_chat)
        chat.start()

        # Wait for the run to park on the approval, then deny it.
        for _ in range(100):
            if state.approvals:
                break
            time.sleep(0.02)
        assert state.approvals, "no approval was requested"
        tid = next(iter(state.approvals))
        urllib.request.urlopen(urllib.request.Request(
            base + "/api/approve", data=json.dumps({"tool_use_id": tid, "allow": False}).encode(),
            headers={"Content-Type": "application/json"}, method="POST",
        ), timeout=5)

        chat.join(timeout=10)
        kinds = [json.loads(line[6:])["kind"] for line in result["body"].splitlines() if line.startswith("data: ")]
        assert "approval_requested" in kinds
        assert "approval_resolved" in kinds
        assert "__done__" in kinds
        # The denied tool returned an error result.
        assert "not approved" in result["body"].lower()
    finally:
        srv.shutdown()
