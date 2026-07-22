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
