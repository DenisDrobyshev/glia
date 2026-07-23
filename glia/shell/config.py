"""Local settings for the shell, stored as JSON in the user's config directory.

The API key lives only on the user's machine and is never sent back to the UI —
:meth:`Config.public` returns a ``has_key`` boolean instead of the value.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path

APP_DIR = "glia"


def config_dir() -> Path:
    if os.name == "nt":
        base = os.environ.get("APPDATA") or str(Path.home())
        return Path(base) / APP_DIR
    xdg = os.environ.get("XDG_CONFIG_HOME")
    return (Path(xdg) if xdg else Path.home() / ".config") / APP_DIR


def config_path() -> Path:
    return config_dir() / "config.json"


@dataclass
class Config:
    mode: str = "demo"  # "demo" (offline echo) | "claude" | "ollama" | "openai"
    anthropic_api_key: str = ""
    model: str = "claude-opus-4-8"
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5"
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o-mini"
    system: str = "You are glia, a helpful and concise assistant."
    use_tools: bool = True
    approve_tools: bool = False  # ask before running each tool

    @classmethod
    def load(cls) -> Config:
        path = config_path()
        if path.exists():
            try:
                data = json.loads(path.read_text("utf-8"))
                known = {f.name for f in fields(cls)}
                return cls(**{k: v for k, v in data.items() if k in known})
            except Exception:  # noqa: BLE001 - a corrupt config shouldn't crash the app
                pass
        return cls()

    def save(self) -> None:
        directory = config_dir()
        directory.mkdir(parents=True, exist_ok=True)
        config_path().write_text(json.dumps(asdict(self), indent=2), "utf-8")

    def public(self) -> dict:
        """A UI-safe view — raw keys are never exposed."""
        return {
            "mode": self.mode,
            "model": self.model,
            "ollama_host": self.ollama_host,
            "ollama_model": self.ollama_model,
            "openai_base_url": self.openai_base_url,
            "openai_model": self.openai_model,
            "system": self.system,
            "use_tools": self.use_tools,
            "approve_tools": self.approve_tools,
            "has_key": bool(self.anthropic_api_key),
            "has_openai_key": bool(self.openai_api_key),
        }
