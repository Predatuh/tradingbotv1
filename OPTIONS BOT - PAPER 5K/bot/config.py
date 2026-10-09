"""Config loading: config.yaml for behavior, .env for secrets."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def _load_env(path: Path) -> None:
    """Tiny .env loader (no external dependency)."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


@dataclass
class Config:
    raw: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Config":
        _load_env(ROOT / ".env")
        cfg_path = Path(path) if path else ROOT / "config.yaml"
        with open(cfg_path) as f:
            raw = yaml.safe_load(f)
        return cls(raw=raw)

    # --- convenience accessors ---
    @property
    def api_key(self) -> str:
        return os.environ.get("ALPACA_API_KEY", "")

    @property
    def api_secret(self) -> str:
        return os.environ.get("ALPACA_API_SECRET", "")

    @property
    def paper(self) -> bool:
        return self.raw.get("mode", "paper") != "live"

    @property
    def stock_symbols(self) -> list[str]:
        return list(self.raw.get("symbols", {}).get("stocks", []))

    @property
    def crypto_symbols(self) -> list[str]:
        return list(self.raw.get("symbols", {}).get("crypto", []))

    @property
    def all_symbols(self) -> list[str]:
        return self.stock_symbols + self.crypto_symbols

    @property
    def timeframe_minutes(self) -> int:
        return int(self.raw.get("timeframe_minutes", 5))

    @property
    def loop_seconds(self) -> int:
        return int(self.raw.get("loop_seconds", 60))

    @property
    def lookback_bars(self) -> int:
        return int(self.raw.get("lookback_bars", 300))

    @property
    def strategy(self) -> dict:
        return self.raw.get("strategy", {})

    @property
    def risk(self) -> dict:
        return self.raw.get("risk", {})

    @property
    def dashboard(self) -> dict:
        return self.raw.get("dashboard", {"host": "127.0.0.1", "port": 8050})

    def validate_keys(self) -> None:
        if not self.api_key or not self.api_secret:
            raise SystemExit(
                "\n  Missing Alpaca API keys.\n"
                "  1) Sign up free at https://alpaca.markets\n"
                "  2) In the dashboard pick 'Paper' account -> generate API keys\n"
                "  3) Copy .env.example to .env and paste them in\n"
            )
