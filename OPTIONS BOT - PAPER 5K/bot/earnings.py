"""Earnings-date calendar with a daily cache — the IV-crush guard's memory.

Why this exists: option premiums inflate before earnings and collapse the
morning after ("IV crush") REGARDLESS of which way the stock moves. Buying
short-dated options into earnings is how directionally-correct trades still
lose 30%. The engine uses this calendar to (a) refuse new entries when
earnings are imminent and (b) exit held contracts the day before.

Data source: yfinance (free, no API key). Install once:  pip install yfinance
If yfinance is missing or Yahoo is down, the guard degrades gracefully to
"unknown" (no blocking) and says so once in the event feed.

Fetches run in a background thread (Yahoo takes ~1-3s per symbol) and are
cached to earnings_cache.json for 12 hours, so the trading loop never waits.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from datetime import date, datetime, timezone
from pathlib import Path

log = logging.getLogger("earnings")

CACHE_TTL_S = 12 * 3600


class EarningsCalendar:
    def __init__(self, symbols: list[str], cache_file: Path):
        self.symbols = list(symbols)
        self.cache_file = cache_file
        self.dates: dict[str, str] = {}          # symbol -> "YYYY-MM-DD"
        self.available = True                    # yfinance importable?
        self.ready = False                       # first fetch finished?
        self._load_cache()
        self._maybe_refresh()

    # ---------- public ----------
    def next_earnings(self, symbol: str) -> date | None:
        """Next known earnings date for symbol, or None if unknown/none soon."""
        self._maybe_refresh()
        s = self.dates.get(symbol)
        if not s:
            return None
        try:
            d = date.fromisoformat(s)
            return d if d >= date.today() else None
        except ValueError:
            return None

    def days_until(self, symbol: str) -> int | None:
        d = self.next_earnings(symbol)
        return (d - date.today()).days if d else None

    # ---------- internals ----------
    def _load_cache(self) -> None:
        try:
            raw = json.loads(self.cache_file.read_text())
            if time.time() - raw.get("fetched_at", 0) < CACHE_TTL_S:
                self.dates = raw.get("dates", {})
                self.ready = True
        except Exception:
            pass

    def _maybe_refresh(self) -> None:
        try:
            raw = json.loads(self.cache_file.read_text())
            fresh = time.time() - raw.get("fetched_at", 0) < CACHE_TTL_S
        except Exception:
            fresh = False
        if fresh or getattr(self, "_fetching", False):
            return
        self._fetching = True
        threading.Thread(target=self._fetch_all, daemon=True).start()

    def _fetch_all(self) -> None:
        try:
            try:
                import yfinance as yf
            except ImportError:
                self.available = False
                log.warning("yfinance not installed — earnings guard inactive. "
                            "Fix: pip install yfinance")
                return
            found: dict[str, str] = {}
            for sym in self.symbols:
                try:
                    t = yf.Ticker(sym)
                    df = t.get_earnings_dates(limit=8)
                    if df is not None and len(df):
                        future = [d for d in df.index.tolist()
                                  if d.date() >= date.today()]
                        if future:
                            found[sym] = min(future).date().isoformat()
                except Exception as e:
                    log.debug("earnings %s: %s", sym, e)
                time.sleep(0.4)                  # be polite to Yahoo
            self.dates.update(found)
            self.ready = True
            self.cache_file.write_text(json.dumps(
                {"fetched_at": time.time(),
                 "fetched_at_h": datetime.now(timezone.utc).isoformat(),
                 "dates": self.dates}, indent=1))
            log.info("earnings calendar refreshed: %d/%d symbols have dates",
                     len(found), len(self.symbols))
        finally:
            self._fetching = False
