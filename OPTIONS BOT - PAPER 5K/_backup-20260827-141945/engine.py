"""Live trading engine.

Every `loop_seconds`:
  1. refresh account equity -> feed kill switches
  2. for each symbol: pull bars -> evaluate ensemble signal
  3. manage open positions (exits, trailing stops)
  4. open new positions where the signal clears the threshold
  5. write full state to state.json for the dashboard
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

from .broker import AlpacaBroker, is_crypto
from .config import Config
from .risk import RiskManager
from .strategy import make_strategy

log = logging.getLogger("engine")
from .paths import app_dir

STATE_FILE = app_dir() / "state.json"


class PDTGuard:
    """RETIRED. FINRA eliminated the Pattern Day Trader rule on 2026-06-04.

    There is no longer a 3-day-trades-in-5-days limit, no $25k minimum for
    active trading, and brokers no longer count round trips — it was replaced
    by a real-time intraday margin framework. This guard used to BLOCK entries
    and, worse, DEFER EXITS ("HOLD ... exit deferred"), which could keep a
    losing trade open. Since the rule it enforced no longer exists, it stays
    off unless someone deliberately sets risk.pdt_guard: true in config.yaml.

    Kept (inert) rather than deleted so old configs and saved pdt_*.json files
    keep loading without errors.
    """

    def __init__(self, enabled: bool = True, pdt_file=None):
        import datetime as _dt
        self._dt = _dt
        self.enabled = enabled
        self.cash_account = False      # set from the broker on the first cycle
        self.pdt_file = pdt_file or (app_dir() / "pdt_log.json")
        try:
            self.dates: list[str] = json.loads(self.pdt_file.read_text())
        except Exception:
            self.dates = []

    def used(self) -> int:
        cutoff = (datetime.now(timezone.utc) - self._dt.timedelta(days=7)).strftime("%Y-%m-%d")
        self.dates = [d for d in self.dates if d >= cutoff]
        return len(self.dates)

    def record(self) -> None:
        self.dates.append(datetime.now(timezone.utc).strftime("%Y-%m-%d"))
        try:
            self.pdt_file.write_text(json.dumps(self.dates))
        except Exception:
            pass

    def active(self, equity: float, paper: bool) -> bool:
        # The PDT rule was retired 2026-06-04 — nothing to guard against.
        # `enabled` is False unless config.yaml explicitly opts back in.
        if not self.enabled:
            return False
        if self.cash_account:
            return False
        return not paper and equity < 25_500
TRADES_FILE = app_dir() / "trades_log.jsonl"


def symbol_mode(cfg, symbol: str) -> str:
    """'buy' (default) | 'watch' (analyze, never enter) | 'off' (ignore)."""
    return (cfg.raw.get("symbol_modes") or {}).get(symbol, "buy")


class TradingEngine:
    def __init__(self, cfg: Config, broker: AlpacaBroker,
                 state_file=None, trades_file=None):
        self.state_file = state_file or STATE_FILE
        self.trades_file = trades_file or TRADES_FILE
        self.cfg = cfg
        self.broker = broker
        self.strategy = make_strategy(cfg.strategy)
        self.risk = RiskManager(cfg.risk,
                                anchor_file=self.state_file.parent / "day_anchor.json")
        # symbol -> {"side","entry","stop","tp","qty"}
        self.book: dict[str, dict] = {}
        self.last_signals: dict[str, dict] = {}
        self.events: list[dict] = []
        self._scout_thread = None
        self._scout_last_try = 0.0
        self._stop = False
        self.manual_close: set[str] = set()   # symbols the user asked to sell now
        self.manual_buy: set[str] = set()     # symbols the user asked to buy now
        self.charts: dict[str, list] = {}     # symbol -> recent closes for dashboard graphs
        self.chart_times: dict[str, list] = {}   # symbol -> unix ts per close
        self.bars_cache: dict[str, dict] = {}    # symbol -> OHLCV for the big chart
        # phone pushes (ntfy.sh) — set `ntfy_topic: your-topic` in config.yaml
        self.ntfy_topic = str(cfg.raw.get("ntfy_topic", "") or "").strip()
        self.pdt = PDTGuard(enabled=cfg.risk.get("pdt_guard", False),
                            pdt_file=(state_file.parent / f"pdt_{state_file.stem}.json")
                            if state_file is not None else None)
        self._equity = 0.0
        self._acct_said = False
        self._cash_said = False

    def request_stop(self) -> None:
        self._stop = True

    def _book_key(self, symbol: str) -> str | None:
        """Map whatever the dashboard sent to our book key.

        Alpaca reports crypto WITHOUT the slash (BTCUSD) while the book keys it
        WITH one (BTC/USD) — so a plain lookup silently misses every crypto
        position. Match on the slashless, upper-cased form.
        """
        s = (symbol or "").strip().upper()
        if not s:
            return None
        if symbol in self.book:
            return symbol
        flat = s.replace("/", "")
        return next((k for k in self.book if k.upper().replace("/", "") == flat), None)

    def request_manual_close(self, symbol: str) -> bool:
        """User clicked Sell on the dashboard: close this position next tick."""
        key = self._book_key(symbol)
        if key is None:
            return False
        self.manual_close.add(key)
        return True

    def request_manual_buy(self, symbol: str, direction: str = "long"):
        """User clicked Buy on the dashboard: buy this symbol next tick."""
        sym = (symbol or "").strip().upper()
        flat = sym.replace("/", "")
        match = next((s for s in self.cfg.all_symbols
                      if s.upper() == sym or s.upper().replace("/", "") == flat), None)
        if not match:
            return False, f"{symbol} isn't on this bot's watch list."
        if self._book_key(match) is not None:
            return False, f"Already holding {match} — use Sell now to exit."
        self.manual_buy.add(match)
        return True, ""

    # ---------- helpers ----------
    def _event(self, msg: str) -> None:
        stamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
        self.events.append({"t": stamp, "msg": msg})
        self.events = self.events[-80:]
        log.info(msg)
        try:                                   # permanent record — survives restarts
            with open(self.state_file.parent / "events_log.txt", "a",
                      encoding="utf-8") as _f:
                _f.write(datetime.now(timezone.utc).isoformat(timespec="seconds")
                         + "  " + msg + "\n")
        except Exception:
            pass
        # phone pushes for real-money events worth knowing about
        if self.ntfy_topic:
            from .notify import push
            if msg.startswith("ENTER ") or msg.startswith("EXIT "):
                push(self.ntfy_topic, "Trade " + ("opened" if msg.startswith("ENTER") else "closed"),
                     msg, priority="high", tags="chart_with_upwards_trend")
            elif "KILL SWITCH" in msg or "DAILY LOSS" in msg or "kill switch" in msg:
                push(self.ntfy_topic, "Bot safety stop triggered", msg,
                     priority="urgent", tags="octagonal_sign")

    def _log_trade(self, record: dict) -> None:
        record["ts"] = datetime.now(timezone.utc).isoformat()
        with open(self.trades_file, "a") as f:
            f.write(json.dumps(record) + "\n")

    def _sync_book(self) -> list[dict]:
        """Reconcile our book with what Alpaca actually holds."""
        positions = self.broker.positions()
        held = set()
        for p in positions:
            sym = p["symbol"]
            # crypto positions come back without the slash
            for our_sym in self.cfg.all_symbols:
                if our_sym.replace("/", "") == sym:
                    sym = our_sym
                    break
            held.add(sym)
            if sym not in self.book:
                self.book[sym] = {"side": p["side"], "entry": p["avg_entry"],
                                  "stop": None, "tp": None, "qty": p["qty"]}
            else:
                self.book[sym].pop("pending_since", None)  # order has filled
        for sym in list(self.book):
            if sym not in held:
                # keep entries whose order was just submitted and hasn't filled yet
                pending = self.book[sym].get("pending_since")
                if pending is not None and time.time() - pending < 300:
                    continue
                if pending is not None:      # limit order never filled — clean it up
                    self.broker.cancel_open_orders(sym)
                    self._event(f"CANCELLED unfilled entry order for {sym}")
                del self.book[sym]
        return positions

    # ---------- smart-money scout ----------
    def _scout_boosts(self) -> dict[str, float]:
        """Load fresh scout boosts; kick off a daily background rescan."""
        if not self.cfg.raw.get("scout", {}).get("enabled", True):
            return {}
        try:
            import scout as scout_mod
        except ImportError:
            return {}
        boosts = scout_mod.load_boosts(max_age_hours=24)   # rescan roughly daily
        stale = not boosts
        alive = self._scout_thread is not None and self._scout_thread.is_alive()
        if stale and not alive and time.time() - self._scout_last_try > 6 * 3600:
            import threading
            self._scout_last_try = time.time()

            def _run():
                try:
                    scout_mod.run_scout(self.cfg, log=lambda m: log.info("scout: %s", m))
                    self._event("Scout scan finished — smart-money boosts refreshed")
                except Exception as e:
                    log.warning("scout scan failed: %s", e)
            self._scout_thread = threading.Thread(target=_run, daemon=True)
            self._scout_thread.start()
            self._event("Scout scan started (SEC insider filings + news)…")
        return boosts

    # ---------- core cycle ----------
    def cycle(self) -> None:
        account = self.broker.account()
        self._equity = account["equity"]
        if account.get("is_cash_account") is not None and not self._acct_said:
            self._acct_said = True
            self.pdt.cash_account = bool(account["is_cash_account"])
            mult = account.get("multiplier")
            kind = ("LIMITED MARGIN (1x buying power — under $2k, no shorting)"
                    if self.pdt.cash_account else f"MARGIN ({mult}x buying power)")
            self._event(f"Account: {kind}. Day-trade limits do not apply — FINRA "
                        f"retired the PDT rule on 2026-06-04, so there is no "
                        f"3-trades-per-week cap and no $25k minimum.")
        from zoneinfo import ZoneInfo
        today = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
        self.risk.update_equity(account["equity"], today)

        if self.risk.hard_killed and self.book:
            self._event("MAX DRAWDOWN KILL SWITCH — flattening this bot's STOCK positions "
                        "(options are the options bot's job — never touched)")
            for p in self.broker.positions():
                sym = p.get("symbol", "")
                if len(sym) > 12:          # OCC option symbol -> NOT ours to close
                    continue
                self.broker.cancel_open_orders(sym)
                self.broker.close_position(sym)
                self._log_trade({"action": "exit", "symbol": sym,
                                 "reason": "kill switch", "price": p.get("current_price"),
                                 "pnl_pct": None})
            self.book.clear()

        positions = self._sync_book()
        clock = self.broker.clock()
        boosts = self._scout_boosts()

        for symbol in self.cfg.all_symbols:
            mode = symbol_mode(self.cfg, symbol)
            if mode == "off":
                self.last_signals.pop(symbol, None)
                continue
            stock_closed = not is_crypto(symbol) and not clock["is_open"]
            df = self.broker.bars(symbol, self.cfg.timeframe_minutes, self.cfg.lookback_bars)
            if df.empty or len(df) < 60:
                continue
            sig = self.strategy.evaluate(df)
            if symbol in boosts:                    # smart-money scout nudge (capped ±0.10)
                sig.composite = float(max(-1.0, min(1.0, sig.composite + boosts[symbol])))
                sig.parts["scout"] = boosts[symbol]
            d = sig.to_dict()
            d["mode"] = mode
            self.last_signals[symbol] = d
            try:                               # live mini-graph for the dashboard
                tail = df["close"].tail(120)
                self.charts[symbol] = [round(float(c), 4) for c in tail]
                self.chart_times[symbol] = [int(t.timestamp()) for t in tail.index]
                # full OHLCV for the big candlestick chart — served on request by
                # the dashboard's /chart endpoint, kept OUT of state.json so the
                # 3-second refresh stays small
                b = df.tail(400)
                self.bars_cache[symbol] = {
                    "t": [int(x.timestamp()) for x in b.index],
                    "o": [round(float(v), 4) for v in b["open"]],
                    "h": [round(float(v), 4) for v in b["high"]],
                    "l": [round(float(v), 4) for v in b["low"]],
                    "c": [round(float(v), 4) for v in b["close"]],
                    "v": [int(v) for v in b["volume"]],
                    "tf": self.cfg.timeframe_minutes,
                }
            except Exception:
                pass

            pos = self.book.get(symbol)
            forced = symbol in self.manual_buy
            if pos:
                self.manual_buy.discard(symbol)
                self._manage_position(symbol, pos, sig, stock_closed)
            elif forced:
                self.manual_buy.discard(symbol)
                if stock_closed:
                    self._event(f"CAN'T BUY {symbol}: market is closed")
                else:
                    self._maybe_enter(symbol, sig, account["equity"],
                                      forced=True, cash=account.get("cash"))
            elif not stock_closed and mode == "buy":
                self._maybe_enter(symbol, sig, account["equity"],
                                  cash=account.get("cash"))

        self._write_state(account, positions, clock)

    def _manage_position(self, symbol: str, pos: dict, sig, market_closed: bool) -> None:
        price = sig.price
        side = pos["side"]

        # initialize stops for positions we adopted (e.g. after restart)
        if pos["stop"] is None:
            plan = self.risk.plan_position(side, pos["entry"], sig.atr, 1e9, True)
            if plan:
                pos["stop"], pos["tp"] = plan.stop_price, plan.take_profit

        pos["stop"] = self.risk.trail_stop(side, pos["stop"], price, sig.atr)

        hit_stop = price <= pos["stop"] if side == "long" else price >= pos["stop"]
        hit_tp = pos["tp"] is not None and (
            price >= pos["tp"] if side == "long" else price <= pos["tp"])
        # a position YOU opened with Buy now isn't dumped by the same signal you
        # overrode — stops, targets and trailing still protect it
        signal_exit = self.strategy.wants_exit(sig, side) and not pos.get("manual_entry")
        manual = symbol in self.manual_close

        if market_closed:
            if manual and not is_crypto(symbol):
                self.manual_close.discard(symbol)
                self._event(f"CAN'T SELL {symbol}: market is closed — try during market hours")
            return  # can't exit a stock position while the market is closed

        if manual or hit_stop or hit_tp or signal_exit:
            self.manual_close.discard(symbol)
            reason = ("manual close (you clicked Sell)" if manual else
                      "stop-loss" if hit_stop else
                      ("take-profit" if hit_tp else "signal fade"))
            # user's explicit order is treated like a stop: never PDT-deferred
            hit_stop = hit_stop or manual
            today_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            if (not is_crypto(symbol) and pos.get("opened_date") == today_utc
                    and self.pdt.active(self._equity, self.cfg.paper)):
                if not hit_stop and self.pdt.used() >= 3:
                    if not pos.get("_pdt_deferred"):
                        pos["_pdt_deferred"] = True
                        self._event(f"HOLD {symbol}: exit deferred — 4th day trade "
                                    f"would trigger the PDT flag")
                    return
                self.pdt.record()
                self._event(f"PDT: day trade used ({self.pdt.used()}/3 this week)")
            pnl_pct = ((price - pos["entry"]) / pos["entry"]) * (1 if side == "long" else -1)
            self.broker.cancel_open_orders(symbol)
            self.broker.close_position(symbol)
            self._event(f"EXIT {symbol} ({reason}) ~{pnl_pct:+.2%}")
            self._log_trade({"action": "exit", "symbol": symbol, "reason": reason,
                             "price": price, "pnl_pct": round(pnl_pct, 5)})
            if pnl_pct < 0:
                self.risk.register_loss(symbol)
            self.book.pop(symbol, None)

    def _maybe_enter(self, symbol: str, sig, equity: float, forced: bool = False,
                     cash: float | None = None) -> None:
        side = self.strategy.wants_entry(sig)
        if forced and not side:
            side = "long"                      # manual Buy now = buy, signal or not
        if not side:
            return
        if forced:
            self._event(f"MANUAL BUY: you clicked Buy on {symbol} — entering long, "
                        f"sized by your normal spending cap")
        if side == "short" and is_crypto(symbol):
            return  # Alpaca spot crypto can't short
        ok, why = self.risk.can_open(symbol, len(self.book), list(self.book))
        if not ok:
            self._event(f"SKIP {symbol}: {why}")
            return
        if (not is_crypto(symbol) and self.pdt.active(self._equity, self.cfg.paper)
                and self.pdt.used() >= 3):
            self._event(f"SKIP {symbol}: PDT guard — 3 day trades used this week")
            return
        plan = self.risk.plan_position(side, sig.price, sig.atr, equity,
                                       fractionable=True, cash=cash)
        if not plan or plan.notional < 10:
            if (self.risk.cash_only and cash is not None and cash < 10
                    and not self._cash_said):
                self._cash_said = True
                self._event(f"SKIP {symbol}: only ${cash:,.0f} cash left — "
                            f"cash-only mode, this bot never buys on margin")
            return
        self._cash_said = False
        order_id = self.broker.submit_entry(symbol, side, plan.qty, ref_price=sig.price)
        if order_id:
            self.book[symbol] = {"side": side, "entry": sig.price,
                                 "stop": plan.stop_price, "tp": plan.take_profit,
                                 "qty": plan.qty, "pending_since": time.time(),
                                 "manual_entry": bool(forced),
                                 "opened_date": datetime.now(timezone.utc).strftime("%Y-%m-%d")}
            self._event(
                f"ENTER {side.upper()} {symbol} qty={plan.qty:.6f} @~{sig.price:.4f} "
                f"score={sig.composite:+.2f} stop={plan.stop_price} tp={plan.take_profit}")
            self._log_trade({"action": "enter", "symbol": symbol, "side": side,
                             "qty": plan.qty, "price": sig.price,
                             "und_price": round(float(sig.price), 4),
                             "score": round(sig.composite, 4),
                             "stop": plan.stop_price, "tp": plan.take_profit,
                             "signal_parts": sig.to_dict()["parts"]})

    # ---------- state for dashboard ----------
    def _write_state(self, account: dict, positions: list, clock: dict) -> None:
        state = {
            "updated": datetime.now(timezone.utc).isoformat(),
            "mode": "PAPER" if self.cfg.paper else "LIVE",
            "account": account,
            "market": clock,
            "positions": positions,
            "book": self.book,
            "signals": self.last_signals,
            "charts": self.charts,
            "chart_times": self.chart_times,
            "risk": self.risk.status(),
            "events": self.events[::-1],
            "config": {
                "entry_threshold": self.strategy.entry_threshold,
                "controls": True,
                "bot_name": (self.cfg.raw.get("dashboard") or {}).get("name"),       # dashboard shows Buy/Sell buttons
                "symbols": self.cfg.all_symbols,
                "timeframe_minutes": self.cfg.timeframe_minutes,
                "risk_profile": self.cfg.raw.get("_risk_profile_name"),
            },
        }
        self.state_file.write_text(json.dumps(state, indent=1, default=str))

    # ---------- main loop ----------
    def run_forever(self) -> None:
        brain = "ML model" if type(self.strategy).__name__ == "MLStrategy" else "signal ensemble"
        prof = self.cfg.raw.get("_risk_profile_name")
        if prof:
            self._event(f"Risk profile: {prof}")
        self._event(f"Engine started — {'PAPER' if self.cfg.paper else 'LIVE'} mode, "
                    f"brain: {brain}, {len(self.cfg.all_symbols)} symbols, "
                    f"{self.cfg.timeframe_minutes}m bars, every {self.cfg.loop_seconds}s")
        while not self._stop:
            started = time.time()
            try:
                self.cycle()
            except KeyboardInterrupt:
                raise
            except Exception as e:
                log.exception("cycle error: %s", e)
                self._event(f"cycle error: {e}")
            elapsed = time.time() - started
            remaining = max(1.0, self.cfg.loop_seconds - elapsed)
            while remaining > 0 and not self._stop:
                if self.manual_close or self.manual_buy:
                    break        # user clicked Buy/Sell — run the next cycle right away
                time.sleep(min(1.0, remaining))
                remaining -= 1.0
        self._event("Engine stopped.")
