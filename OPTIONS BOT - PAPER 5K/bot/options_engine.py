"""Options trading engine: calls and puts driven by the same ML/ensemble brain.

The logic per cycle:
  1. Evaluate the signal on each UNDERLYING (same model as the stock bot).
  2. Very bullish  -> buy a near-the-money CALL, 7-21 days out.
     Very bearish  -> buy a near-the-money PUT (options can profit on the
     way down without shorting).
  3. Manage each open contract on the PREMIUM, not the stock price:
       +profit_target_pct  -> take profit
       -stop_pct           -> cut it
       <= close_dte days   -> close regardless (never bleed out on theta)
       signal flips        -> close early
  4. Kill switches on account equity, same as always.

Options-specific honesty: a wide bid/ask spread can eat 10%+ per round trip,
so the broker layer refuses contracts with spreads over max_spread_pct, and
entries are midpoint limit orders — never market, never chasing the ask.
"""
from __future__ import annotations

import json
import logging
import time
from datetime import date, datetime, timezone
from pathlib import Path

from .config import Config
from .engine import PDTGuard, symbol_mode
from .options_broker import OptionsBroker
from .risk import RiskManager
from .strategy import make_strategy

log = logging.getLogger("opt_engine")
from .paths import app_dir

ROOT = app_dir()
STATE_FILE = ROOT / "state.json"
TRADES_FILE = ROOT / "trades_log.jsonl"
BOOK_FILE = ROOT / "options_book.json"        # survives restarts


class OptionsEngine:
    def __init__(self, cfg: Config, broker: OptionsBroker,
                 state_file=None, trades_file=None, book_file=None):
        self.state_file = state_file or STATE_FILE
        self.trades_file = trades_file or TRADES_FILE
        self.book_file = book_file or BOOK_FILE
        self.cfg = cfg
        self.broker = broker
        self.strategy = make_strategy(cfg.strategy)
        self.risk = RiskManager(cfg.risk,
                                anchor_file=self.state_file.parent / "day_anchor.json")
        o = cfg.raw.get("options", {})
        self.underlyings: list[str] = o.get("underlyings", ["SPY", "QQQ", "NVDA"])
        self.call_score = float(o.get("call_score", 0.20))    # composite >= -> call
        self.put_score = float(o.get("put_score", -0.20))     # composite <= -> put
        # dashboard slider override (strictness.json) beats config.yaml when present
        try:
            ov = json.loads((self.state_file.parent / "strictness.json").read_text())
            v = min(0.30, max(0.10, float(ov["value"])))
            self.call_score, self.put_score = v, -v
        except Exception:
            pass
        self.dte_min = int(o.get("dte_min", 7))
        self.dte_max = int(o.get("dte_max", 21))
        self.close_dte = int(o.get("close_dte", 2))
        self.max_premium_pct = float(o.get("max_premium_pct", 0.06))
        self.profit_target_pct = float(o.get("profit_target_pct", 0.60))
        self.stop_pct = float(o.get("stop_pct", 0.40))
        self.max_spread_pct = float(o.get("max_spread_pct", 0.12))
        # trailing ratchet v2:
        #  * tiered profit floors — the win you KEEP rises in steps with the peak
        #  * adaptive trail — width scales with how jumpy THIS contract actually is
        self.trail_arm_pct = float(o.get("trail_arm_pct", 0.15))  # protect gains sooner
        # peak gain reached -> minimum gain locked in (sell if it touches the floor)
        # "bank the wins" tiers: give back far less of each peak than v2 did
        self.profit_floors = o.get("profit_floors",
                                   [[0.20, 0.12], [0.30, 0.22], [0.40, 0.32],
                                    [0.55, 0.45], [0.75, 0.62]])
        # execution: "auto" places real orders; "alert" only ANNOUNCES trades
        # (for manual execution in another broker while funds settle, etc.)
        self.alert_only = str(o.get("execution", "auto")).lower() == "alert"
        self.advice_capital = float(o.get("advice_capital", 500))
        self.trail_vol_mult = float(o.get("trail_vol_mult", 3.0))   # trail = 3x typical wiggle
        self.trail_min_pct = float(o.get("trail_min_pct", 0.07))    # never tighter than 7% of peak
        self.trail_max_pct = float(o.get("trail_max_pct", 0.15))    # never give back >15% of peak
        # book: contract_symbol -> {underlying, type, entry_premium, expiry, qty, pending_since}
        self.book: dict[str, dict] = self._load_book()
        self.last_signals: dict[str, dict] = {}
        self.events: list[dict] = []
        self.pdt = PDTGuard(enabled=cfg.risk.get("pdt_guard", False),
                            pdt_file=self.book_file.parent / f"pdt_{self.book_file.stem}.json")
        self._equity = 0.0
        self._acct_said = False
        self._stop = False
        self.manual_close: set[str] = set()   # contracts the user asked to sell now
        self.manual_buy: dict[str, str] = {}  # underlying -> "call"/"put" user asked to buy now
        self.charts: dict[str, list] = {}     # underlying -> recent closes for dashboard graphs
        self.block: dict[str, str] = {}       # underlying -> why it did NOT buy
        self.thoughts: dict[str, str] = {}    # underlying -> what it's thinking, in words
        self._settle_said = ""                # once-a-day settlement notice
        self.would: dict[str, str] = {}       # what it WOULD have traded (observe mode)
        self._would_said: dict[str, str] = {}  # don't repeat the same WOULD every cycle
        # OBSERVE ONLY: watch, chart and reason — never place a single order.
        # Persisted so a restart can't silently re-arm real trading.
        self.observe_only = bool(o.get("observe_only", False))
        try:
            _ov = json.loads((self.state_file.parent / "observe.json").read_text())
            self.observe_only = bool(_ov["value"])
        except Exception:
            pass
        try:                       # startup: gate the broker too
            self.broker.observe_only = self.observe_only
        except Exception:
            pass
        self.notices: list[dict] = self._load_notices()   # buy/sell stories for the phone
        # WATCHLIST PRICES: what one at-the-money contract costs on each
        # underlying, even when we own nothing. Refreshed a couple of symbols
        # per cycle (round-robin) so 20 symbols cost ~4 API calls a cycle
        # instead of hundreds. Display only — never used to pick trades.
        self.candidates: dict[str, dict] = {}
        self._cand_rr = 0
        # POST-EXIT TRACKER (instrumentation only, never affects trading):
        # after any sale, keep quoting the contract for a few hours and log the
        # premium path to post_exit_log.jsonl — so "did that stop/flip sell the
        # bottom?" gets answered with real numbers instead of guesses.
        self.post_watch: dict[str, dict] = {}
        self.chart_times: dict[str, list] = {}   # underlying -> unix ts per close
        self.bars_cache: dict[str, dict] = {}    # symbol -> OHLCV for the big chart
        # phone pushes (ntfy.sh) — set `ntfy_topic: your-topic` in config.yaml
        self.ntfy_topic = str(cfg.raw.get("ntfy_topic", "") or "").strip()
        # ---- DAY LOCK: bank the morning, refuse to give it back ----
        dl = o.get("day_lock") or {}
        self.dl_enabled = bool(dl.get("enabled", False))
        self.dl_arm_pct = float(dl.get("arm_pct", 0.05))        # day up 5% arms the lock
        self.dl_giveback = float(dl.get("giveback_pct", 0.40))  # give back 40% of day peak -> lock
        self.dl_target = float(dl.get("target_pct", 0.15))      # or bank & stop at +15% day
        self.entry_cutoff_utc = int(o.get("entry_cutoff_utc", 0))  # 0=off; 16 = no buys after noon ET
        # ---- FAST-BLEED CUT: kill wrong-way trades small, early ----
        fb = o.get("fast_bleed") or {}
        self.fb_enabled = bool(fb.get("enabled", True))
        self.fb_within_min = float(fb.get("within_min", 25))   # only in the first N minutes
        self.fb_drop_pct = float(fb.get("drop_pct", 0.18))     # down this much fast -> cut now
        self._dl_date = None
        self._dl_peak = 0.0
        self._dl_locked = False
        self._cutoff_said = None
        # structural guards: earnings/IV-crush avoidance + overpriced-premium gate
        self.earnings_guard = bool(o.get("earnings_guard", True))
        self.earnings_skip_days = int(o.get("earnings_skip_days", 4))
        self.iv_guard = bool(o.get("iv_guard", True))
        self.iv_max_ratio = float(o.get("iv_max_ratio", 1.6))
        self.earnings_cal = None
        if self.earnings_guard:
            try:
                from .earnings import EarningsCalendar
                self.earnings_cal = EarningsCalendar(
                    self.underlyings,
                    self.state_file.parent / "earnings_cache.json")
            except Exception as e:
                log.warning("earnings guard unavailable: %s", e)

    # ---------- dashboard controls ----------
    def request_stop(self) -> None:
        self._stop = True

    def request_manual_close(self, symbol: str) -> bool:
        """User clicked Sell on the dashboard: close this contract next tick."""
        if self.observe_only:
            return False
        if symbol not in self.book:
            return False
        self.manual_close.add(symbol)
        return True

    def request_manual_buy(self, symbol: str, direction: str = "call"):
        """User clicked Buy CALL / Buy PUT on the dashboard: enter next tick."""
        sym = (symbol or "").strip().upper()
        d = (direction or "call").strip().lower()
        if self.observe_only:
            return False, "OBSERVE ONLY is on — this bot places no orders."
        if sym not in self.underlyings:
            return False, f"{symbol} isn't on this bot's watch list."
        if d not in ("call", "put"):
            return False, "Direction must be call or put."
        if self._held_for(sym):
            return False, f"Already holding a {sym} contract (one per underlying)."
        self.manual_buy[sym] = d
        return True, ""

    def apply_observe(self, on: bool) -> bool:
        """OBSERVE ONLY on/off. On = the bot watches and reasons but places
        NO orders of any kind, and the Buy/Sell buttons disappear."""
        self.observe_only = bool(on)
        try:                       # push the gate down to the broker
            self.broker.observe_only = self.observe_only
        except Exception:
            pass
        try:
            (self.state_file.parent / "observe.json").write_text(
                json.dumps({"value": self.observe_only}))
        except Exception:
            pass
        if self.observe_only:
            self.manual_buy.clear()
            self.manual_close.clear()
            n_open = len([b for b in self.book.values() if not b.get("advice")])
            extra = (f" WARNING: {n_open} position(s) still open — while observe "
                     f"mode is on they are NOT managed (no stop, no target). "
                     f"Turn it off or sell them yourself." if n_open else "")
            self._event("OBSERVE ONLY is ON — watching and charting only, no "
                        "orders will be placed." + extra)
        else:
            self._event("OBSERVE ONLY is OFF — the bot can place orders again.")
        return self.observe_only

    def apply_strictness(self, value: float) -> float:
        """User moved the dashboard slider: new entry bar, saved across restarts."""
        v = min(0.30, max(0.10, float(value)))
        self.call_score, self.put_score = v, -v
        try:
            (self.state_file.parent / "strictness.json").write_text(
                json.dumps({"value": v}))
        except Exception:
            pass
        self._event(f"Signal strictness set to ±{v:.2f} from the dashboard "
                    f"(saved — survives restarts, overrides config.yaml)")
        return v

    # ---------- persistence ----------
    def _load_book(self) -> dict:
        try:
            return json.loads(self.book_file.read_text())
        except Exception:
            return {}

    def _save_book(self) -> None:
        self.book_file.write_text(json.dumps(self.book, indent=1))

    def _event(self, msg: str) -> None:
        self.events.append({"t": datetime.now(timezone.utc).strftime("%H:%M:%S"),
                            "msg": msg})
        self.events = self.events[-80:]
        log.info(msg)
        try:                                   # permanent record — survives restarts
            with open(self.state_file.parent / "events_log.txt", "a",
                      encoding="utf-8") as _f:
                _f.write(datetime.now(timezone.utc).isoformat(timespec="seconds")
                         + "  " + msg + "\n")
        except Exception:
            pass
        # phone pushes for anything that needs a human RIGHT NOW
        if self.ntfy_topic:
            from .notify import push
            if "📢" in msg:
                push(self.ntfy_topic, "🚨 ACT NOW — manual trade signal", msg,
                     priority="urgent", tags="rotating_light")
            elif "ORDER REJECTED" in msg:
                push(self.ntfy_topic, "Bot order failed — check it", msg,
                     priority="urgent", tags="warning")
            elif "KILL SWITCH" in msg or "DAILY LOSS" in msg:
                push(self.ntfy_topic, "Bot safety stop triggered", msg,
                     priority="high", tags="octagonal_sign")

    def _load_notices(self) -> list:
        try:
            return json.loads((self.state_file.parent / "notices.json").read_text())[-40:]
        except Exception:
            return []

    def _notice(self, kind: str, symbol: str, title: str, detail: str) -> None:
        """A buy/sell story for the phone: what happened and the REAL why."""
        self.notices.append({"ts": int(time.time()), "kind": kind,
                             "symbol": symbol, "title": title, "detail": detail})
        self.notices = self.notices[-40:]
        try:
            (self.state_file.parent / "notices.json").write_text(
                json.dumps(self.notices, indent=1))
        except Exception:
            pass

    def _think(self, und, sig, is_open: bool) -> str:
        """One honest sentence: what the bot sees on this symbol RIGHT NOW."""
        c = float(sig.composite); need = self.call_score
        trend = "UP" if sig.regime > 0 else ("DOWN" if sig.regime < 0 else "flat")
        held = self._held_for(und)
        if held:
            b = self.book.get(held[0], {})
            if not self.alert_only and b.get("advice"):
                return ("broker rejected the auto-buy — sent you a BUY-IT-YOURSELF "
                        "signal; it clears on its own if you don't take it")
            return (f"holding the {(b.get('type') or '').upper()} — managing it, "
                    f"not adding; score {c:+.2f}, trend {trend}")
        if not is_open:
            return f"market closed — watching only; last score {c:+.2f}, trend {trend}"
        if getattr(self, "_dl_locked", False):
            return "day's profit is locked in — done buying until tomorrow"
        if self.entry_cutoff_utc and datetime.now(timezone.utc).hour >= self.entry_cutoff_utc:
            return ("past the afternoon entry cutoff — late-day buys have been "
                    "the losers, so it only manages what it holds now")
        bl = self.block.get(und)
        if bl:
            return "won't buy: " + bl
        if c >= need and sig.regime > 0:
            return (f"score {c:+.2f} clears the +{need:.2f} bar and the trend is UP "
                    f"— lining up a CALL")
        if c <= -need and sig.regime < 0:
            return (f"score {c:+.2f} clears the -{need:.2f} bar and the trend is DOWN "
                    f"— lining up a PUT")
        if c >= need and sig.regime <= 0:
            return (f"bullish score {c:+.2f} but the trend is {trend} — waiting for "
                    f"the trend to turn UP before it will buy a call")
        if c <= -need and sig.regime >= 0:
            return (f"bearish score {c:+.2f} but the trend is {trend} — waiting for "
                    f"a real downtrend before it will buy a put")
        gap = need - abs(c)
        lean = ("leaning bullish" if c > 0.03 else
                "leaning bearish" if c < -0.03 else "no lean either way")
        return (f"score {c:+.2f} ({lean}), trend {trend} — needs {gap:.2f} more "
                f"conviction to hit the ±{need:.2f} entry bar, so it waits")

    def _exit_story(self, reason: str, pnl_pct: float, entry: float,
                    current: float, peak_gain: float, held_min: float) -> str:
        p = f"{pnl_pct:+.0%}"
        d = f"in at ${entry:.2f}, out at ${current:.2f}"
        if reason.startswith("profit target"):
            return (f"Hit the +{self.profit_target_pct:.0%} profit target — sold and "
                    f"took the win at {p} ({d}).")
        if reason.startswith("trailing"):
            return (f"Rode a solid rise to +{peak_gain:.0%}, then the premium started "
                    f"rolling over — the trailing ratchet sold to keep the gain "
                    f"instead of letting it bleed back: {p} banked ({d}).")
        if reason.startswith("fast-bleed"):
            return (f"Went the wrong way almost immediately ({p} within "
                    f"{held_min:.0f} minutes of entry) — cut it small on purpose "
                    f"rather than riding a wrong-way trade deeper ({d}).")
        if reason.startswith("premium stop"):
            return (f"Premium fell {self.stop_pct:.0%} from entry — stop-loss sold at "
                    f"{p} to protect the account from a bigger hit ({d}).")
        if reason.startswith("signal flipped"):
            return (f"Saw the signal flip against the position — momentum reversed, "
                    f"so it sold at {p} instead of hoping ({d}).")
        if reason.startswith("time exit"):
            return (f"Too close to expiry — from here time decay eats the premium "
                    f"every hour even if the stock goes nowhere, closed at {p} ({d}).")
        if reason.startswith("pre-earnings"):
            return (f"Earnings are about to hit — IV crush deflates option premiums "
                    f"whichever way the stock moves, so it banked {p} first ({d}).")
        if reason.startswith("day lock"):
            return (f"Day-lock fired — the day's profit was banked and trading "
                    f"stopped to protect it; this position closed at {p} ({d}).")
        if reason.startswith("manual"):
            return f"You tapped Sell — closed at market for {p} ({d})."
        return f"{reason} — closed at {p} ({d})."

    def _log_trade(self, record: dict) -> None:
        record["ts"] = datetime.now(timezone.utc).isoformat()
        with open(self.trades_file, "a") as f:
            f.write(json.dumps(record) + "\n")

    # ---------- core ----------
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
        clock = self.broker.clock()

        positions = self.broker.option_positions()
        held = {p["symbol"]: p for p in positions}
        # reconcile: drop book entries whose position is gone (respect fill grace)
        for sym in list(self.book):
            if self.book[sym].get("advice"):
                # in EXECUTION mode an advice entry is a "buy it yourself"
                # suggestion left after a broker reject — it must not haunt
                # the book forever, or it blocks new buys on that underlying
                if (not self.alert_only and
                        time.time() - self.book[sym].get("opened_at", 0) > 1800):
                    self._event(f"cleared the 30-min-old manual-buy suggestion "
                                f"{sym} — you didn't take it, so "
                                f"{self.book[sym].get('underlying')} is buyable again")
                    self.book.pop(sym, None)
                continue                       # advisor-mode entries live outside the broker
            if sym not in held:
                pending = self.book[sym].get("pending_since")
                if pending and time.time() - pending < 300:
                    continue
                if pending:
                    self.broker.cancel_open_orders(sym)
                    self._event(f"CANCELLED unfilled option order {sym}")
                    b2 = self.book.get(sym) or {}
                    self._notice("sell", sym,
                                 f"ORDER CANCELLED — {b2.get('underlying')} never filled",
                                 f"The buy order sat ~{int((time.time()-pending)/60)} "
                                 f"minutes without filling (the price moved away), so "
                                 f"it was cancelled. Nothing was bought, no money spent.")
                self.book.pop(sym, None)
            else:
                if self.book[sym].pop("pending_since", None):
                    b2 = self.book[sym]
                    self._notice("buy", sym,
                                 f"FILLED {b2.get('underlying')} "
                                 f"{(b2.get('type') or '').upper()} ${b2.get('strike'):g}",
                                 f"The order filled — you now hold {b2.get('qty')} "
                                 f"contract(s) in at ~${b2.get('entry_premium'):.2f}/share. "
                                 f"It's live in Open positions now.")
        self._save_book()

        if self.risk.hard_killed and positions and getattr(self, "observe_only", False):
            if not getattr(self, "_ks_muted", False):
                self._ks_muted = True
                self._event("MAX DRAWDOWN KILL SWITCH tripped, but OBSERVE "
                            "ONLY is ON - nothing was sold. Positions are "
                            "untouched and UNMANAGED: no stop, no target. "
                            "Turn observe off or close them yourself.")
        elif self.risk.hard_killed and positions:
            self._event("MAX DRAWDOWN KILL SWITCH — closing all option positions")
            for p in positions:
                self.broker.close_position(p["symbol"])
            self.book.clear()
            self._save_book()

        # ---- DAY LOCK: bank the morning's profit, stop the afternoon bleed ----
        if (self.dl_enabled and clock.get("is_open")
                and not getattr(self, "observe_only", False)):
            if self._dl_date != today:            # new trading day -> reset
                self._dl_date, self._dl_peak, self._dl_locked = today, 0.0, False
            day_start = self.risk._day_start_equity or account["equity"]
            unreal = sum(p.get("unrealized_pl", 0) or 0 for p in positions)
            day_pnl = (account["equity"] + 0 - day_start) / day_start if day_start else 0.0
            self._dl_peak = max(self._dl_peak, day_pnl)
            if not self._dl_locked:
                hit_target = day_pnl >= self.dl_target
                armed = self._dl_peak >= self.dl_arm_pct
                gave_back = armed and day_pnl <= self._dl_peak * (1 - self.dl_giveback)
                if hit_target or gave_back:
                    self._dl_locked = True
                    why = (f"day target +{self.dl_target:.0%} hit" if hit_target else
                           f"gave back {self.dl_giveback:.0%} of the day's peak "
                           f"(+{self._dl_peak:.1%} -> +{day_pnl:.1%})")
                    self._event(f"🔒 DAY LOCK ({why}) — selling everything, done for today "
                                f"at {day_pnl:+.1%}")
                    for p in positions:                      # broker truth, not just the book
                        self.broker.cancel_open_orders(p["symbol"])
                        self.broker.close_position(p["symbol"])
                        self._log_trade({"action": "exit", "symbol": p["symbol"],
                                         "underlying": self.book.get(p["symbol"], {}).get("underlying"),
                                         "reason": "day lock",
                                         "price": p.get("current_price"),
                                         "pnl_pct": None})
                    self.book.clear()
                    self._save_book()

        # signals on underlyings (need market open for both entry & data freshness)
        for und in self.underlyings:
            mode = symbol_mode(self.cfg, und)
            if mode == "off":
                self.last_signals.pop(und, None)
                continue
            df = self.broker.bars(und, self.cfg.timeframe_minutes, self.cfg.lookback_bars)
            if df.empty or len(df) < 60:
                continue
            sig = self.strategy.evaluate(df)
            d = sig.to_dict(); d["mode"] = mode
            self.last_signals[und] = d
            try:
                self.thoughts[und] = self._think(und, sig, clock["is_open"])
            except Exception:
                pass
            try:                               # live mini-graph for the dashboard
                tail = df["close"].tail(120)
                self.charts[und] = [round(float(c), 4) for c in tail]
                self.chart_times[und] = [int(t.timestamp()) for t in tail.index]
                # full OHLCV for the big candlestick chart — served on request by
                # the dashboard's /chart endpoint, kept OUT of state.json so the
                # 3-second refresh stays small
                b = df.tail(400)
                self.bars_cache[und] = {
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
            forced = self.manual_buy.pop(und, None) if clock["is_open"] else None
            if clock["is_open"] and (mode == "buy" or forced):
                self._manage_underlying(und, sig, held, account["equity"],
                                        forced=forced, cash=account.get("cash"),
                                        obp=account.get("options_buying_power"))

        if self.manual_buy and not clock["is_open"]:
            for u, d_ in self.manual_buy.items():
                self._event(f"CAN'T BUY {u} {d_.upper()}: market is closed")
            self.manual_buy.clear()

        if clock["is_open"]:
            self._manage_positions(held)
            self._refresh_candidates()
            self._track_exited()

        self._write_state(account, positions, clock)

    def _refresh_candidates(self, per_cycle: int = 4) -> None:
        """Price an at-the-money contract for a few underlyings each cycle.

        Round-robin: with 20 underlyings and 2 per cycle on a 15s loop, every
        symbol refreshes about every 2.5 minutes for ~4 API calls per cycle.
        Also records a small price history so the premium chart works for
        symbols you have NOT bought.
        """
        if not hasattr(self.broker, "peek_contract"):
            return
        syms = [s for s in self.underlyings if symbol_mode(self.cfg, s) != "off"]
        if not syms:
            return
        now = time.time()
        for _ in range(min(per_cycle, len(syms))):
            und = syms[self._cand_rr % len(syms)]
            self._cand_rr += 1
            sig = self.last_signals.get(und) or {}
            spot = sig.get("price")
            if not spot:
                continue
            # show the side the bot leans toward: uptrend -> call, downtrend -> put
            direction = "call" if (sig.get("regime", 0) or 0) >= 0 else "put"
            prev = self.candidates.get(und) or {}
            # re-use the chosen contract for 15 min; only re-quote it in between
            reuse = (prev.get("symbol") and prev.get("type") == direction
                     and now - prev.get("picked_at", 0) < 900)
            info = None
            if reuse:
                try:
                    q = self.broker._quote(prev["symbol"])
                    if q and q[0] > 0 and q[1] > 0:
                        info = dict(prev)
                        info.update({"bid": round(q[0], 4), "ask": round(q[1], 4),
                                     "mid": round((q[0] + q[1]) / 2, 4)})
                except Exception:
                    info = None
            if info is None:
                try:
                    got = self.broker.peek_contract(und, direction, float(spot),
                                                   self.dte_min, self.dte_max)
                except Exception:
                    got = None
                if not got:
                    continue
                info = dict(got); info["picked_at"] = now
                info["hist"] = prev.get("hist", []) if prev.get("symbol") == got["symbol"] else []
                info["hist_t"] = prev.get("hist_t", []) if prev.get("symbol") == got["symbol"] else []
            h = info.setdefault("hist", []); ht = info.setdefault("hist_t", [])
            h.append(info["mid"]); ht.append(int(now))
            del h[:-400]; del ht[:-400]
            info["cost"] = round(info["mid"] * 100, 2)     # one contract, in dollars
            info["updated"] = int(now)
            self.candidates[und] = info

    def _track_exited(self) -> None:
        """Quote recently-sold contracts for 4h and log the price path.

        Pure logging — reads quotes, writes post_exit_log.jsonl, never trades.
        This is the data that tells us whether an exit rule (stop, signal flip,
        target) is selling at good prices or at the bottom.
        """
        if not self.post_watch:
            return
        now = time.time()
        for sym in list(self.post_watch):
            w = self.post_watch[sym]
            if now - w["exit_ts"] > 4 * 3600:
                self.post_watch.pop(sym, None)
                continue
            try:
                q = self.broker._quote(sym) if hasattr(self.broker, "_quote") else None
            except Exception:
                q = None
            if not q or q[0] <= 0 or q[1] <= 0:
                continue
            try:
                with open(self.state_file.parent / "post_exit_log.jsonl", "a",
                          encoding="utf-8") as f:
                    f.write(json.dumps({
                        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                        "symbol": sym, "underlying": w["underlying"],
                        "mid": round((q[0] + q[1]) / 2, 4),
                        "exit_px": w["exit_px"], "entry": w["entry"],
                        "reason": w["reason"],
                        "min_since_exit": round((now - w["exit_ts"]) / 60)}) + "\n")
            except Exception:
                pass

    def _held_for(self, underlying: str) -> list[str]:
        return [s for s, b in self.book.items() if b.get("underlying") == underlying]

    def _manage_underlying(self, und: str, sig, held: dict, equity: float,
                           forced: str | None = None, cash: float | None = None,
                           obp: float | None = None) -> None:
        if self._dl_locked and not forced:
            return                                   # day's profit is banked — no new buys
        if self.entry_cutoff_utc and not forced:
            now = datetime.now(timezone.utc)
            if now.hour >= self.entry_cutoff_utc:
                today = now.strftime("%Y-%m-%d")
                if self._cutoff_said != today:
                    self._cutoff_said = today
                    self._event(f"⏰ ENTRY CUTOFF — after {self.entry_cutoff_utc}:00 UTC "
                                f"no new positions open (afternoon trades have been the losers). "
                                f"Existing positions still managed normally.")
                return
        held_syms = self._held_for(und)
        if held_syms:
            if not self.alert_only and all(
                    self.book.get(x, {}).get("advice") for x in held_syms):
                self.block[und] = ("broker rejected the auto-buy (unsettled funds) — "
                                   "sent you a buy-it-yourself signal instead; it "
                                   "clears on its own in ~30 min")
            else:
                self.block[und] = "already holding a contract on this one"
            return                                   # one contract per underlying
        direction = forced
        if not direction:
            if sig.composite >= self.call_score and sig.regime > 0:
                direction = "call"
            elif sig.composite <= self.put_score and sig.regime < 0:
                direction = "put"                    # bearish + downtrend -> put
        if not direction:
            if sig.composite >= self.call_score and sig.regime <= 0:
                self.block[und] = ("score says CALL but the trend is DOWN — "
                                   "won't buy calls into a downtrend")
            elif sig.composite <= self.put_score and sig.regime >= 0:
                self.block[und] = ("score says PUT but the trend is UP — "
                                   "won't buy puts into an uptrend")
            else:
                self.block[und] = ""       # simply below the bar: normal waiting
            return
        if forced:
            self._event(f"MANUAL BUY: you clicked Buy {direction.upper()} on {und} — "
                        f"skipping signal checks, sized by your normal premium cap")
        if self.pdt.active(self._equity, self.cfg.paper) and self.pdt.used() >= 3:
            self._event(f"SKIP {und}: PDT guard — 3 day trades used this week")
            return
        real = {s2: bb for s2, bb in self.book.items()
                if not (bb.get("advice") and not self.alert_only)}
        held_unds = [bb["underlying"] for bb in real.values()]
        ok, why = self.risk.can_open(und, len(real), held_unds)
        if not ok:
            self.block[und] = why
            self._event(f"SKIP {und}: {why}")
            return
        # EARNINGS GUARD: never buy into IV crush — premiums deflate after
        # earnings no matter which way the stock moves
        if self.earnings_cal is not None and not forced:
            days = self.earnings_cal.days_until(und)
            if days is not None and days <= self.earnings_skip_days:
                self.block[und] = f"earnings in {days}d — won't buy into IV crush"
                self._event(f"SKIP {und}: earnings in {days}d — won't buy into IV crush")
                return
        pick = self.broker.pick_contract(und, direction, sig.price,
                                         self.dte_min, self.dte_max,
                                         self.max_spread_pct)
        if pick is None:
            self.block[und] = f"no tradeable {direction} — spreads too wide right now"
            self._event(f"SKIP {und}: no tradeable {direction} (bad spreads or no chain)")
            return
        # IV GUARD: refuse premiums priced far above how the stock actually
        # moves — overpaying for volatility is a loss before the trade starts
        if self.iv_guard and not forced and hasattr(self.broker, "implied_vol"):
            iv = self.broker.implied_vol(pick.symbol)
            if iv:
                # realized vol proxy from our own bars: ATR% per 30-min bar,
                # annualized (~13 bars/day, 252 days)
                rv = max((sig.atr / sig.price) * (13 * 252) ** 0.5, 0.05) \
                    if sig.price > 0 else 0.05
                ratio = iv / rv
                if ratio > self.iv_max_ratio:
                    self.block[und] = (f"premium overpriced — IV {iv:.0%} is {ratio:.1f}x "
                                       f"how much the stock actually moves")
                    self._event(f"SKIP {und}: premium overpriced — IV {iv:.0%} is "
                                f"{ratio:.1f}x the stock's actual movement "
                                f"(limit {self.iv_max_ratio:.1f}x)")
                    return
        premium_cap = (self.advice_capital if self.alert_only else equity) * self.max_premium_pct
        cap_src = "cap"
        # CASH-ONLY: long options must be paid for in full — never let the
        # budget exceed the cash actually available (1% cushion for slippage).
        if (not self.alert_only and getattr(self.risk, "cash_only", True)
                and cash is not None):
            c2 = max(0.0, cash * 0.99)
            if c2 < premium_cap:
                premium_cap, cap_src = c2, "cash"
        # SETTLED funds only: the broker rejects option buys on unsettled cash,
        # so respect its own options buying power number up front
        if not self.alert_only and obp is not None:
            c3 = max(0.0, float(obp) * 0.99)
            if c3 < premium_cap:
                premium_cap, cap_src = c3, "settled"
        cost_per_contract = pick.mid * 100           # options are 100 shares/contract
        contracts = int(premium_cap // cost_per_contract)
        if contracts < 1:
            if cap_src == "settled":
                self.block[und] = (f"WAITING FOR FUNDS TO SETTLE — one contract is "
                                   f"${cost_per_contract:,.0f} but only "
                                   f"${premium_cap:,.0f} of your cash is settled "
                                   f"(sales settle the next trading day)")
                if self._settle_said != datetime.now(timezone.utc).strftime("%Y-%m-%d"):
                    self._settle_said = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                    self._event(f"SKIP {und}: waiting for funds to settle — only "
                                f"${premium_cap:,.0f} settled vs ${cost_per_contract:,.0f} "
                                f"needed. Options must be paid with settled cash; "
                                f"yesterday's sale proceeds free up next trading day.")
            else:
                self.block[und] = (f"TOO EXPENSIVE — one contract costs "
                                   f"${cost_per_contract:,.0f}, your cap is ${premium_cap:,.0f}")
                self._event(f"SKIP {und}: one {direction} costs ${cost_per_contract:,.0f}, "
                            f"over the ${premium_cap:,.0f} premium cap")
            return
        if self.observe_only:
            plan_stop = round(pick.mid * (1 - self.stop_pct), 2)
            plan_tp = round(pick.mid * (1 + self.profit_target_pct), 2)
            line = (f"WOULD BUY {und} {direction.upper()} ${pick.strike:g} "
                    f"exp {pick.expiry} @ ~${pick.mid:.2f} "
                    f"(${pick.mid*100*contracts:,.0f} for {contracts}) "
                    f"— score {sig.composite:+.2f}")
            self.would[und] = line
            if self._would_said.get(pick.symbol) != direction:
                self._would_said[pick.symbol] = direction
                self._event("👁 " + line + " — OBSERVE ONLY, no order placed")
                self._notice("buy", pick.symbol,
                             f"WOULD BUY {und} {direction.upper()} ${pick.strike:g}",
                             f"Observe mode: no money moved. The bot would have "
                             f"bought {contracts} contract(s) at ~${pick.mid:.2f}/share "
                             f"(${pick.mid*100*contracts:,.0f}), exp {pick.expiry}, "
                             f"because the score {sig.composite:+.2f} cleared its "
                             f"\u00b1{self.call_score:.2f} bar with the trend behind it. "
                             f"Plan would be: take profit ${plan_tp}, bail ${plan_stop}.")
            return
        if self.alert_only:
            plan_stop = round(pick.mid * (1 - self.stop_pct), 2)
            plan_tp = round(pick.mid * (1 + self.profit_target_pct), 2)
            self.book[pick.symbol] = {
                "underlying": und, "type": direction, "qty": contracts,
                "entry_premium": pick.mid, "expiry": pick.expiry,
                "strike": pick.strike, "advice": True,
                "opened_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                "opened_at": time.time(), "manual_entry": bool(forced),
                "und_entry_px": round(float(sig.price), 4),
            }
            self._save_book()
            self._event(f"📢 BUY NOW ({und} {direction.upper()}): {pick.symbol} — "
                        f"${pick.strike:g} strike, exp {pick.expiry}, "
                        f"~${pick.mid:.2f}/share (${pick.mid*100:.0f}/contract) x{contracts}. "
                        f"Set sell-limit ${plan_tp} · bail below ${plan_stop}. "
                        f"Signal {sig.composite:+.2f}")
            self._log_trade({"action": "enter", "symbol": pick.symbol, "advice": True,
                             "underlying": und, "side": direction, "qty": contracts,
                             "price": pick.mid, "score": round(sig.composite, 4), "und_price": round(float(sig.price), 4),
                             "expiry": pick.expiry, "strike": pick.strike})
            return
        order_id = self.broker.buy_option(pick, contracts)
        if not order_id:
            # HYBRID FALLBACK: broker said no -> announce for manual execution.
            reason = (self.broker.reject_reason()
                      if hasattr(self.broker, "reject_reason") else "other")
            if reason == "approval":
                why = ("this LIVE account is NOT approved to BUY options "
                       "(needs options Level 2 at your broker) — this will "
                       "reject EVERY auto-buy until you fix it")
            elif reason == "funds":
                why = "not enough / unsettled buying power"
            else:
                why = "broker refused the order — see the console log"
            plan_stop = round(pick.mid * (1 - self.stop_pct), 2)
            plan_tp = round(pick.mid * (1 + self.profit_target_pct), 2)
            self.book[pick.symbol] = {
                "underlying": und, "type": direction, "qty": contracts,
                "entry_premium": pick.mid, "expiry": pick.expiry,
                "strike": pick.strike, "advice": True,
                "opened_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                "opened_at": time.time(), "manual_entry": bool(forced),
                "und_entry_px": round(float(sig.price), 4),
            }
            self._save_book()
            self._event(f"AUTO ORDER REJECTED ({why}) — MANUAL SIGNAL instead:")
            self._notice("buy", pick.symbol,
                         f"COULDN'T AUTO-BUY {und} {direction.upper()} ${pick.strike:g}",
                         f"The broker rejected the real order ({why}). If you still "
                         f"want it, buy it yourself: {pick.symbol} at "
                         f"~${pick.mid:.2f}/share (${pick.mid*100:.0f}/contract). "
                         f"The bot's plan was: take profit near "
                         f"${pick.mid*(1+self.profit_target_pct):.2f}, bail near "
                         f"${pick.mid*(1-self.stop_pct):.2f}. This suggestion "
                         f"clears itself in 30 minutes.")
            self._event(f"📢 BUY NOW ({und} {direction.upper()}): {pick.symbol} — "
                        f"${pick.strike:g} strike, exp {pick.expiry}, "
                        f"~${pick.mid:.2f}/share (${pick.mid*100:.0f}/contract) x{contracts}. "
                        f"Set sell-limit ${plan_tp} · bail below ${plan_stop}. "
                        f"Signal {sig.composite:+.2f}")
            self._log_trade({"action": "enter", "symbol": pick.symbol, "advice": True,
                             "underlying": und, "side": direction, "qty": contracts,
                             "price": pick.mid, "score": round(sig.composite, 4), "und_price": round(float(sig.price), 4),
                             "expiry": pick.expiry, "strike": pick.strike})
            return
        if order_id:
            self.block.pop(und, None)
            self.book[pick.symbol] = {
                "underlying": und, "type": direction, "qty": contracts,
                "entry_premium": pick.mid, "expiry": pick.expiry,
                "strike": pick.strike, "pending_since": time.time(),
                "opened_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                "opened_at": time.time(), "manual_entry": bool(forced),
                "und_entry_px": round(float(sig.price), 4),
            }
            self._save_book()
            self._event(f"BUY {direction.upper()} {und} {pick.symbol} x{contracts} "
                        f"@~${pick.mid:.2f} (score {sig.composite:+.2f})")
            trend_w = "UP" if sig.regime > 0 else ("DOWN" if sig.regime < 0 else "flat")
            why = (f"You tapped Buy on the app — signal checks skipped, sized by the "
                   f"normal premium cap." if forced else
                   f"Signal score {sig.composite:+.2f} cleared the "
                   f"\u00b1{self.call_score:.2f} bar with the trend {trend_w} — "
                   f"exactly the setup it hunts for.")
            self._notice("buy", pick.symbol,
                         f"BUYING {und} {direction.upper()} ${pick.strike:g} "
                         f"\u00b7 ${pick.mid:.2f}",
                         f"{contracts} contract{'s' if contracts > 1 else ''} at "
                         f"~${pick.mid:.2f}/share (${pick.mid*100*contracts:,.0f} total), "
                         f"exp {pick.expiry}. {why} Plan: bail near "
                         f"${pick.mid*(1-self.stop_pct):.2f}, take profit near "
                         f"${pick.mid*(1+self.profit_target_pct):.2f}. "
                         f"(Order placed — you'll get a FILLED or "
                         f"CANCELLED notice next.)")
            self._log_trade({"action": "enter", "symbol": pick.symbol,
                             "underlying": und, "side": direction,
                             "qty": contracts, "price": pick.mid,
                             "score": round(sig.composite, 4), "und_price": round(float(sig.price), 4),
                             "expiry": pick.expiry, "strike": pick.strike,
                             "signal_parts": sig.to_dict()["parts"]})

    def _manage_positions(self, held: dict) -> None:
        today = date.today()
        for sym, b in list(self.book.items()):
            if b.get("advice"):
                q = self.broker._quote(sym) if hasattr(self.broker, "_quote") else None
                if not q or q[0] <= 0:
                    continue
                p = {"symbol": sym, "qty": b["qty"], "current_price": (q[0] + q[1]) / 2,
                     "avg_entry": b["entry_premium"]}
            else:
                p = held.get(sym)
            if p is None:
                continue
            entry = b.get("entry_premium") or p["avg_entry"]
            current = p["current_price"]
            pnl_pct = (current - entry) / entry if entry > 0 else 0.0
            # track high-water mark + recent premium wiggle for the ratchet
            hist = b.setdefault("hist", [])
            if not hist or hist[-1] != current:
                hist.append(current)
                del hist[:-30]                       # keep last ~30 observations
            # ---- chart history (display only; `hist` above drives the ratchet) ----
            # Two things matter here:
            #  1. `current` is Alpaca's position price = the LAST TRADE. A thinly
            #     traded contract can go 20+ minutes without a trade, so that
            #     number sits frozen. The live bid/ask MIDPOINT moves continuously,
            #     so the chart follows the mid when a quote is available.
            #  2. Record EVERY cycle, not only when the number changes — a flat
            #     stretch is real information, and skipping it made the chart look
            #     broken and mis-spaced the time axis.
            chart_px = current
            if not b.get("advice") and hasattr(self.broker, "_quote"):
                try:
                    q = self.broker._quote(sym)
                    if q and q[0] > 0 and q[1] > 0:
                        chart_px = (q[0] + q[1]) / 2
                except Exception:
                    pass
            ph = b.setdefault("phist", [])
            pht = b.setdefault("phist_t", [])
            ph.append(round(float(chart_px), 4))
            pht.append(int(time.time()))
            del ph[:-1500]                       # ~6 hours at a 15s loop
            del pht[:-1500]
            if current > b.get("peak", 0):
                b["peak"] = current
            self._save_book()
            peak = b.get("peak", current)
            peak_gain = (peak - entry) / entry if entry > 0 else 0.0
            armed = peak_gain >= self.trail_arm_pct
            trail_level = None
            if armed:
                # adaptive width: typical check-to-check wiggle of THIS contract
                if len(hist) >= 5:
                    moves = [abs(hist[i] - hist[i-1]) / hist[i-1]
                             for i in range(1, len(hist)) if hist[i-1] > 0]
                    wiggle = sum(moves) / len(moves)
                else:
                    wiggle = 0.05
                width = min(self.trail_max_pct,
                            max(self.trail_min_pct, self.trail_vol_mult * wiggle))
                vol_trail = peak * (1 - width)
                # tiered floor: the biggest floor whose tier the peak has reached
                floor_gain = 0.02
                for tier, keep in self.profit_floors:
                    if peak_gain >= tier:
                        floor_gain = keep
                trail_level = max(vol_trail, entry * (1 + floor_gain))
            try:
                dte = (date.fromisoformat(str(b["expiry"])) - today).days
            except Exception:
                dte = 99
            sig = self.last_signals.get(b["underlying"], {})
            comp = sig.get("composite", 0.0)
            flipped = (b["type"] == "call" and comp < -0.05) or \
                      (b["type"] == "put" and comp > 0.05)
            if b.get("manual_entry"):
                flipped = False   # your Buy click overrode the signal — don't insta-exit on it

            earn_tomorrow = False
            if self.earnings_cal is not None:
                ed = self.earnings_cal.days_until(b["underlying"])
                earn_tomorrow = ed is not None and ed <= 1
            manual = sym in self.manual_close
            # fast-bleed cut: a fresh contract already deep red minutes after
            # entry is a wrong-way trade — cut it small instead of riding to -40%
            held_min = (time.time() - b.get("opened_at", 0)) / 60 if b.get("opened_at") else 999
            fast_bleed = (self.fb_enabled and held_min <= self.fb_within_min
                          and pnl_pct <= -self.fb_drop_pct and peak_gain < 0.05)
            reason = None
            if manual:
                self.manual_close.discard(sym)
                reason = "manual close (you clicked Sell)"
            elif pnl_pct >= self.profit_target_pct:
                reason = "profit target"
            elif fast_bleed:
                reason = f"fast-bleed cut (down {pnl_pct:.0%} in {held_min:.0f}m — cut small)"
            elif earn_tomorrow:
                reason = "pre-earnings exit (IV crush guard)"
            elif trail_level is not None and current <= trail_level:
                reason = f"trailing ratchet (peaked +{peak_gain:.0%}, kept the floor)"
            elif pnl_pct <= -self.stop_pct:
                reason = "premium stop"
            elif dte <= self.close_dte:
                reason = f"time exit ({dte}d to expiry)"
            elif flipped:
                reason = "signal flipped"
            if reason and self.observe_only:
                line = (f"WOULD SELL {b['underlying']} "
                        f"{(b.get('type') or '').upper()} ${b.get('strike'):g} "
                        f"({reason}) {pnl_pct:+.0%}")
                self.would[b["underlying"]] = line
                if self._would_said.get("x" + sym) != reason:
                    self._would_said["x" + sym] = reason
                    self._event("👁 " + line + " — OBSERVE ONLY, nothing sold")
                    self._notice("sell", sym,
                                 f"WOULD SELL {b['underlying']} "
                                 f"{(b.get('type') or '').upper()} \u00b7 {pnl_pct:+.0%}",
                                 "Observe mode: nothing was sold. " +
                                 self._exit_story(reason, pnl_pct, entry, current,
                                                  peak_gain, held_min))
                continue
            if reason and b.get("advice"):
                pnl_est = (current - entry) / entry if entry > 0 else 0.0
                self._event(f"📢 SELL NOW: {sym} ({reason}) — quote ~${current:.2f}, "
                            f"est {pnl_est:+.0%} on the ${entry:.2f} entry")
                self._log_trade({"action": "exit", "symbol": sym, "advice": True,
                                 "underlying": b["underlying"], "reason": reason,
                                 "price": current, "pnl_pct": round(pnl_est, 5)})
                self.book.pop(sym, None)
                self._save_book()
                continue
            if reason:
                today_s = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                if (b.get("opened_date") == today_s
                        and self.pdt.active(self._equity, self.cfg.paper)):
                    protective = (manual or fast_bleed or reason == "premium stop"
                                  or current <= entry * 1.02)   # user's order / capital protection: never defer
                    if not protective and self.pdt.used() >= 3:
                        if not b.get("_pdt_deferred"):
                            b["_pdt_deferred"] = True
                            self._event(f"HOLD {sym}: exit deferred — PDT limit")
                        continue
                    self.pdt.record()
                    self._event(f"PDT: day trade used ({self.pdt.used()}/3)")
                self.broker.cancel_open_orders(sym)
                self.broker.close_position(sym)
                self.post_watch[sym] = {"exit_ts": time.time(), "exit_px": current,
                                        "entry": entry, "reason": reason,
                                        "underlying": b["underlying"]}
                self._event(f"CLOSE {sym} ({reason}) {pnl_pct:+.0%} on premium")
                self._notice("sell", sym,
                             f"SOLD {b['underlying']} {(b.get('type') or '').upper()} "
                             f"${b.get('strike'):g} \u00b7 {pnl_pct:+.0%}",
                             self._exit_story(reason, pnl_pct, entry, current,
                                              peak_gain, held_min))
                self._log_trade({"action": "exit", "symbol": sym,
                                 "underlying": b["underlying"], "reason": reason,
                                 "price": current, "pnl_pct": round(pnl_pct, 5)})
                if pnl_pct < 0:
                    self.risk.register_loss(b["underlying"])
                self.book.pop(sym, None)
                self._save_book()

    def _display_stop(self, b: dict) -> float:
        entry = b.get("entry_premium") or 0
        peak = b.get("peak", 0)
        base = entry * (1 - self.stop_pct)
        if entry <= 0 or peak < entry * (1 + self.trail_arm_pct):
            return base
        peak_gain = (peak - entry) / entry
        floor_gain = 0.02
        for tier, keep in self.profit_floors:
            if peak_gain >= tier:
                floor_gain = keep
        return max(base, entry * (1 + floor_gain))

    # ---------- state ----------
    def _write_state(self, account: dict, positions: list, clock: dict) -> None:
        state = {
            "updated": datetime.now(timezone.utc).isoformat(),
            "mode": ("PAPER" if self.cfg.paper else "LIVE") + " · OPTIONS",
            "account": account,
            "market": clock,
            "positions": positions,
            "book": {s: {"side": b["type"], "entry": b["entry_premium"],
                         "stop": round(self._display_stop(b), 2),
                         "tp": round(b["entry_premium"] * (1 + self.profit_target_pct), 2),
                         "qty": b["qty"], "underlying": b.get("underlying"),
                         "und_entry_px": b.get("und_entry_px")}
                     for s, b in self.book.items()},
            "signals": self.last_signals,
            "blocked": {k: v for k, v in self.block.items() if v},
            "candidates": {k: {kk: vv for kk, vv in v.items()
                               if kk not in ("hist", "hist_t")}
                           for k, v in self.candidates.items()},
            "charts": self.charts,
            "chart_times": self.chart_times,
            "premium": {s: {"p": b.get("phist", []), "t": b.get("phist_t", []),
                            "underlying": b.get("underlying"),
                            "entry": b.get("entry_premium"),
                            "stop": round(self._display_stop(b), 2),
                            "tp": round((b.get("entry_premium") or 0)
                                        * (1 + self.profit_target_pct), 2),
                            "type": b.get("type"), "strike": b.get("strike"),
                            "expiry": str(b.get("expiry"))}
                        for s, b in self.book.items()},
            "risk": self.risk.status(),
            "events": self.events[::-1],
            "thoughts": self.thoughts,
            "would": self.would,
            "notices": self.notices[-40:],
            "config": {"entry_threshold": self.call_score,
                       "strictness": self.call_score,
                       "strict_min": 0.10, "strict_max": 0.30,
                       "controls": not self.observe_only,   # no buttons in observe mode
                       "observe_only": self.observe_only,
                       "bot_name": (self.cfg.raw.get("dashboard") or {}).get("name"),
                       "symbols": self.underlyings,
                       "timeframe_minutes": self.cfg.timeframe_minutes},
        }
        self.state_file.write_text(json.dumps(state, indent=1, default=str))

    def run_forever(self) -> None:
        if self.alert_only:
            self._event("MODE: ADVISOR — announces trades, never orders. "
                        "(config: options.execution)")
        else:
            self._event("MODE: AUTO+FALLBACK — places real orders; if the broker "
                        "rejects (unsettled funds), announces the trade for manual entry")
        self._event(f"OPTIONS engine started — "
                    f"{'PAPER' if self.cfg.paper else 'LIVE (REAL MONEY)'}, "
                    f"{'ML model' if type(self.strategy).__name__ == 'MLStrategy' else 'ensemble'} brain, "
                    f"{len(self.underlyings)} underlyings, calls+puts, "
                    f"{self.dte_min}-{self.dte_max} DTE")
        while not self._stop:
            started = time.time()
            try:
                self.cycle()
            except KeyboardInterrupt:
                raise
            except Exception as e:
                log.exception("cycle error: %s", e)
                self._event(f"cycle error: {e}")
            remaining = max(1.0, self.cfg.loop_seconds - (time.time() - started))
            while remaining > 0 and not self._stop:
                if self.manual_close or self.manual_buy:
                    break        # user clicked Buy/Sell — run the next cycle right away
                time.sleep(min(1.0, remaining))
                remaining -= 1.0
        self._event("OPTIONS engine stopped.")
