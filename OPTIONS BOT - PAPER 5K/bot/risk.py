"""Risk management: position sizing, stops, kill switches, cooldowns.

This module is deliberately conservative — it is the difference between a
bot that survives and one that donates the account to the market.
"""
from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class PositionPlan:
    qty: float
    stop_price: float
    take_profit: float
    notional: float


class RiskManager:
    def __init__(self, cfg: dict, anchor_file=None):
        # anchor_file persists the day-start equity so "up today" survives bot
        # restarts: the day runs midnight-to-midnight, not restart-to-restart.
        self.anchor_file = anchor_file
        self.equity_risk_per_trade = float(cfg.get("equity_risk_per_trade", 0.01))
        self.max_position_pct = float(cfg.get("max_position_pct", 0.10))
        # CASH-ONLY: never spend more than the cash actually sitting in the
        # account. Sizing is equity-based, and equity does NOT shrink as cash is
        # spent — so with positions already open, a later buy could quietly be
        # funded with MARGIN (borrowed money). This makes every bot behave like
        # the live account does: cash in hand, nothing borrowed.
        self.cash_only = bool(cfg.get("cash_only", True))
        self.max_open_positions = int(cfg.get("max_open_positions", 6))
        self.stop_atr = float(cfg.get("stop_loss_atr_mult", 2.0))
        self.tp_atr = float(cfg.get("take_profit_atr_mult", 3.5))
        self.trailing = bool(cfg.get("trailing_stop", True))
        self.max_daily_loss_pct = float(cfg.get("max_daily_loss_pct", 0.03))
        self.max_drawdown_pct = float(cfg.get("max_drawdown_pct", 0.10))
        self.cooldown_s = int(cfg.get("cooldown_minutes_after_loss", 30)) * 60
        self.now_fn = time.time                  # injectable clock (backtester overrides)
        # correlation guard: symbol -> group, max concurrent positions per group
        self.corr_groups: dict[str, str] = {}
        for group, syms in (cfg.get("correlation_groups") or {}).items():
            for s in syms:
                self.corr_groups[s] = group
        self.max_per_group = int(cfg.get("max_positions_per_group", 2))

        self._cooldowns: dict[str, float] = {}   # symbol -> unix time cooldown ends
        self._day_start_equity: float | None = None
        self._peak_equity: float | None = None
        self._day: str | None = None
        self.halted_today = False
        self.hard_killed = False

    # ---------- sizing ----------
    def plan_position(self, side: str, price: float, atr: float,
                      equity: float, fractionable: bool,
                      cash: float | None = None) -> PositionPlan | None:
        if price <= 0 or equity <= 0:
            return None
        atr = max(atr, price * 0.002)               # floor ATR at 0.2% so stops are never zero-width
        stop_dist = self.stop_atr * atr
        risk_dollars = equity * self.equity_risk_per_trade
        qty = risk_dollars / stop_dist
        # cap by max position size
        max_notional = equity * self.max_position_pct
        if self.cash_only and cash is not None:
            # leave a 1% cushion so fees/slippage can't tip it into borrowing
            max_notional = min(max_notional, max(0.0, cash * 0.99))
        if qty * price > max_notional:
            qty = max_notional / price
        if not fractionable:
            qty = float(int(qty))
        if qty <= 0:
            return None
        if side == "long":
            stop = price - stop_dist
            tp = price + self.tp_atr * atr
        else:
            stop = price + stop_dist
            tp = price - self.tp_atr * atr
        return PositionPlan(qty=qty, stop_price=round(stop, 4),
                            take_profit=round(tp, 4), notional=qty * price)

    # ---------- trailing stop ----------
    def trail_stop(self, side: str, current_stop: float, price: float, atr: float) -> float:
        if not self.trailing:
            return current_stop
        atr = max(atr, price * 0.002)
        if side == "long":
            candidate = price - self.stop_atr * atr
            return max(current_stop, candidate)
        candidate = price + self.stop_atr * atr
        return min(current_stop, candidate)

    # ---------- kill switches ----------
    def update_equity(self, equity: float, today: str) -> None:
        if self._day != today:
            self._day = today
            self._day_start_equity = equity
            self.halted_today = False
            # a RESTART mid-day is not a new day: reuse the day's stored anchor
            # so the dashboard's "up today" tells the truth across restarts
            if self.anchor_file is not None:
                import json as _json
                try:
                    a = _json.loads(self.anchor_file.read_text())
                    if a.get("date") == today and a.get("equity"):
                        self._day_start_equity = float(a["equity"])
                except Exception:
                    pass
                try:
                    self.anchor_file.write_text(_json.dumps(
                        {"date": today, "equity": self._day_start_equity}))
                except Exception:
                    pass
        if self._peak_equity is None or equity > self._peak_equity:
            self._peak_equity = equity

        if self._day_start_equity and self._day_start_equity > 0:
            day_pnl = (equity - self._day_start_equity) / self._day_start_equity
            if day_pnl <= -self.max_daily_loss_pct:
                self.halted_today = True
        if self._peak_equity and self._peak_equity > 0:
            dd = (equity - self._peak_equity) / self._peak_equity
            if dd <= -self.max_drawdown_pct:
                self.hard_killed = True

    def can_open(self, symbol: str, open_positions: int,
                 held_symbols: list[str] | None = None) -> tuple[bool, str]:
        if self.hard_killed:
            return False, "hard kill switch (max drawdown) active"
        if self.halted_today:
            return False, "daily loss limit hit — no new trades today"
        if open_positions >= self.max_open_positions:
            return False, "max open positions reached"
        if self._cooldowns.get(symbol, 0) > self.now_fn():
            return False, f"{symbol} in post-loss cooldown"
        # correlation guard: don't stack the same bet under different tickers
        group = self.corr_groups.get(symbol)
        if group and held_symbols:
            same_group = sum(1 for s in held_symbols if self.corr_groups.get(s) == group)
            if same_group >= self.max_per_group:
                return False, (f"correlation guard: already holding {same_group} "
                               f"'{group}' positions")
        return True, ""

    def register_loss(self, symbol: str) -> None:
        self._cooldowns[symbol] = self.now_fn() + self.cooldown_s

    def status(self) -> dict:
        return {
            "halted_today": self.halted_today,
            "hard_killed": self.hard_killed,
            "day_start_equity": self._day_start_equity,
            "peak_equity": self._peak_equity,
            "cooldowns": {s: max(0, int(t - self.now_fn())) for s, t in self._cooldowns.items()
                          if t > self.now_fn()},
        }
