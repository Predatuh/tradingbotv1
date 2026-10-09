"""Backtester: runs the exact same strategy + risk logic over historical bars.

Data sources, in order of preference:
  1. Alpaca historical bars (if API keys are set)  — same data the live bot sees
  2. A local CSV you provide (columns: timestamp,open,high,low,close,volume)
  3. Synthetic data (random-walk with regimes) — for smoke-testing the machinery
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config
from .risk import RiskManager
from .strategy import make_strategy

log = logging.getLogger("backtest")


# ---------------------------------------------------------------- data
def synthetic_bars(n: int = 3000, start_price: float = 100.0, seed: int = 7) -> pd.DataFrame:
    """Regime-switching random walk so the strategy has trends to find."""
    rng = np.random.default_rng(seed)
    drift = 0.0
    prices, vols = [], []
    price = start_price
    for i in range(n):
        if i % 250 == 0:                       # new regime every ~250 bars
            drift = rng.normal(0, 0.0006)
        vol = 0.002 * (1 + 0.5 * np.sin(i / 120))
        price *= 1 + rng.normal(drift, vol)
        prices.append(price)
        vols.append(vol)
    close = np.array(prices)
    spread = close * np.array(vols)
    high = close + np.abs(rng.normal(0, 1, n)) * spread
    low = close - np.abs(rng.normal(0, 1, n)) * spread
    open_ = np.roll(close, 1); open_[0] = start_price
    volume = rng.lognormal(10, 0.5, n) * (1 + 5 * np.abs(np.diff(np.append(close[0], close)) / close))
    idx = pd.date_range(end=datetime.now(timezone.utc), periods=n, freq="5min")
    return pd.DataFrame({"open": open_, "high": high, "low": low,
                         "close": close, "volume": volume}, index=idx)


def alpaca_bars(cfg: Config, symbol: str, days: int) -> pd.DataFrame:
    from .broker import AlpacaBroker, is_crypto  # late import: keys may be absent
    from alpaca.data.requests import CryptoBarsRequest, StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    broker = AlpacaBroker(cfg.api_key, cfg.api_secret, paper=cfg.paper)
    tf = TimeFrame(cfg.timeframe_minutes, TimeFrameUnit.Minute)
    start = datetime.now(timezone.utc) - timedelta(days=days)
    if is_crypto(symbol):
        req = CryptoBarsRequest(symbol_or_symbols=symbol, timeframe=tf, start=start)
        df = broker.crypto_data.get_crypto_bars(req).df
    else:
        req = StockBarsRequest(symbol_or_symbols=symbol, timeframe=tf, start=start)
        df = broker.stock_data.get_stock_bars(req).df
    if isinstance(df.index, pd.MultiIndex):
        df = df.xs(symbol, level=0)
    return df[["open", "high", "low", "close", "volume"]]


# ---------------------------------------------------------------- engine
@dataclass
class BacktestResult:
    symbol: str
    trades: list = field(default_factory=list)
    equity_curve: list = field(default_factory=list)

    def summary(self) -> dict:
        eq = pd.Series([e for _, e in self.equity_curve])
        wins = [t for t in self.trades if t["pnl_pct"] > 0]
        losses = [t for t in self.trades if t["pnl_pct"] <= 0]
        total_return = eq.iloc[-1] / eq.iloc[0] - 1 if len(eq) > 1 else 0.0
        peak = eq.cummax()
        max_dd = ((eq - peak) / peak).min() if len(eq) > 1 else 0.0
        rets = eq.pct_change().dropna()
        sharpe = float(rets.mean() / rets.std() * np.sqrt(252 * 78)) if len(rets) > 2 and rets.std() > 0 else 0.0
        avg_win = np.mean([t["pnl_pct"] for t in wins]) if wins else 0.0
        avg_loss = np.mean([t["pnl_pct"] for t in losses]) if losses else 0.0
        gross_w = sum(t["pnl_pct"] for t in wins)
        gross_l = abs(sum(t["pnl_pct"] for t in losses))
        return {
            "symbol": self.symbol,
            "trades": len(self.trades),
            "win_rate": round(len(wins) / len(self.trades), 3) if self.trades else 0.0,
            "total_return_pct": round(total_return * 100, 2),
            "max_drawdown_pct": round(float(max_dd) * 100, 2),
            "sharpe_est": round(sharpe, 2),
            "avg_win_pct": round(float(avg_win) * 100, 3),
            "avg_loss_pct": round(float(avg_loss) * 100, 3),
            "profit_factor": round(gross_w / gross_l, 2) if gross_l > 0 else None,
        }


def run_backtest(cfg: Config, df: pd.DataFrame, symbol: str = "SYNTH",
                 starting_equity: float = 100_000.0, fee_pct: float = 0.0005,
                 slippage_pct: float = 0.0003) -> BacktestResult:
    strategy = make_strategy(cfg.strategy)
    risk = RiskManager(cfg.risk)
    result = BacktestResult(symbol=symbol)

    # drive the risk manager's clock from BAR time, not wall-clock time,
    # so post-loss cooldowns work correctly inside the simulation
    sim_now = {"t": df.index[0].timestamp()}
    risk.now_fn = lambda: sim_now["t"]

    # vectorized: all signals for all bars in one pass (fast)
    signals = strategy.evaluate_all(df)

    equity = starting_equity
    pos = None  # {"side","entry","stop","tp","qty"}
    warmup = 60

    for i in range(warmup, len(df)):
        sim_now["t"] = df.index[i].timestamp()
        sig = type(strategy).signal_from_row(signals.iloc[i])
        price = sig.price
        bar = df.iloc[i]
        today = str(df.index[i].date())
        risk.update_equity(equity + (0 if not pos else _open_pnl(pos, price)), today)

        if pos:
            pos["stop"] = risk.trail_stop(pos["side"], pos["stop"], price, sig.atr)
            exit_price, reason = None, None
            if pos["side"] == "long":
                if bar["low"] <= pos["stop"]:
                    exit_price, reason = pos["stop"], "stop-loss"
                elif bar["high"] >= pos["tp"]:
                    exit_price, reason = pos["tp"], "take-profit"
                elif strategy.wants_exit(sig, "long"):
                    exit_price, reason = price, "signal fade"
            else:
                if bar["high"] >= pos["stop"]:
                    exit_price, reason = pos["stop"], "stop-loss"
                elif bar["low"] <= pos["tp"]:
                    exit_price, reason = pos["tp"], "take-profit"
                elif strategy.wants_exit(sig, "short"):
                    exit_price, reason = price, "signal fade"
            if exit_price is not None:
                exit_price *= (1 - slippage_pct) if pos["side"] == "long" else (1 + slippage_pct)
                pnl = (exit_price - pos["entry"]) / pos["entry"]
                pnl = pnl if pos["side"] == "long" else -pnl
                pnl -= 2 * fee_pct
                equity *= 1 + pnl * (pos["qty"] * pos["entry"] / equity)
                result.trades.append({"i": i, "side": pos["side"], "entry": pos["entry"],
                                      "exit": round(exit_price, 4), "reason": reason,
                                      "pnl_pct": round(pnl, 5)})
                if pnl < 0:
                    risk.register_loss(symbol)
                pos = None
        else:
            side = strategy.wants_entry(sig)
            ok, _ = risk.can_open(symbol, 0)
            if side and ok:
                plan = risk.plan_position(side, price, sig.atr, equity, fractionable=True)
                if plan and plan.notional >= 10:
                    entry = price * ((1 + slippage_pct) if side == "long" else (1 - slippage_pct))
                    pos = {"side": side, "entry": entry, "stop": plan.stop_price,
                           "tp": plan.take_profit, "qty": plan.qty}
        result.equity_curve.append((str(df.index[i]), round(equity, 2)))

    # close any open position at the end
    if pos:
        price = float(df["close"].iloc[-1])
        pnl = (price - pos["entry"]) / pos["entry"]
        pnl = pnl if pos["side"] == "long" else -pnl
        equity *= 1 + pnl * (pos["qty"] * pos["entry"] / equity)
        result.trades.append({"i": len(df) - 1, "side": pos["side"], "entry": pos["entry"],
                              "exit": price, "reason": "end", "pnl_pct": round(pnl, 5)})
        result.equity_curve.append((str(df.index[-1]), round(equity, 2)))
    return result


def _open_pnl(pos: dict, price: float) -> float:
    d = (price - pos["entry"]) * pos["qty"]
    return d if pos["side"] == "long" else -d
