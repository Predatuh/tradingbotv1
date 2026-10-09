"""Alpaca broker layer: market data + order execution for stocks and crypto.

Everything the engine needs from Alpaca goes through this one class, so the
rest of the bot never touches the SDK directly (and the backtester can swap
in a simulated broker with the same interface).
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import pandas as pd

from alpaca.data.historical import CryptoHistoricalDataClient, StockHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest, StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import LimitOrderRequest, MarketOrderRequest

log = logging.getLogger("broker")


def is_crypto(symbol: str) -> bool:
    return "/" in symbol


class AlpacaBroker:
    def __init__(self, api_key: str, api_secret: str, paper: bool = True):
        self.trading = TradingClient(api_key, api_secret, paper=paper)
        self.stock_data = StockHistoricalDataClient(api_key, api_secret)
        self.crypto_data = CryptoHistoricalDataClient(api_key, api_secret)

    # ---------- account ----------
    def account(self) -> dict:
        a = self.trading.get_account()
        # multiplier "1" = CASH account (no margin). PDT rules are a margin-account
        # rule — they do NOT apply to cash accounts. Cash accounts instead have
        # settlement rules (proceeds must settle before reuse).
        mult = str(getattr(a, "multiplier", "1") or "1")
        return {
            "equity": float(a.equity),
            "cash": float(a.cash),
            "buying_power": float(a.buying_power),
            "currency": a.currency,
            "is_cash_account": mult in ("1", "1.0"),
            "multiplier": mult,
            "options_buying_power": (float(a.options_buying_power)
                                     if getattr(a, "options_buying_power", None)
                                     is not None else None),
            "daytrade_count": int(getattr(a, "daytrade_count", 0) or 0),
            "flagged_pdt": bool(getattr(a, "pattern_day_trader", False)),
        }

    def clock(self) -> dict:
        c = self.trading.get_clock()
        return {"is_open": c.is_open,
                "next_open": str(c.next_open), "next_close": str(c.next_close)}

    # ---------- positions ----------
    def positions(self) -> list[dict]:
        out = []
        for p in self.trading.get_all_positions():
            out.append({
                "symbol": p.symbol,
                "qty": float(p.qty),
                "side": "long" if float(p.qty) > 0 else "short",
                "avg_entry": float(p.avg_entry_price),
                "market_value": float(p.market_value),
                "unrealized_pl": float(p.unrealized_pl),
                "unrealized_plpc": float(p.unrealized_plpc),
                "current_price": float(p.current_price),
            })
        return out

    def close_position(self, symbol: str) -> None:
        api_sym = symbol.replace("/", "")
        try:
            self.trading.close_position(api_sym)
            log.info("Closed position %s", symbol)
        except Exception as e:  # position may already be gone
            log.warning("close_position(%s): %s", symbol, e)

    def close_all(self) -> None:
        try:
            self.trading.close_all_positions(cancel_orders=True)
            log.warning("FLATTENED ALL POSITIONS")
        except Exception as e:
            log.error("close_all: %s", e)

    # ---------- orders ----------
    def submit_entry(self, symbol: str, side: str, qty: float,
                     ref_price: float | None = None) -> str | None:
        """Entry order. With ref_price: a MARKETABLE LIMIT — fills like a market
        order in normal conditions but caps how bad the fill can be if the
        spread is wide or the price gaps (0.2% worst-case vs the signal price).
        Without ref_price: falls back to a plain market order."""
        try:
            api_sym = symbol.replace("/", "")
            tif = TimeInForce.GTC if is_crypto(symbol) else TimeInForce.DAY
            o_side = OrderSide.BUY if side == "long" else OrderSide.SELL
            if ref_price and ref_price > 0:
                pad = 1.002 if side == "long" else 0.998
                limit = round(ref_price * pad, 2 if ref_price >= 1 else 6)
                order = self.trading.submit_order(LimitOrderRequest(
                    symbol=api_sym, qty=round(qty, 9), side=o_side,
                    time_in_force=tif, limit_price=limit))
                log.info("ENTRY %s %s qty=%.6f limit=%.4f id=%s",
                         side.upper(), symbol, qty, limit, order.id)
            else:
                order = self.trading.submit_order(MarketOrderRequest(
                    symbol=api_sym, qty=round(qty, 9), side=o_side, time_in_force=tif))
                log.info("ENTRY %s %s qty=%.6f (market) id=%s", side.upper(), symbol, qty, order.id)
            return str(order.id)
        except Exception as e:
            log.error("submit_entry(%s): %s", symbol, e)
            return None

    def cancel_open_orders(self, symbol: str) -> None:
        try:
            for o in self.trading.get_orders():
                if o.symbol == symbol.replace("/", ""):
                    self.trading.cancel_order_by_id(o.id)
        except Exception as e:
            log.warning("cancel_open_orders(%s): %s", symbol, e)

    # ---------- market data ----------
    def bars(self, symbol: str, minutes: int, lookback: int) -> pd.DataFrame:
        """Recent OHLCV bars as a DataFrame (open, high, low, close, volume)."""
        tf = TimeFrame(minutes, TimeFrameUnit.Minute)
        start = datetime.now(timezone.utc) - timedelta(minutes=minutes * (lookback + 10) * 2)
        try:
            if is_crypto(symbol):
                req = CryptoBarsRequest(symbol_or_symbols=symbol, timeframe=tf, start=start)
                bars = self.crypto_data.get_crypto_bars(req)
            else:
                # feed="iex": REAL-TIME bars. The account default is 15-minute
                # DELAYED consolidated data — the bot was seeing (and trading)
                # the market a quarter hour late. IEX covers less volume but
                # its prices track the market within pennies on liquid names.
                try:
                    from alpaca.data.enums import DataFeed
                    req = StockBarsRequest(symbol_or_symbols=symbol, timeframe=tf,
                                           start=start, feed=DataFeed.IEX)
                    bars = self.stock_data.get_stock_bars(req)
                except Exception:
                    req = StockBarsRequest(symbol_or_symbols=symbol, timeframe=tf,
                                           start=start)
                    bars = self.stock_data.get_stock_bars(req)
            df = bars.df
            if df.empty:
                return pd.DataFrame()
            if isinstance(df.index, pd.MultiIndex):
                df = df.xs(symbol, level=0)
            df = df[["open", "high", "low", "close", "volume"]].tail(lookback)
            return df
        except Exception as e:
            log.error("bars(%s): %s", symbol, e)
            return pd.DataFrame()
