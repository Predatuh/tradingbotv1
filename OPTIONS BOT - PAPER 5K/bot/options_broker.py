"""Options broker layer: contract discovery, quotes, and orders on Alpaca.

Long calls and puts ONLY — defined risk, the most you can lose on any trade
is the premium paid. No selling/writing options, ever.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta

from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.data.requests import OptionLatestQuoteRequest
from alpaca.trading.enums import AssetStatus, ContractType, OrderSide, TimeInForce
from alpaca.trading.requests import GetOptionContractsRequest, LimitOrderRequest

from .broker import AlpacaBroker

log = logging.getLogger("opt_broker")


@dataclass
class ContractPick:
    symbol: str            # OCC symbol e.g. NVDA260918C00190000
    underlying: str
    type: str              # "call" / "put"
    strike: float
    expiry: str            # YYYY-MM-DD
    bid: float
    ask: float

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2

    @property
    def spread_pct(self) -> float:
        return (self.ask - self.bid) / self.mid if self.mid > 0 else 9.9


class OptionsBroker(AlpacaBroker):
    def __init__(self, api_key: str, api_secret: str, paper: bool = True):
        super().__init__(api_key, api_secret, paper=paper)
        self.option_data = OptionHistoricalDataClient(api_key, api_secret)

    # ---------- contract discovery ----------
    def pick_contract(self, underlying: str, direction: str, spot: float,
                      dte_min: int = 7, dte_max: int = 21,
                      max_spread_pct: float = 0.12) -> ContractPick | None:
        """Nearest-the-money contract inside the DTE window with a sane spread.

        direction: 'call' (bullish) or 'put' (bearish).
        Returns None when nothing tradeable is found — skipping IS a decision.
        """
        today = date.today()
        try:
            req = GetOptionContractsRequest(
                underlying_symbols=[underlying],
                status=AssetStatus.ACTIVE,
                type=ContractType.CALL if direction == "call" else ContractType.PUT,
                expiration_date_gte=today + timedelta(days=dte_min),
                expiration_date_lte=today + timedelta(days=dte_max),
                strike_price_gte=str(round(spot * 0.93, 2)),
                strike_price_lte=str(round(spot * 1.07, 2)),
                limit=300,
            )
            resp = self.trading.get_option_contracts(req)
            contracts = list(resp.option_contracts or [])
        except Exception as e:
            log.error("contract search %s: %s", underlying, e)
            return None
        if not contracts:
            return None

        # prefer: strike closest to spot, then earliest expiry inside the window
        contracts.sort(key=lambda c: (abs(float(c.strike_price) - spot),
                                      str(c.expiration_date)))
        for c in contracts[:6]:                 # try a few in case quotes are bad
            quote = self._quote(c.symbol)
            if quote is None:
                continue
            bid, ask = quote
            if bid <= 0 or ask <= 0:
                continue
            pick = ContractPick(symbol=c.symbol, underlying=underlying,
                                type=direction, strike=float(c.strike_price),
                                expiry=str(c.expiration_date), bid=bid, ask=ask)
            if pick.spread_pct <= max_spread_pct:
                return pick
            log.info("skip %s: spread %.0f%% too wide", c.symbol, pick.spread_pct * 100)
        return None

    def peek_contract(self, underlying: str, direction: str, spot: float,
                      dte_min: int = 7, dte_max: int = 21):
        """CHEAP price check for the watchlist: what would one contract cost?

        pick_contract() quotes every strike in a wide band — far too expensive to
        run for 20 symbols on a loop. This looks at a tight band around the money
        and quotes exactly ONE contract, so a watchlist price costs 1 chain call
        + 1 quote instead of dozens.
        """
        today = date.today()
        try:
            req = GetOptionContractsRequest(
                underlying_symbols=[underlying],
                status=AssetStatus.ACTIVE,
                type=ContractType.CALL if direction == "call" else ContractType.PUT,
                expiration_date_gte=today + timedelta(days=dte_min),
                expiration_date_lte=today + timedelta(days=dte_max),
                strike_price_gte=str(round(spot * 0.97, 2)),
                strike_price_lte=str(round(spot * 1.03, 2)),
                limit=60,
            )
            contracts = list(self.trading.get_option_contracts(req).option_contracts or [])
        except Exception as e:
            log.debug("peek chain %s: %s", underlying, e)
            return None
        if not contracts:
            return None
        # nearest strike to spot, soonest expiry among ties
        best = min(contracts, key=lambda c: (abs(float(c.strike_price) - spot),
                                             str(c.expiration_date)))
        q = self._quote(best.symbol)
        if not q or q[0] <= 0 or q[1] <= 0:
            return None
        bid, ask = q
        return {"symbol": best.symbol, "type": direction,
                "strike": float(best.strike_price), "expiry": str(best.expiration_date),
                "bid": round(bid, 4), "ask": round(ask, 4),
                "mid": round((bid + ask) / 2, 4)}


    def option_bars(self, symbol: str, minutes: int = 5) -> dict | None:
        """Historical premium CANDLES for one contract straight from the
        market data API — powers the premium chart even for contracts
        nobody owns (and survives bot restarts, unlike the tick history)."""
        try:
            from datetime import datetime, timedelta, timezone
            from alpaca.data.requests import OptionBarsRequest
            from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
            req = OptionBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame(minutes, TimeFrameUnit.Minute),
                start=datetime.now(timezone.utc) - timedelta(days=5))
            df = self.option_data.get_option_bars(req).df
            if df is None or len(df) == 0:
                return None
            if getattr(df.index, "nlevels", 1) > 1:
                df = df.droplevel(0)
            df = df.tail(300)
            return {
                "t": [int(ts.timestamp()) for ts in df.index],
                "o": [round(float(v), 4) for v in df["open"]],
                "h": [round(float(v), 4) for v in df["high"]],
                "l": [round(float(v), 4) for v in df["low"]],
                "c": [round(float(v), 4) for v in df["close"]],
                "v": [int(v) for v in df["volume"]],
                "tf": minutes,
            }
        except Exception as e:
            log.info(f"option bars {symbol}: {e}")
            return None

    def _quote(self, symbol: str) -> tuple[float, float] | None:
        try:
            q = self.option_data.get_option_latest_quote(
                OptionLatestQuoteRequest(symbol_or_symbols=symbol))
            quote = q[symbol]
            return float(quote.bid_price), float(quote.ask_price)
        except Exception as e:
            log.warning("quote %s: %s", symbol, e)
            return None

    def implied_vol(self, symbol: str) -> float | None:
        """Contract's implied volatility from Alpaca's snapshot (annualized,
        e.g. 0.45 = 45%). None when the feed doesn't have it — never blocks."""
        try:
            from alpaca.data.requests import OptionSnapshotRequest
            snap = self.option_data.get_option_snapshot(
                OptionSnapshotRequest(symbol_or_symbols=symbol))
            iv = getattr(snap[symbol], "implied_volatility", None)
            return float(iv) if iv else None
        except Exception as e:
            log.debug("iv %s: %s", symbol, e)
            return None

    # ---------- orders ----------
    def buy_option(self, pick: ContractPick, contracts: int) -> str | None:
        """Limit buy at the midpoint plus a small nudge — never chase the ask."""
        if getattr(self, "observe_only", False):
            log.warning("OBSERVE ONLY: refused %s", "buy_option")
            return None
        limit = round(pick.mid + 0.25 * (pick.ask - pick.mid), 2)
        self.last_error = None
        try:
            order = self.trading.submit_order(LimitOrderRequest(
                symbol=pick.symbol, qty=contracts, side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY, limit_price=max(limit, 0.01)))
            log.info("BUY %s x%d limit=%.2f id=%s", pick.symbol, contracts, limit, order.id)
            return str(order.id)
        except Exception as e:
            self.last_error = str(e)          # engine reads this to explain WHY
            log.error("buy_option(%s): %s", pick.symbol, e)
            return None

    def reject_reason(self) -> str:
        """Human summary of the last buy rejection, for the event feed."""
        e = (getattr(self, "last_error", "") or "").lower()
        if "not eligible" in e or "not approved" in e or "not permitted" in e:
            return "approval"          # account can't buy options at all
        if "insufficient" in e or "buying power" in e or "unsettled" in e:
            return "funds"             # not enough / unsettled buying power
        return "other"

    def option_positions(self) -> list[dict]:
        """Only option positions (OCC symbols are long, e.g. NVDA260918C00190000)."""
        return [p for p in self.positions() if len(p["symbol"]) > 12]
