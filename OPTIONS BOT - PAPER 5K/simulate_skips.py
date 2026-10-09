#!/usr/bin/env python3
"""What if the bot COULD have bought the trades it skipped over the premium cap?

Reads the SKIP events from a bot's state file, re-picks the contract the bot
would have chosen at that moment, pulls that contract's real minute-by-minute
prices from Alpaca, and replays the bot's own exit rules (+60% target, -40%
stop, trailing ratchet v2). Verdict per skipped trade: where it would be now.

    python simulate_skips.py                                    # standalone folder (state.json + .env)
    python simulate_skips.py --state dist\TradingBot\state_<botid>.json   # fleet app bot

Honest limits: the signal-flip exit isn't replayed (needs the full signal
history), and the re-picked contract is the closest reconstruction, not a
certainty. Same-day skips only — the events file doesn't store dates.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

SKIP_RE = re.compile(r"SKIP (\S+): one (call|put) costs \$([\d,]+), over the \$[\d,]+ premium cap")
BUY_RE = re.compile(r"BUY (CALL|PUT) (\S+) (\S+) x(\d+) @~\$([\d.]+)")

# the bot's exit rules (mirror of options_engine)
PROFIT_TARGET, STOP = 0.60, 0.40
TRAIL_ARM, TRAIL_VOL_MULT, TRAIL_MIN, TRAIL_MAX = 0.25, 4.0, 0.10, 0.30
FLOORS = [[0.25, 0.05], [0.40, 0.20], [0.50, 0.30], [0.75, 0.45], [1.00, 0.60]]


def parse_episodes(events: list[dict], day: str):
    """Group SKIP events into distinct trade attempts (gap > 30 min = new one)."""
    skips, buys = [], []
    for e in events:
        m = SKIP_RE.search(e.get("msg", ""))
        if m:
            ts = datetime.fromisoformat(f"{day}T{e['t']}+00:00")
            skips.append({"sym": m.group(1), "dir": m.group(2), "t": ts,
                          "premium": float(m.group(3).replace(",", "")) / 100.0})
            continue
        m = BUY_RE.search(e.get("msg", ""))
        if m:
            buys.append({"sym": m.group(2), "dir": m.group(1).lower(),
                         "t": datetime.fromisoformat(f"{day}T{e['t']}+00:00"),
                         "premium": float(m.group(5))})
    episodes = []
    for s in sorted(skips, key=lambda x: x["t"]):
        for ep in episodes:
            if (ep["sym"], ep["dir"]) == (s["sym"], s["dir"]) and \
                    s["t"] - ep["last"] <= timedelta(minutes=30):
                ep["last"] = s["t"]
                ep["seen"].append(s["premium"])
                break
        else:
            episodes.append({"sym": s["sym"], "dir": s["dir"], "start": s["t"],
                             "last": s["t"], "entry_logged": s["premium"],
                             "seen": [s["premium"]]})
    for ep in episodes:
        ep["later_buy"] = next((b for b in buys if b["sym"] == ep["sym"]
                                and b["dir"] == ep["dir"] and b["t"] >= ep["start"]), None)
    return episodes


def simulate_exits(bars: list[tuple], entry: float):
    """Replay ratchet v2 + target + stop over (ts, close) bars. Returns verdict dict."""
    peak, hist = entry, [entry]
    for ts, price in bars:
        hist.append(price)
        del hist[:-30]
        peak = max(peak, price)
        pnl = (price - entry) / entry
        peak_gain = (peak - entry) / entry
        if pnl >= PROFIT_TARGET:
            return {"exit": "profit target", "t": ts, "price": price, "pnl": pnl}
        if peak_gain >= TRAIL_ARM:
            moves = [abs(hist[i] - hist[i - 1]) / hist[i - 1]
                     for i in range(1, len(hist)) if hist[i - 1] > 0]
            wiggle = (sum(moves) / len(moves)) if len(moves) >= 4 else 0.05
            width = min(TRAIL_MAX, max(TRAIL_MIN, TRAIL_VOL_MULT * wiggle))
            floor_gain = 0.02
            for tier, keep in FLOORS:
                if peak_gain >= tier:
                    floor_gain = keep
            trail = max(peak * (1 - width), entry * (1 + floor_gain))
            if price <= trail:
                return {"exit": f"trailing ratchet (peaked {peak_gain:+.0%})",
                        "t": ts, "price": price, "pnl": pnl}
        if pnl <= -STOP:
            return {"exit": "premium stop", "t": ts, "price": price, "pnl": pnl}
    last = bars[-1] if bars else (None, entry)
    return {"exit": None, "t": last[0], "price": last[1],
            "pnl": (last[1] - entry) / entry,
            "peak_gain": (peak - entry) / entry}


def load_keys(state_path: Path):
    """Keys: .env in cwd, else fleet.json matching the state file's bot id."""
    import os
    env = Path(".env")
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
        k, s = os.environ.get("ALPACA_API_KEY", ""), os.environ.get("ALPACA_API_SECRET", "")
        if k and s:
            return k, s
    m = re.match(r"state_(\w+)", state_path.stem)
    for fleet_file in (state_path.parent / "fleet.json", Path("fleet.json")):
        if fleet_file.exists():
            bots = json.loads(fleet_file.read_text()).get("bots", {})
            spec = bots.get(m.group(1)) if m else None
            spec = spec or next(iter(bots.values()), None)
            if spec and spec.get("keys", {}).get("key"):
                return spec["keys"]["key"], spec["keys"]["secret"]
    raise SystemExit("No API keys found (.env or fleet.json). Pass --key and --secret.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default="state.json")
    ap.add_argument("--key"), ap.add_argument("--secret")
    args = ap.parse_args()
    state_path = Path(args.state)
    if not state_path.exists():
        raise SystemExit(f"{state_path} not found — point --state at the bot's state file.")
    st = json.loads(state_path.read_text())
    day = (st.get("updated") or datetime.now(timezone.utc).isoformat())[:10]
    episodes = parse_episodes(st.get("events", []), day)
    if not episodes:
        raise SystemExit("No premium-cap SKIP events in this state file.")
    key, secret = (args.key, args.secret) if args.key else load_keys(state_path)

    from alpaca.data.historical.option import OptionHistoricalDataClient
    from alpaca.data.historical.stock import StockHistoricalDataClient
    from alpaca.data.requests import OptionBarsRequest, StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import GetOptionContractsRequest

    stocks = StockHistoricalDataClient(key, secret)
    opts = OptionHistoricalDataClient(key, secret)
    try:
        trading = TradingClient(key, secret, paper=True)
        trading.get_account()
    except Exception:
        trading = TradingClient(key, secret, paper=False)

    print(f"\n===== WHAT THE SKIPPED TRADES WOULD HAVE DONE ({day}) =====")
    print("(re-picked contracts + real Alpaca minute bars + the bot's own exit rules;")
    print(" signal-flip exits not replayed — results are 'if held per price rules')\n")
    for ep in episodes:
        sym, d = ep["sym"], ep["dir"]
        label = f"{sym} {d.upper()} first wanted {ep['start'].strftime('%H:%M')} UTC @ ~${ep['entry_logged']:.2f}"
        try:
            sb = stocks.get_stock_bars(StockBarsRequest(
                symbol_or_symbols=sym, timeframe=TimeFrame.Minute,
                start=ep["start"] - timedelta(minutes=3), end=ep["start"] + timedelta(minutes=3)))
            spot = list(sb.data.values())[0][-1].close
            today = datetime.now(timezone.utc).date()
            chain = trading.get_option_contracts(GetOptionContractsRequest(
                underlying_symbols=[sym], type=d,
                expiration_date_gte=(ep["start"].date() + timedelta(days=6)).isoformat(),
                expiration_date_lte=(ep["start"].date() + timedelta(days=21)).isoformat(),
                strike_price_gte=str(round(spot * 0.93, 2)),
                strike_price_lte=str(round(spot * 1.07, 2)), limit=200)).option_contracts
            if not chain:
                print(f"  {label}\n    -> no matching chain found, skipped\n")
                continue
            pick = min(chain, key=lambda c: (abs(float(c.strike_price) - spot),
                                             c.expiration_date))
            ob = opts.get_option_bars(OptionBarsRequest(
                symbol_or_symbols=pick.symbol, timeframe=TimeFrame.Minute,
                start=ep["start"], end=datetime.now(timezone.utc)))
            bars = [(b.timestamp, b.close) for b in list(ob.data.values())[0]] \
                if ob.data else []
            if not bars:
                print(f"  {label}\n    -> no option bars available yet, skipped\n")
                continue
            entry = bars[0][1]
            r = simulate_exits(bars[1:], entry)
            print(f"  {label}")
            print(f"    contract {pick.symbol} (strike {pick.strike_price}, exp {pick.expiration_date})")
            note = "" if abs(entry - ep["entry_logged"]) / ep["entry_logged"] < 0.15 else \
                "  [entry differs from log — likely a different strike than the bot saw]"
            print(f"    sim entry ${entry:.2f} (log said ${ep['entry_logged']:.2f}){note}")
            if r["exit"]:
                print(f"    -> EXITED: {r['exit']} at {r['t'].strftime('%H:%M')} "
                      f"@ ${r['price']:.2f} = {r['pnl']:+.1%} (${r['pnl'] * entry * 100:+,.0f}/contract)")
            else:
                print(f"    -> STILL OPEN at ${r['price']:.2f} = {r['pnl']:+.1%} "
                      f"(${r['pnl'] * entry * 100:+,.0f}/contract, peaked {r.get('peak_gain', 0):+.1%})")
            if ep["later_buy"]:
                b = ep["later_buy"]
                print(f"    note: the bot DID enter later at {b['t'].strftime('%H:%M')} "
                      f"@ ${b['premium']:.2f} ({b['sym']})")
            print()
        except Exception as e:
            print(f"  {label}\n    -> couldn't simulate: {str(e)[:140]}\n")
    print("Remember: one afternoon proves nothing either way — this answers")
    print("'what happened today', not 'should the cap change'.")


if __name__ == "__main__":
    main()
