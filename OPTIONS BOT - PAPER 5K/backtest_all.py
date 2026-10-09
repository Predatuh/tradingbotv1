#!/usr/bin/env python3
"""Simulate the WHOLE watchlist over real recent history in one shot.

    python backtest_all.py            # every symbol, last 30 days
    python backtest_all.py 60         # last 60 days
    python backtest_all.py 30 stocks  # only the stock symbols

This is a month of market replayed through the exact strategy in seconds.
It uses real Alpaca history (needs your keys in .env). Read the combined
profit factor at the bottom — that is the number that says whether the edge
is real, and it comes from hundreds of simulated trades instead of the 8-11
you have live. Backtests are optimistic (perfect fills, no news gaps), so a
strategy that can't clear PF ~1.2 here won't survive live.
"""
import sys

from bot.backtest import alpaca_bars, run_backtest
from bot.config import Config


def main():
    cfg = Config.load()
    cfg.validate_keys()
    days = 30
    which = "all"
    for a in sys.argv[1:]:
        if a.isdigit():
            days = int(a)
        elif a in ("stocks", "crypto"):
            which = a

    syms = []
    if which in ("all", "stocks"):
        syms += cfg.raw.get("symbols", {}).get("stocks", [])
    if which in ("all", "crypto"):
        syms += cfg.raw.get("symbols", {}).get("crypto", [])
    # options bots list their underlyings separately
    syms += [s for s in cfg.raw.get("options", {}).get("underlyings", []) if s not in syms]
    if not syms:
        raise SystemExit("No symbols found in config.yaml.")

    print(f"\n  Backtesting {len(syms)} symbols over the last {days} days "
          f"(real market data)...\n")
    print(f"  {'SYMBOL':10} {'trades':>6} {'win%':>6} {'return':>8} "
          f"{'maxDD':>7} {'PF':>6}")
    print("  " + "-" * 50)

    all_trades, gross_w, gross_l = 0, 0.0, 0.0
    rows = []
    for sym in syms:
        try:
            df = alpaca_bars(cfg, sym, days)
            if df is None or len(df) < 80:
                print(f"  {sym:10} {'(not enough data)':>30}")
                continue
            r = run_backtest(cfg, df, symbol=sym).summary()
            pf = r["profit_factor"]
            print(f"  {sym:10} {r['trades']:>6} {r['win_rate']*100:>5.0f}% "
                  f"{r['total_return_pct']:>7.1f}% {r['max_drawdown_pct']:>6.1f}% "
                  f"{(pf if pf is not None else 0):>6.2f}")
            all_trades += r["trades"]
            rows.append(r)
        except Exception as e:
            print(f"  {sym:10} error: {str(e)[:40]}")

    # combine into one honest verdict
    for r in rows:
        aw, al, n = r["avg_win_pct"], r["avg_loss_pct"], r["trades"]
        w = r["win_rate"]
        gross_w += aw * w * n
        gross_l += abs(al) * (1 - w) * n
    combined_pf = (gross_w / gross_l) if gross_l > 0 else 0
    print("  " + "-" * 50)
    print(f"  COMBINED: {all_trades} simulated trades across {len(rows)} symbols")
    print(f"  Combined profit factor: {combined_pf:.2f}  "
          f"({'EDGE — worth pursuing' if combined_pf >= 1.2 else 'BREAK-EVEN/LOSING — needs work' if combined_pf >= 0.95 else 'LOSING — do not risk real money'})")
    print("\n  Reminder: backtests flatter the strategy (perfect fills, no")
    print("  earnings gaps). Clear PF ~1.2 HERE before trusting live results.\n")


if __name__ == "__main__":
    main()
