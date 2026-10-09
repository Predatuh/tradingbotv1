#!/usr/bin/env python3
"""Backtest the strategy before letting it loose (even on paper).

    python run_backtest.py                       # synthetic data smoke test
    python run_backtest.py SPY 30                # SPY, last 30 days (needs API keys)
    python run_backtest.py BTC/USD 60            # crypto, last 60 days
    python run_backtest.py --csv mydata.csv      # your own OHLCV csv
"""
import json
import sys

import pandas as pd

from bot.backtest import alpaca_bars, run_backtest, synthetic_bars
from bot.config import Config


def main():
    cfg = Config.load()
    args = sys.argv[1:]

    if args and args[0] == "--csv":
        df = pd.read_csv(args[1], parse_dates=["timestamp"], index_col="timestamp")
        symbol = args[1]
    elif args:
        symbol = args[0]
        days = int(args[1]) if len(args) > 1 else 30
        cfg.validate_keys()
        print(f"Fetching {days} days of {cfg.timeframe_minutes}m bars for {symbol} from Alpaca…")
        df = alpaca_bars(cfg, symbol, days)
    else:
        symbol = "SYNTHETIC"
        print("No symbol given — running synthetic-data smoke test.")
        df = synthetic_bars(3000)

    if df.empty:
        raise SystemExit("No data returned.")

    print(f"Backtesting {symbol}: {len(df)} bars "
          f"({df.index[0]} → {df.index[-1]})")
    result = run_backtest(cfg, df, symbol=symbol)
    summary = result.summary()

    print("\n===== RESULTS =====")
    for k, v in summary.items():
        print(f"  {k:>18}: {v}")

    with open("backtest_result.json", "w") as f:
        json.dump({"summary": summary, "trades": result.trades,
                   "equity_curve": result.equity_curve[-500:]}, f, indent=1)
    print("\nFull detail saved to backtest_result.json")
    if summary["trades"] < 5:
        print("NOTE: very few trades — consider a longer period or lower entry_threshold.")


if __name__ == "__main__":
    main()
