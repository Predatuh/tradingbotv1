#!/usr/bin/env python3
"""Walk-forward parameter tuner — finds settings that work on REAL data.

    python tune.py SPY QQQ NVDA BTC/USD ETH/USD --days 120

For every combination in the grid it backtests on the FIRST 70% of the data
(train) and then checks the winners on the LAST 30% it has never seen (test).
Only settings that hold up out-of-sample are trustworthy — anything else is
curve-fitting. The best config is printed and saved to tuned_config.yaml.

Runs on your machine using your Alpaca keys (needed for historical data).
A 5-symbol / 120-day run takes a few minutes.
"""
from __future__ import annotations

import argparse
import copy
import itertools
import sys

import numpy as np
import yaml

from bot.backtest import alpaca_bars, run_backtest
from bot.config import Config

# ---- the search grid (edit freely; it multiplies out fast) ----
GRID = {
    "timeframe_minutes": [15, 30],
    "entry_threshold": [0.40, 0.50, 0.60],
    "stop_loss_atr_mult": [2.5, 3.5],
    "take_profit_atr_mult": [5.0, 7.0],
    "exit_threshold": [-0.05, -1.0],   # -1.0 = never exit on signal, stops only
}


def apply(cfg: Config, combo: dict) -> Config:
    c = copy.deepcopy(cfg)
    c.raw["timeframe_minutes"] = combo["timeframe_minutes"]
    c.raw["strategy"]["entry_threshold"] = combo["entry_threshold"]
    c.raw["strategy"]["exit_threshold"] = combo["exit_threshold"]
    c.raw["risk"]["stop_loss_atr_mult"] = combo["stop_loss_atr_mult"]
    c.raw["risk"]["take_profit_atr_mult"] = combo["take_profit_atr_mult"]
    return c


def score(summaries: list[dict]) -> float:
    """Rank a combo across symbols: reward return, punish drawdown,
    require a real trade sample."""
    total_trades = sum(s["trades"] for s in summaries)
    if total_trades < 10:
        return -999.0
    ret = np.mean([s["total_return_pct"] for s in summaries])
    dd = np.mean([abs(s["max_drawdown_pct"]) for s in summaries])
    pf = np.mean([(s["profit_factor"] or 0) for s in summaries])
    return float(ret - 0.5 * dd + 2.0 * min(pf, 3.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("symbols", nargs="*", default=["SPY", "NVDA", "BTC/USD"])
    ap.add_argument("--days", type=int, default=120)
    args = ap.parse_args()
    symbols = args.symbols or ["SPY", "NVDA", "BTC/USD"]

    base = Config.load()
    base.validate_keys()

    # fetch each dataset once per timeframe
    print(f"Fetching {args.days} days of data for {', '.join(symbols)}…")
    data: dict[tuple, object] = {}
    for tf in GRID["timeframe_minutes"]:
        cfg_tf = copy.deepcopy(base)
        cfg_tf.raw["timeframe_minutes"] = tf
        for sym in symbols:
            df = alpaca_bars(cfg_tf, sym, args.days)
            if df.empty:
                print(f"  ! no data for {sym} @ {tf}m — skipping")
                continue
            split = int(len(df) * 0.7)
            data[(sym, tf)] = (df.iloc[:split], df.iloc[split:])
            print(f"  {sym} @ {tf}m: {split} train bars / {len(df)-split} test bars")

    combos = [dict(zip(GRID, v)) for v in itertools.product(*GRID.values())]
    print(f"\nSearching {len(combos)} combinations (train split)…")

    ranked = []
    for i, combo in enumerate(combos, 1):
        cfg = apply(base, combo)
        train_sums = []
        for sym in symbols:
            key = (sym, combo["timeframe_minutes"])
            if key not in data:
                continue
            train_df, _ = data[key]
            train_sums.append(run_backtest(cfg, train_df, symbol=sym).summary())
        if train_sums:
            ranked.append((score(train_sums), combo, train_sums))
        print(f"\r  {i}/{len(combos)}", end="", flush=True)

    ranked.sort(key=lambda x: -x[0])
    print("\n\n===== TOP 5 ON TRAIN DATA — now validating on unseen test data =====")
    final = []
    for train_score, combo, _ in ranked[:5]:
        cfg = apply(base, combo)
        test_sums = []
        for sym in symbols:
            key = (sym, combo["timeframe_minutes"])
            if key not in data:
                continue
            _, test_df = data[key]
            test_sums.append(run_backtest(cfg, test_df, symbol=sym).summary())
        t_score = score(test_sums) if test_sums else -999
        avg_ret = np.mean([s["total_return_pct"] for s in test_sums]) if test_sums else 0
        n_trades = sum(s["trades"] for s in test_sums)
        final.append((t_score, combo, test_sums))
        print(f"  {combo}  ->  test: avg return {avg_ret:+.2f}%  trades {n_trades}  score {t_score:.2f}")

    final.sort(key=lambda x: -x[0])
    best_score, best, best_sums = final[0]
    print("\n===== WINNER (best out-of-sample) =====")
    print(f"  {best}")
    for s in best_sums:
        print(f"    {s['symbol']:>8}: {s['trades']} trades, {s['total_return_pct']:+.2f}%, "
              f"dd {s['max_drawdown_pct']:.2f}%, pf {s['profit_factor']}")

    if best_score <= 0:
        print("\n  CAUTION: even the best combo is weak out-of-sample in this period.")
        print("  That's honest information — consider different symbols, a longer")
        print("  period, or keep paper trading the defaults while collecting data.")

    tuned = copy.deepcopy(base.raw)
    tuned["timeframe_minutes"] = best["timeframe_minutes"]
    tuned["strategy"]["entry_threshold"] = best["entry_threshold"]
    tuned["strategy"]["exit_threshold"] = best["exit_threshold"]
    tuned["risk"]["stop_loss_atr_mult"] = best["stop_loss_atr_mult"]
    tuned["risk"]["take_profit_atr_mult"] = best["take_profit_atr_mult"]
    with open("tuned_config.yaml", "w") as f:
        yaml.safe_dump(tuned, f, sort_keys=False)
    print("\nSaved to tuned_config.yaml — review it, then replace config.yaml with it")
    print("(or copy the winning values across) and restart the bot.")


if __name__ == "__main__":
    main()
