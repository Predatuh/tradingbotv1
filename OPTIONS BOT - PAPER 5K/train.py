#!/usr/bin/env python3
"""Train the ML brain on real market history — with an honest verdict.

    python train.py                          # default symbol set, 2 years
    python train.py SPY QQQ NVDA BTC/USD --days 730

What it does:
  1. Pulls historical bars for each symbol from Alpaca (your keys, your machine)
  2. Builds ~19 features per bar and labels each bar with what happened next
  3. WALK-FORWARD validates: 5 folds, each tested only on future data it
     never trained on, with an embargo gap so labels can't leak
  4. Tells you plainly whether the model found a real edge:
        USE       -> saves model.pkl; the bot will use it (strategy.use_ml: auto)
        MARGINAL  -> saves it, but keep expectations low
        SKIP      -> does NOT save; the hand-tuned ensemble stays in charge
  5. Retrain whenever you like (weekly is plenty); the bot picks up the new
     model on restart.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from bot.backtest import alpaca_bars
from bot.config import Config
from bot.features import training_rows
from bot.ml import train_final, walk_forward

DEFAULT_SYMBOLS = ["SPY", "QQQ", "NVDA", "TSLA", "AMD", "META",
                   "GLD", "TLT", "EEM",
                   "BTC/USD", "ETH/USD", "DOGE/USD", "LTC/USD"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("symbols", nargs="*", default=None)
    ap.add_argument("--days", type=int, default=730)
    ap.add_argument("--threshold", type=float, default=0.58,
                    help="probability the model must reach before the bot buys")
    args = ap.parse_args()
    symbols = args.symbols or DEFAULT_SYMBOLS

    cfg = Config.load()
    cfg.validate_keys()

    print(f"Fetching {args.days} days of {cfg.timeframe_minutes}m bars "
          f"for {len(symbols)} symbols…")
    all_rows = []
    for sym in symbols:
        try:
            df = alpaca_bars(cfg, sym, args.days)
        except Exception as e:
            print(f"  ! {sym}: {e} — skipping")
            continue
        if df.empty or len(df) < 500:
            print(f"  ! {sym}: not enough data ({len(df)} bars) — skipping")
            continue
        rows = training_rows(df, sym)
        all_rows.append(rows)
        print(f"  {sym}: {len(df)} bars -> {len(rows)} training rows")

    if not all_rows:
        raise SystemExit("No data — check your API keys and symbols.")
    rows = pd.concat(all_rows).sort_index(kind="stable")
    up = rows["y"].mean()
    print(f"\nTotal: {len(rows):,} rows | base up-rate {up:.1%} "
          f"(the coin the model must beat)")

    print("\nWalk-forward validation (5 folds, embargoed)…")
    report = walk_forward(rows, threshold=args.threshold)
    print(f"\n  {'fold':>4} {'AUC':>6} {'model hit-rate':>15} {'base rate':>10} "
          f"{'edge':>7} {'avg ret (liked)':>16}")
    for f in report.folds:
        print(f"  {f.fold:>4} {f.auc:>6.3f} {f.hit_rate_at_thr:>14.1%} "
              f"{f.base_up_rate:>10.1%} {f.edge():>+6.1%} "
              f"{f.avg_fwd_ret_at_thr:>15.3%}")
    print(f"\n  mean AUC {report.mean_auc():.3f} | mean edge {report.mean_edge():+.1%} "
          f"| verdict: {report.verdict()}")

    v = report.verdict()
    if v == "SKIP":
        print("\n  NOT SAVING a model. Honest answer: no reliable edge was found")
        print("  out-of-sample in this period. The hand-tuned ensemble stays in")
        print("  charge. Try more history (--days 1095), different symbols, or")
        print("  simply keep collecting paper results — a no is worth knowing.")
        sys.exit(0)

    print("\n  Training final model on all data…")
    _, meta = train_final(rows, meta_extra={
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "walk_forward": {"mean_auc": report.mean_auc(),
                         "mean_edge": report.mean_edge(),
                         "verdict": v,
                         "threshold": args.threshold},
        "timeframe_minutes": cfg.timeframe_minutes,
        "days": args.days,
    })
    print("  Saved model.pkl + model_meta.json")
    print("\n  What the model pays attention to (top 8):")
    for name, w in meta["importance"][:8]:
        print(f"    {name:>14}: {w}")
    if v == "MARGINAL":
        print("\n  Verdict was MARGINAL — the bot will use it, but watch the paper")
        print("  results skeptically and retrain with more data when you can.")
    print("\n  Restart the bot (python run_live.py) and it will trade with the model.")
    print("  To compare against the old brain: python run_backtest.py SPY 60")


if __name__ == "__main__":
    main()
