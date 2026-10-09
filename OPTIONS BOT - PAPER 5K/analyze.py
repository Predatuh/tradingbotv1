#!/usr/bin/env python3
"""Performance analyst: reads the trade log and tells you the truth.

    python analyze.py              # analyze all logged trades
    python analyze.py --days 7     # just the last week

This is the feedback loop that makes the bot smarter over time: it shows
where money is actually made and lost — per market, per hour, per exit
reason — and gives concrete suggestions (symbols to cut, patterns to watch).
Run it weekly. Trust it over your feelings about any individual trade.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TRADES_FILE = ROOT / "trades_log.jsonl"


def load_trades(days: int | None) -> list[dict]:
    if not TRADES_FILE.exists():
        raise SystemExit("No trades_log.jsonl yet — let the bot trade first.")
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)) if days else None
    out = []
    for line in TRADES_FILE.read_text().splitlines():
        try:
            t = json.loads(line)
        except json.JSONDecodeError:
            continue
        if cutoff and datetime.fromisoformat(t["ts"]) < cutoff:
            continue
        out.append(t)
    return out


def pair_trades(records: list[dict]) -> list[dict]:
    """Match each exit with its entry so we can attribute context to results."""
    open_entries: dict[str, dict] = {}
    completed = []
    for r in records:
        if r["action"] == "enter":
            open_entries[r["symbol"]] = r
        elif r["action"] == "exit":
            e = open_entries.pop(r["symbol"], None)
            completed.append({
                "symbol": r["symbol"],
                "pnl_pct": r.get("pnl_pct", 0.0),
                "reason": r.get("reason", "?"),
                "entry_ts": e["ts"] if e else None,
                "exit_ts": r["ts"],
                "score": e.get("score") if e else None,
                "signal_parts": e.get("signal_parts", {}) if e else {},
            })
    return completed


def bucket_stats(trades: list[dict], key_fn) -> dict:
    buckets = defaultdict(list)
    for t in trades:
        buckets[key_fn(t)].append(t["pnl_pct"])
    out = {}
    for k, pnls in buckets.items():
        wins = [p for p in pnls if p > 0]
        gross_w = sum(wins)
        gross_l = abs(sum(p for p in pnls if p <= 0))
        out[k] = {
            "trades": len(pnls),
            "total_pnl_pct": round(sum(pnls) * 100, 2),
            "win_rate": round(len(wins) / len(pnls), 2),
            "profit_factor": round(gross_w / gross_l, 2) if gross_l > 0 else float("inf"),
        }
    return dict(sorted(out.items(), key=lambda kv: -kv[1]["total_pnl_pct"]))


def fmt_table(stats: dict, label: str) -> None:
    print(f"\n  {label:<22} {'trades':>6} {'total P&L':>10} {'win%':>6} {'PF':>6}")
    for k, s in stats.items():
        pf = "inf" if s["profit_factor"] == float("inf") else f"{s['profit_factor']:.2f}"
        print(f"  {str(k):<22} {s['trades']:>6} {s['total_pnl_pct']:>+9.2f}% "
              f"{s['win_rate']*100:>5.0f}% {pf:>6}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=None)
    args = ap.parse_args()

    records = load_trades(args.days)
    trades = pair_trades(records)
    if not trades:
        raise SystemExit("No completed round-trip trades in this window yet.")

    pnls = [t["pnl_pct"] for t in trades]
    wins = [p for p in pnls if p > 0]
    gross_w, gross_l = sum(wins), abs(sum(p for p in pnls if p <= 0))
    window = f"last {args.days} days" if args.days else "all time"

    print(f"===== TRADING REPORT ({window}) =====")
    print(f"  completed trades : {len(trades)}")
    print(f"  total P&L        : {sum(pnls)*100:+.2f}% (sum of per-trade returns)")
    print(f"  win rate         : {len(wins)/len(trades):.0%}")
    print(f"  avg win / loss   : {(sum(wins)/len(wins)*100 if wins else 0):+.2f}% / "
          f"{(-gross_l/max(len(pnls)-len(wins),1)*100):+.2f}%")
    print(f"  profit factor    : {gross_w/gross_l:.2f}" if gross_l else "  profit factor    : inf")

    fmt_table(bucket_stats(trades, lambda t: t["symbol"]), "BY SYMBOL")
    fmt_table(bucket_stats(trades, lambda t: t["reason"]), "BY EXIT REASON")
    fmt_table(bucket_stats(
        trades, lambda t: "crypto" if "/" in t["symbol"] or t["symbol"].endswith("USD")
        else "stocks/ETFs"), "BY MARKET TYPE")
    fmt_table(bucket_stats(
        trades, lambda t: f"{datetime.fromisoformat(t['exit_ts']).hour:02d}:00 UTC"
        if t.get("exit_ts") else "?"), "BY HOUR (exit)")

    ml_trades = [t for t in trades if "ml_prob" in (t.get("signal_parts") or {})]
    if ml_trades:
        fmt_table(bucket_stats(
            ml_trades, lambda t: "high conf (p>=0.62)"
            if t["signal_parts"]["ml_prob"] >= 0.62 else "low conf (p<0.62)"),
            "ML CONFIDENCE")

    # ---- plain-language suggestions ----
    print("\n===== SUGGESTIONS =====")
    sym_stats = bucket_stats(trades, lambda t: t["symbol"])
    n_suggest = 0
    for sym, s in sym_stats.items():
        if s["trades"] >= 5 and s["profit_factor"] < 0.6:
            print(f"  - {sym}: {s['trades']} trades, PF {s['profit_factor']} — "
                  f"consistently losing; consider removing it from config.yaml")
            n_suggest += 1
    reason_stats = bucket_stats(trades, lambda t: t["reason"])
    sl = reason_stats.get("stop-loss")
    if sl and sl["trades"] / len(trades) > 0.6:
        print(f"  - {sl['trades']}/{len(trades)} exits are stop-losses — stops may "
              f"still be too tight; consider stop_loss_atr_mult 3.5 -> 4.0")
        n_suggest += 1
    if len(trades) < 30:
        print(f"  - only {len(trades)} trades so far — treat every number above as "
              f"preliminary; statistical noise dominates below ~30 trades")
        n_suggest += 1
    if not n_suggest:
        print("  - nothing jumps out; keep collecting data and re-run weekly")


if __name__ == "__main__":
    main()
