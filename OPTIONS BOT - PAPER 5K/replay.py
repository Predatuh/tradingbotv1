#!/usr/bin/env python3
"""Session replay — what the bot was thinking, minute by minute, and the
intraday arc of realized P&L so you can SEE the "up then flat" pattern.

    python replay.py                 # today
    python replay.py --date 2026-08-19
    python replay.py --days 3        # last 3 days combined

Reads trades_log.jsonl (every entry/exit with its score, ML confidence, and
exit reason) plus the live book's price history, and prints:
  * a timeline of every decision with the reasoning behind it
  * the running realized P&L through the day (the arc)
  * currently-open positions and where they stand vs. their peak
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TRADES = ROOT / "trades_log.jsonl"
BOOK = ROOT / "options_book.json"


def load(days, date_str):
    if not TRADES.exists():
        raise SystemExit("No trades_log.jsonl yet.")
    rows = []
    for line in TRADES.read_text().splitlines():
        try:
            r = json.loads(line)
            r["_t"] = datetime.fromisoformat(r["ts"])
            rows.append(r)
        except Exception:
            continue
    if date_str:
        rows = [r for r in rows if r["ts"][:10] == date_str]
    elif days:
        cut = datetime.now(timezone.utc) - timedelta(days=days)
        rows = [r for r in rows if r["_t"] >= cut]
    else:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        rows = [r for r in rows if r["ts"][:10] == today]
    return sorted(rows, key=lambda r: r["_t"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date"); ap.add_argument("--days", type=int)
    a = ap.parse_args()
    rows = load(a.days, a.date)
    if not rows:
        raise SystemExit("No trades in that window. Try --days 7.")

    print("\n===== SESSION REPLAY =====")
    realized = 0.0            # cumulative % (sum of per-trade returns)
    open_cost = {}            # symbol -> entry premium (for context)
    for r in rows:
        t = r["_t"].strftime("%H:%M")
        sym = r["symbol"]
        und = r.get("underlying", sym[:4])
        if r["action"] == "enter":
            sc = r.get("score")
            ml = (r.get("signal_parts") or {}).get("ml_prob")
            open_cost[sym] = r.get("price", 0)
            conf = f", ML {ml:.0%}" if ml is not None else ""
            adv = "  [advice/manual]" if r.get("advice") else ""
            print(f"  {t}  BUY  {und:5} {r.get('side','?'):4} @ ${r.get('price',0):.2f}"
                  f"   (score {sc:+.2f}{conf}){adv}")
        else:
            pnl = r.get("pnl_pct", 0) * 100
            realized += pnl
            arrow = "WIN " if pnl >= 0 else "LOSS"
            print(f"  {t}  SELL {und:5}      @ ${r.get('price',0):.2f}"
                  f"   {arrow} {pnl:+.0f}%   [{r.get('reason','?')}]"
                  f"   -> day realized {realized:+.0f}%")

    print("\n  ---- day realized total (sum of per-trade %): "
          f"{realized:+.0f}% ----")

    # open positions right now
    try:
        book = json.loads(BOOK.read_text())
    except Exception:
        book = {}
    if book:
        print("\n  STILL OPEN (unrealized — not yet counted above):")
        for sym, b in book.items():
            entry = b.get("entry_premium", 0)
            hist = b.get("hist") or []
            now = hist[-1] if hist else entry
            peak = b.get("peak", entry)
            pnl = (now - entry) / entry * 100 if entry else 0
            pk = (peak - entry) / entry * 100 if entry else 0
            armed = "ratchet ARMED" if pk >= 15 else "ratchet not armed yet"
            print(f"    {b.get('underlying','?'):5} {b.get('type','?'):4} "
                  f"entry ${entry:.2f} -> now ${now:.2f} ({pnl:+.0f}%), "
                  f"peaked {pk:+.0f}% [{armed}]")

    print("\n  Read the arc above: if the day climbs then returns toward 0,")
    print("  winners and losers are ~balanced (break-even edge). That's the")
    print("  profit-factor problem — not a broken exit. Watch it over 30 trades.\n")


if __name__ == "__main__":
    main()
