#!/usr/bin/env python3
"""OUT-OF-SAMPLE test: is the 'large caps win, small caps lose' pattern real,
or was it luck in one 30-day window?

    python oos_test.py

Fetches ~75 days of real data, splits it into TWO non-overlapping windows
(older half + recent half), and backtests each independently. The groups
below are fixed by LIQUIDITY up front — NOT by who won — so this can't
cheat. If the large-cap group beats the small-cap group in BOTH windows,
the edge is likely real and restricting the bot to large caps is justified.
If the ranking flips between windows, it was noise and we stop here.
"""
import sys

from bot.backtest import alpaca_bars, run_backtest
from bot.config import Config

# defined a-priori by liquidity / cleanliness, before seeing any result
LARGE = ["SPY", "QQQ", "MSFT", "META", "AMZN", "GOOGL", "NVDA", "AAPL", "TSLA"]
SMALL = ["IWM", "PLTR", "COIN", "INTC", "SOFI", "F", "HOOD", "MARA", "RIVN", "SNAP"]


def pf_of(summ):
    aw, al, n, w = (summ["avg_win_pct"], summ["avg_loss_pct"],
                    summ["trades"], summ["win_rate"])
    gw = aw * w * n
    gl = abs(al) * (1 - w) * n
    return (gw, gl, n)


def main():
    cfg = Config.load()
    cfg.validate_keys()
    print("\n  Fetching ~75 days and splitting into two clean windows...\n")
    print(f"  {'SYMBOL':8} {'OLDER PF':>9} {'RECENT PF':>10}   persistent?")
    print("  " + "-" * 48)

    group_tot = {g: {"old": [0.0, 0.0, 0], "recent": [0.0, 0.0, 0]}
                 for g in ("LARGE", "SMALL")}

    for group, syms in (("LARGE", LARGE), ("SMALL", SMALL)):
        for sym in syms:
            try:
                df = alpaca_bars(cfg, sym, 75)
                if df is None or len(df) < 200:
                    print(f"  {sym:8} {'(not enough data)':>20}")
                    continue
                mid = len(df) // 2
                old_df, rec_df = df.iloc[:mid], df.iloc[mid:]
                o = run_backtest(cfg, old_df, symbol=sym).summary()
                r = run_backtest(cfg, rec_df, symbol=sym).summary()
                opf = o["profit_factor"] or 0
                rpf = r["profit_factor"] or 0
                mark = "yes" if (opf >= 1.1 and rpf >= 1.1) else \
                       ("no (flipped)" if (opf >= 1.1) != (rpf >= 1.1) else "")
                print(f"  {sym:8} {opf:>9.2f} {rpf:>10.2f}   {mark}")
                for win, s in (("old", o), ("recent", r)):
                    gw, gl, n = pf_of(s)
                    group_tot[group][win][0] += gw
                    group_tot[group][win][1] += gl
                    group_tot[group][win][2] += n
            except Exception as e:
                print(f"  {sym:8} error: {str(e)[:34]}")

    print("\n  ===== GROUP VERDICT (the actual test) =====")
    for g in ("LARGE", "SMALL"):
        opf = (group_tot[g]["old"][0] / group_tot[g]["old"][1]
               if group_tot[g]["old"][1] else 0)
        rpf = (group_tot[g]["recent"][0] / group_tot[g]["recent"][1]
               if group_tot[g]["recent"][1] else 0)
        print(f"  {g:6} caps:  OLDER window PF {opf:.2f}   |   "
              f"RECENT window PF {rpf:.2f}   "
              f"({group_tot[g]['old'][2]}+{group_tot[g]['recent'][2]} trades)")

    lg = group_tot["LARGE"]
    lo = lg["old"][0] / lg["old"][1] if lg["old"][1] else 0
    lr = lg["recent"][0] / lg["recent"][1] if lg["recent"][1] else 0
    print("\n  READ IT:")
    if lo >= 1.2 and lr >= 1.2:
        print("  Large caps cleared 1.2 in BOTH independent windows -> the edge")
        print("  looks REAL. Restricting the bot to large caps is justified and")
        print("  worth a fresh paper run. (Still verify live; backtests flatter.)")
    elif lo >= 1.05 and lr >= 1.05:
        print("  Large caps are positive in both windows but thin (under 1.2).")
        print("  A weak, possibly-real edge — not enough to trust real money,")
        print("  but worth one more window before deciding.")
    else:
        print("  The large-cap edge did NOT hold across both windows. That means")
        print("  the earlier result was noise. Honest conclusion: this strategy")
        print("  has no durable edge — do not risk real money on it.")
    print()


if __name__ == "__main__":
    main()
