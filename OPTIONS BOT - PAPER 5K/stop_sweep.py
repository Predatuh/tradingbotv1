#!/usr/bin/env python3
"""Does cutting losers faster actually help? Backtest several stop widths on
real history and let the numbers decide — no guessing.

    python stop_sweep.py            # ~60 days, this bot's symbols

For each stop width it runs the full strategy over real data and reports the
combined profit factor. If a TIGHTER stop lifts PF, cutting losses faster is a
real win. If PF drops, the tight stop is shaking out winners and it's a trap.
This tests the STOCK signal (the part that's rigorously backtestable); the
options premium version is then forward-tested on paper.
"""
import sys, copy
from bot.backtest import alpaca_bars, run_backtest
from bot.config import Config

def main():
    cfg = Config.load(); cfg.validate_keys()
    days = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 60
    syms = list(cfg.raw.get("symbols", {}).get("stocks", []))
    syms += [s for s in cfg.raw.get("options", {}).get("underlyings", []) if s not in syms]
    # pre-fetch once per symbol (data is the slow part)
    print(f"\n  Fetching {len(syms)} symbols, {days} days...")
    data = {}
    for s in syms:
        try:
            df = alpaca_bars(cfg, s, days)
            if df is not None and len(df) >= 120: data[s] = df
        except Exception: pass
    print(f"  {len(data)} symbols with enough data.\n")

    print(f"  {'STOP (xATR)':12}{'trades':>8}{'win%':>7}{'PF':>8}   loss size")
    print("  " + "-"*46)
    base = float(cfg.raw.get("risk", {}).get("stop_loss_atr_mult", 3.5))
    for mult in [1.5, 2.0, 2.5, 3.0, 3.5, 4.5]:
        gw = gl = n = 0; losses = []
        for s, df in data.items():
            c2 = copy.deepcopy(cfg.raw); c2.setdefault("risk", {})["stop_loss_atr_mult"] = mult
            r = run_backtest(Config(raw=c2), df, symbol=s).summary()
            aw, al, w, tn = r["avg_win_pct"], r["avg_loss_pct"], r["win_rate"], r["trades"]
            gw += aw*w*tn; gl += abs(al)*(1-w)*tn; n += tn
            if al: losses.append(al)
        pf = gw/gl if gl else 0
        avgloss = sum(losses)/len(losses) if losses else 0
        star = "  <- current" if abs(mult-base) < 0.01 else ("  <- tighter" if mult < base else "")
        print(f"  {mult:<12.1f}{n:>8}{(gw/(gw+gl)*100 if (gw+gl) else 0):>6.0f}%{pf:>8.2f}   {avgloss:>6.1f}%{star}")
    print("\n  Read the PF column: if a tighter stop (lower xATR) shows a HIGHER")
    print("  PF than current, cutting losers faster genuinely helps. If PF falls,")
    print("  the tight stop is killing winners early — don't do it.\n")

if __name__ == "__main__":
    main()
