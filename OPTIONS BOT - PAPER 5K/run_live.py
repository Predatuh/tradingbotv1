#!/usr/bin/env python3
"""Start the stocks+crypto trading bot + dashboard.

    python run_live.py            # start (asks your risk level on first run)
    python run_live.py --risk     # re-open the aggressiveness chooser

PAPER mode (config.yaml mode: paper): fake money.
LIVE mode  (config.yaml mode: live):  REAL MONEY. Requires live-account keys
in .env and a typed confirmation at startup. PDT guard is active for
accounts under $25k. Crypto trades 24/7; stocks during market hours.
"""
import logging
import sys
import threading

from bot.broker import AlpacaBroker
from bot.config import Config
from bot.dashboard import run as run_dashboard
from bot.engine import TradingEngine
from risk_profiles import apply_profile, load_or_choose

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)


def main():
    cfg = Config.load()
    cfg.validate_keys()

    # ---- aggressiveness slider ----
    profile = load_or_choose(force_choose="--risk" in sys.argv)
    apply_profile(cfg, profile)

    broker = AlpacaBroker(cfg.api_key, cfg.api_secret, paper=cfg.paper)
    account = broker.account()
    r = cfg.risk

    if cfg.paper:
        print(f"\n  Connected to Alpaca (PAPER — fake money)")
    else:
        print("\n  " + "!" * 62)
        print("  !!  LIVE MODE — THIS BOT WILL TRADE YOUR REAL MONEY  !!")
        print("  " + "!" * 62)
        print(f"  Live equity: ${account['equity']:,.2f}")
        print(f"  Risk profile: {profile['name']} — "
              f"{r['equity_risk_per_trade']:.1%} risked per trade, day pauses at "
              f"-{r['max_daily_loss_pct']:.0%}, kill switch at -{r['max_drawdown_pct']:.0%}")
        print("  PDT guard: ACTIVE (blocks the 4th same-day round trip under $25k)")
        print("  Crypto positions trade 24/7 — including while you sleep.")
        # launcher-provided confirmation lets a crashed LIVE bot auto-restart
        # instead of sitting dead at this prompt (the MARA lesson)
        import os
        if os.environ.get("CONFIRM_LIVE_TRADING", "").strip() == "I understand":
            print("\n  Live-trading confirmation provided by the launcher — starting.")
        else:
            confirm = input("\n  Type 'I understand' to trade real money, anything else aborts: ")
            if confirm.strip() != "I understand":
                raise SystemExit("  Aborted. Nothing was traded.")

    print(f"  Equity: ${account['equity']:,.2f}   Cash: ${account['cash']:,.2f}")
    print(f"  Risk profile: {profile['name']}")
    print(f"  Watching: {', '.join(cfg.all_symbols)}")

    engine = TradingEngine(cfg, broker)
    dash = cfg.dashboard
    t = threading.Thread(target=run_dashboard,
                         kwargs={"host": dash.get("host", "127.0.0.1"),
                                 "port": int(dash.get("port", 8050)),
                                 "engine": engine},   # enables the Sell button
                         daemon=True)
    t.start()
    print(f"  Dashboard: http://{dash.get('host','127.0.0.1')}:{dash.get('port',8050)}\n")

    try:
        engine.run_forever()
    except KeyboardInterrupt:
        print("\n  Stopped. Open positions remain in the account.")


if __name__ == "__main__":
    main()
