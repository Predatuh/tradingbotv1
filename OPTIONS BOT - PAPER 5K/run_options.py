#!/usr/bin/env python3
"""Start the OPTIONS trading bot + dashboard.

    python run_options.py

PAPER mode (config.yaml mode: paper): fake money, as always.
LIVE mode  (config.yaml mode: live):  REAL MONEY. Requires live-account keys
in .env, options approval on the live Alpaca account, and a typed
confirmation at startup. PDT guard is active for accounts under $25k.
"""
import logging
import threading

from bot.config import Config
from bot.dashboard import run as run_dashboard
from bot.options_broker import OptionsBroker
from bot.options_engine import OptionsEngine

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
                    datefmt="%H:%M:%S")


def main():
    cfg = Config.load()
    cfg.validate_keys()

    broker = OptionsBroker(cfg.api_key, cfg.api_secret, paper=cfg.paper)
    account = broker.account()
    o = cfg.raw.get("options", {})

    if cfg.paper:
        print(f"\n  Connected to Alpaca (PAPER — fake money) — OPTIONS BOT")
    else:
        print("\n  " + "!" * 62)
        print("  !!  LIVE MODE — THIS BOT WILL TRADE YOUR REAL MONEY  !!")
        print("  " + "!" * 62)
        print(f"  Live equity: ${account['equity']:,.2f}")
        cap = account['equity'] * float(o.get('max_premium_pct', 0.08))
        print(f"  Premium cap per trade: ${cap:,.0f} "
              f"({float(o.get('max_premium_pct', 0.08)):.0%} of equity)")
        print("  PDT guard: ACTIVE (blocks the 4th same-day round trip under $25k)")
        print("  Requirements: live keys in .env AND options approval on the live account.")
        # launcher-provided confirmation lets a crashed LIVE bot auto-restart
        # instead of sitting dead at this prompt (the MARA lesson)
        import os
        if os.environ.get("CONFIRM_LIVE_TRADING", "").strip() == "I understand":
            print("\n  Live-trading confirmation provided by the launcher — starting.")
        else:
            confirm = input("\n  Type 'I understand' to trade real money, anything else aborts: ")
            if confirm.strip() != "I understand":
                raise SystemExit("  Aborted. Nothing was traded.")

    print(f"  Equity: ${account['equity']:,.2f}")
    print(f"  Underlyings: {', '.join(o.get('underlyings', []))}")

    engine = OptionsEngine(cfg, broker)
    dash = cfg.dashboard
    threading.Thread(target=run_dashboard,
                     kwargs={"host": dash.get("host", "127.0.0.1"),
                             "port": int(dash.get("port", 8052)),
                             "engine": engine},   # enables Sell buttons + slider
                     daemon=True).start()
    print(f"  Dashboard: http://{dash.get('host','127.0.0.1')}:{dash.get('port',8052)}\n")

    engine.run_forever()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n  Stopped. Open option positions remain in the account.")
