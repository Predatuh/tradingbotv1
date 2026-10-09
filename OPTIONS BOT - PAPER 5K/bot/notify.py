"""Phone push notifications via ntfy.sh — free, instant, no account.

Setup (2 minutes):
  1. Install the "ntfy" app on your phone (Play Store / App Store).
  2. In the app: Subscribe to topic -> enter your secret topic name.
  3. Add one line to config.yaml (top level):  ntfy_topic: your-secret-topic
  4. Restart the bot. Test from PowerShell:
       curl.exe -d "test ping" -H "Title: Bot test" https://ntfy.sh/your-secret-topic

The topic name is the only password — make it long and random, and anyone
who knows it can read your alerts. Never use a guessable one like "trading".
"""
from __future__ import annotations

import logging
import threading

log = logging.getLogger("notify")


def push(topic: str, title: str, message: str,
         priority: str = "high", tags: str = "rotating_light") -> None:
    """Fire-and-forget push. Never blocks or crashes the trading loop."""
    if not topic:
        return

    def _hdr(s: str) -> str:
        # HTTP headers must be latin-1: strip emoji/smart-punctuation so the
        # push never crashes. The emoji live in the BODY (UTF-8) instead.
        s = (s.replace("—", "-").replace("–", "-")
              .replace("’", "'").replace("“", '"').replace("”", '"'))
        return s.encode("ascii", "ignore").decode("ascii").strip() or "Trading Bot"

    def _send():
        try:
            import requests
            requests.post(
                f"https://ntfy.sh/{topic.strip()}",
                data=message.encode("utf-8"),          # body carries emoji fine
                headers={"Title": _hdr(title), "Priority": priority, "Tags": tags},
                timeout=6,
            )
        except Exception as e:               # no internet? bot keeps trading
            log.warning("phone push failed: %s", e)

    threading.Thread(target=_send, daemon=True).start()
