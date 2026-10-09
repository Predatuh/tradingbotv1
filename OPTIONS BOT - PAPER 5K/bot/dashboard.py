"""Local web dashboard. Serves the live state written by the engine.

Run alongside the engine (run_live.py starts both). Then open:
    http://127.0.0.1:8050
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from flask import Flask, jsonify, request, send_file

ROOT = Path(__file__).resolve().parent.parent
STATE_FILE = ROOT / "state.json"
TRADES_FILE = ROOT / "trades_log.jsonl"
PAGE = Path(__file__).resolve().parent / "dashboard.html"

app = Flask(__name__)
ENGINE = None      # set by run(); enables the dashboard's control buttons

# The browser polls /state every few seconds. Werkzeug logs EVERY request, which
# buries the trade events you actually need to see in this window. Silence the
# request log; real errors and tracebacks still print.
logging.getLogger("werkzeug").setLevel(logging.ERROR)


@app.route("/")
def index():
    # UI override order — edit ONE file instead of every bot's copy:
    #   1. dashboard.html in THIS bot's folder (per-bot custom look)
    #   2. SHARED-DASHBOARD.html in the PARENT folder (one skin for every
    #      bot folder inside it, e.g. Desktop\AI TRADING BOTS)
    #   3. the built-in bot/dashboard.html
    for p in (ROOT / "dashboard.html", ROOT.parent / "SHARED-DASHBOARD.html"):
        if p.exists():
            return send_file(p)
    return send_file(PAGE)


@app.route("/state")
def state():
    if not STATE_FILE.exists():
        return jsonify({"status": "waiting for first engine cycle"})
    return jsonify(json.loads(STATE_FILE.read_text()))


@app.route("/trades")
def trades():
    if not TRADES_FILE.exists():
        return jsonify([])
    lines = TRADES_FILE.read_text().strip().splitlines()[-200:]
    return jsonify([json.loads(l) for l in lines])


@app.route("/close", methods=["POST"])
def close_position():
    """Sell-now button: hand the symbol to the running engine."""
    if ENGINE is None:
        return jsonify({"ok": False, "error": "Controls not available (old launcher)."})
    if getattr(ENGINE, "observe_only", False):
        return jsonify({"ok": False, "error": "OBSERVE ONLY is on — this bot places no orders."})
    sym = (request.get_json(force=True).get("symbol") or "").strip()
    if not sym:
        return jsonify({"ok": False, "error": "No symbol given."})
    if not ENGINE.request_manual_close(sym):
        return jsonify({"ok": False, "error": f"{sym} isn't an open position."})
    return jsonify({"ok": True})


def _entry_from_log(symbol: str) -> dict:
    """Recover an open position's entry price/time from the trade log.

    Positions opened before the engine started recording entry markers (or
    before a restart) would otherwise show a blank chart. The trade log is
    permanent, so the entry is always recoverable from it.
    """
    if not TRADES_FILE.exists():
        return {}
    entry = {}
    try:
        for line in TRADES_FILE.read_text().strip().splitlines():
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("symbol") != symbol:
                continue
            if r.get("action") == "enter":
                entry = {"price": r.get("price"), "ts": r.get("ts"),
                         "und_px": r.get("und_price")}
            elif r.get("action") == "exit":
                entry = {}            # closed — a later entry starts fresh
    except Exception:
        return {}
    return entry


@app.route("/chart")
def chart():
    """Full OHLCV bars for the big candlestick chart (fetched on demand).

    ?symbol=SPY        -> the stock/underlying candles
    ?contract=OCC...   -> that option contract's premium history
    """
    if ENGINE is None:
        return jsonify({"error": "Controls not available (old launcher)."})
    contract = (request.args.get("contract") or "").strip()
    # WATCHING a symbol you don't own: chart the at-the-money contract the bot
    # is tracking for it. A live quote is taken on each request so the chart is
    # fresh while you're actually looking at it.
    watch = (request.args.get("watch") or "").strip()
    if watch and not contract:
        cand = (getattr(ENGINE, "candidates", {}) or {}).get(watch)
        if not cand:
            # price it RIGHT NOW instead of waiting for the round-robin
            try:
                sig = (getattr(ENGINE, "last_signals", {}) or {}).get(watch) or {}
                spot = sig.get("price")
                direction = "call" if (sig.get("regime", 0) or 0) >= 0 else "put"
                if spot and hasattr(ENGINE.broker, "peek_contract"):
                    got = ENGINE.broker.peek_contract(
                        watch, direction, float(spot),
                        getattr(ENGINE, "dte_min", 7), getattr(ENGINE, "dte_max", 21))
                    if got:
                        import time as _t
                        cand = dict(got); cand["picked_at"] = _t.time()
                        cand["cost"] = round(cand.get("mid", 0) * 100, 2)
                        ENGINE.candidates[watch] = cand
            except Exception:
                cand = None
        if not cand:
            return jsonify({"error": f"couldn't price a contract for {watch} "
                                     f"right now — try again in a few seconds"})
        # real premium candles from the market data API when available
        if hasattr(ENGINE.broker, "option_bars"):
            try:
                bars = ENGINE.broker.option_bars(cand["symbol"])
            except Exception:
                bars = None
            if bars and bars.get("c"):
                out = {"kind": "premium", "underlying": watch,
                       "type": cand.get("type"), "strike": cand.get("strike"),
                       "expiry": str(cand.get("expiry", "")), "entry": None,
                       "watch_only": True, "symbol": cand.get("symbol")}
                out.update(bars)
                try:
                    lt = out["t"][-1]
                    fresh = [(t2, p2) for t2, p2 in
                             zip(cand.get("hist_t") or [], cand.get("hist") or [])
                             if t2 > lt]
                    for t2, p2 in fresh[-120:]:
                        out["t"].append(t2); out["o"].append(p2)
                        out["h"].append(p2); out["l"].append(p2)
                        out["c"].append(p2); out["v"].append(0)
                except Exception:
                    pass
                out["p"] = out["c"]
                return jsonify(out)
        try:                                   # freshen while the chart is open
            q = ENGINE.broker._quote(cand["symbol"])
            if q and q[0] > 0 and q[1] > 0:
                mid = round((q[0] + q[1]) / 2, 4)
                h, ht = cand.setdefault("hist", []), cand.setdefault("hist_t", [])
                if not h or h[-1] != mid:
                    import time as _t
                    h.append(mid); ht.append(int(_t.time()))
                    del h[:-400]; del ht[:-400]
                cand["mid"] = mid
        except Exception:
            pass
        return jsonify({"kind": "premium", "p": cand.get("hist", []),
                        "t": cand.get("hist_t", []), "entry": None,
                        "underlying": watch, "type": cand.get("type"),
                        "strike": cand.get("strike"), "expiry": cand.get("expiry", ""),
                        "watch_only": True, "symbol": cand.get("symbol")})
    if contract:
        b = (getattr(ENGINE, "book", {}) or {}).get(contract) or {}
        out = {"kind": "premium", "p": b.get("phist", []), "t": b.get("phist_t", []),
               "entry": b.get("entry_premium"), "opened_at": b.get("opened_at"),
               "underlying": b.get("underlying"), "type": b.get("type"),
               "strike": b.get("strike"), "expiry": str(b.get("expiry", ""))}
        if not out["entry"]:                       # pre-restart position: recover it
            out.update({k: v for k, v in
                        (("entry", _entry_from_log(contract).get("price")),) if v})
        # richer history straight from the market data API when available
        if hasattr(ENGINE.broker, "option_bars"):
            try:
                bars = ENGINE.broker.option_bars(contract)
            except Exception:
                bars = None
            if bars and bars.get("c"):
                out.update(bars)
                # stitch the LIVE quote mids recorded every cycle past the last
                # (possibly lagging) bar, so the right edge of the chart is NOW
                try:
                    lt = out["t"][-1]
                    fresh = [(t2, p2) for t2, p2 in
                             zip(b.get("phist_t") or [], b.get("phist") or [])
                             if t2 > lt]
                    for t2, p2 in fresh[-120:]:
                        out["t"].append(t2); out["o"].append(p2)
                        out["h"].append(p2); out["l"].append(p2)
                        out["c"].append(p2); out["v"].append(0)
                except Exception:
                    pass
                out["p"] = out["c"]
        return jsonify(out)
    sym = (request.args.get("symbol") or "").strip()
    # tf: chart timeframe in minutes. The bot THINKS on 30-minute bars, but a
    # 30-min candle only advances twice an hour — which looks frozen when you're
    # watching a live trade. Finer timeframes are pulled on demand, for this one
    # symbol only, so they cost one extra API call instead of one per symbol
    # per cycle.
    try:
        tf = int(request.args.get("tf") or 0)
    except ValueError:
        tf = 0
    data = None
    if tf and tf != getattr(ENGINE.cfg, "timeframe_minutes", 30):
        try:
            df = ENGINE.broker.bars(sym, tf, 400)
            if not df.empty:
                data = {"t": [int(x.timestamp()) for x in df.index],
                        "o": [round(float(v), 4) for v in df["open"]],
                        "h": [round(float(v), 4) for v in df["high"]],
                        "l": [round(float(v), 4) for v in df["low"]],
                        "c": [round(float(v), 4) for v in df["close"]],
                        "v": [int(v) for v in df["volume"]], "tf": tf}
        except Exception as e:
            return jsonify({"error": f"{tf}m bars unavailable: {e}"})
    if data is None:
        bars = (getattr(ENGINE, "bars_cache", {}) or {})
        data = bars.get(sym)
        if data is None:                           # crypto slash mismatch etc.
            flat = sym.replace("/", "").upper()
            k = next((k for k in bars if k.replace("/", "").upper() == flat), None)
            data = bars.get(k) if k else None
    if data is None:
        return jsonify({"error": f"no bars for {sym} yet"})
    out = dict(data); out["kind"] = "bars"
    logged = _entry_from_log(sym)
    if logged.get("ts"):
        out["entry_ts"] = logged["ts"]
    return jsonify(out)


@app.route("/buy", methods=["POST"])
def buy_now():
    """Buy-now button: symbol (+ call/put direction for the options bot)."""
    if ENGINE is None or not hasattr(ENGINE, "request_manual_buy"):
        return jsonify({"ok": False, "error": "Controls not available (old launcher)."})
    if getattr(ENGINE, "observe_only", False):
        return jsonify({"ok": False, "error": "OBSERVE ONLY is on — this bot places no orders."})
    data = request.get_json(force=True)
    sym = (data.get("symbol") or "").strip()
    direction = (data.get("direction") or "").strip()
    if not sym:
        return jsonify({"ok": False, "error": "No symbol given."})
    res = ENGINE.request_manual_buy(sym, direction) if direction \
        else ENGINE.request_manual_buy(sym)
    ok, err = res if isinstance(res, tuple) else (bool(res), "")
    return jsonify({"ok": bool(ok), "error": err or None})


@app.route("/observe", methods=["POST"])
def set_observe():
    """OBSERVE ONLY toggle: run and reason, but place no orders at all."""
    if ENGINE is None or not hasattr(ENGINE, "apply_observe"):
        return jsonify({"ok": False, "error": "Controls not available (old launcher)."})
    on = bool(request.get_json(force=True).get("on"))
    return jsonify({"ok": True, "observe_only": ENGINE.apply_observe(on)})


@app.route("/strictness", methods=["POST"])
def set_strictness():
    """Slider: change the entry bar on the running engine (persists)."""
    if ENGINE is None or not hasattr(ENGINE, "apply_strictness"):
        return jsonify({"ok": False, "error": "Controls not available (old launcher)."})
    try:
        v = float(request.get_json(force=True).get("value"))
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "Value must be a number."})
    if not (0.05 <= v <= 0.95):
        return jsonify({"ok": False, "error": "Strictness out of range."})
    return jsonify({"ok": True, "value": ENGINE.apply_strictness(v)})


def run(host: str = "127.0.0.1", port: int = 8050, engine=None):
    global ENGINE
    ENGINE = engine
    app.run(host=host, port=port, debug=False, use_reloader=False)


if __name__ == "__main__":
    run()
