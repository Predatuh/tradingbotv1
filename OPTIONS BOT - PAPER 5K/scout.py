#!/usr/bin/env python3
"""Smart-money scout: scans public high-value sources and nudges the bot.

    python scout.py            # scan and write scout_boost.json
    python scout.py --show     # just print the current boosts

Sources (all legal, public, and actually fresh):
  1. SEC Form 4 insider filings — corporate officers/directors must report
     their own-company trades within 2 BUSINESS DAYS. Cluster buying by
     insiders is one of the best-documented bullish signals in finance.
  2. Market news via Alpaca's news feed — headline sentiment and unusual
     news volume for every symbol the bot watches (stocks AND crypto).

What it does NOT do (on purpose): congressional trades arrive up to 45 days
late by law and the free feeds for them are dead; social-media hype is mostly
exit liquidity for whoever started the rumor. If you want passive congress
exposure, the NANC ETF exists.

Output: scout_boost.json — a bounded nudge per symbol, capped at ±0.10 on the
strategy's [-1,+1] composite scale. The scout can tilt a close call; it can
never overrule the strategy or the risk manager on its own.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BOOST_FILE = ROOT / "scout_boost.json"

SEC_UA = {"User-Agent": "personal paper-trading research bot (contact: user@example.com)"}
INSIDER_LOOKBACK_DAYS = 14
NEWS_LOOKBACK_DAYS = 3
BOOST_CAP = 0.10

POS_WORDS = ("beat", "beats", "surge", "soar", "record", "upgrade", "upgraded",
             "buyback", "acquisition", "acquire", "partnership", "approval",
             "breakthrough", "outperform", "raise", "raises", "growth", "rally",
             "bullish", "all-time high", "expands", "wins", "contract")
NEG_WORDS = ("miss", "misses", "plunge", "sink", "downgrade", "downgraded",
             "lawsuit", "probe", "investigation", "layoff", "layoffs", "recall",
             "bankruptcy", "fraud", "warning", "cuts", "cut", "decline", "bearish",
             "sell-off", "crash", "halt", "default", "delist")


def _get_json(url: str, headers: dict | None = None, timeout: int = 20):
    req = urllib.request.Request(url, headers=headers or SEC_UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _get_text(url: str, timeout: int = 20) -> str:
    req = urllib.request.Request(url, headers=SEC_UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode(errors="ignore")


# ---------------------------------------------------------------- insiders
def cik_map(tickers: list[str]) -> dict[str, str]:
    data = _get_json("https://www.sec.gov/files/company_tickers.json")
    want = {t.upper() for t in tickers}
    out = {}
    for row in data.values():
        if row["ticker"].upper() in want:
            out[row["ticker"].upper()] = f"{row['cik_str']:010d}"
    return out


def _parse_form4(xml: str) -> list[dict]:
    """Minimal Form 4 parse: open-market transactions with code P (buy) or S (sell)."""
    import re
    txns = []
    for block in re.findall(r"<nonDerivativeTransaction>(.*?)</nonDerivativeTransaction>",
                            xml, re.S):
        code = re.search(r"<transactionCode>(\w)</transactionCode>", block)
        shares = re.search(r"<transactionShares>.*?<value>([\d.]+)</value>", block, re.S)
        price = re.search(r"<transactionPricePerShare>.*?<value>([\d.]+)</value>", block, re.S)
        if code and code.group(1) in ("P", "S") and shares:
            txns.append({
                "code": code.group(1),
                "shares": float(shares.group(1)),
                "price": float(price.group(1)) if price else 0.0,
            })
    return txns


def insider_activity(symbols: list[str], lookback_days: int = INSIDER_LOOKBACK_DAYS,
                     log=print) -> dict[str, dict]:
    """Per symbol: recent insider open-market buys/sells from SEC EDGAR."""
    result = {}
    try:
        ciks = cik_map(symbols)
    except Exception as e:
        log(f"  ! SEC ticker map failed: {e}")
        return result
    cutoff = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).date().isoformat()

    for sym, cik in ciks.items():
        try:
            sub = _get_json(f"https://data.sec.gov/submissions/CIK{cik}.json")
            recent = sub.get("filings", {}).get("recent", {})
            forms = recent.get("form", [])
            dates = recent.get("filingDate", [])
            accs = recent.get("accessionNumber", [])
            docs = recent.get("primaryDocument", [])
            buy_val = sell_val = 0.0
            buyers = 0
            n_checked = 0
            for form, date, acc, doc in zip(forms, dates, accs, docs):
                if form != "4" or date < cutoff or n_checked >= 8:
                    continue
                n_checked += 1
                acc_plain = acc.replace("-", "")
                raw_doc = doc.split("/")[-1]        # strip xsl-rendering prefix -> raw XML
                url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_plain}/{raw_doc}"
                try:
                    xml = _get_text(url)
                except Exception:
                    continue
                txns = _parse_form4(xml)
                b = sum(t["shares"] * t["price"] for t in txns if t["code"] == "P")
                s = sum(t["shares"] * t["price"] for t in txns if t["code"] == "S")
                buy_val += b
                sell_val += s
                if b > 0:
                    buyers += 1
                time.sleep(0.15)          # be polite to SEC servers
            if buy_val or sell_val:
                result[sym] = {"buy_usd": round(buy_val), "sell_usd": round(sell_val),
                               "distinct_buy_filings": buyers}
                log(f"  {sym}: insiders bought ${buy_val:,.0f} / sold ${sell_val:,.0f} "
                    f"({buyers} buy filings, last {lookback_days}d)")
        except Exception as e:
            log(f"  ! {sym}: {e}")
    return result


def insider_score(a: dict) -> float:
    """[-1, 1]: cluster buys strong positive; heavy one-sided selling mild negative
    (insiders sell for many innocent reasons; they buy for only one)."""
    buy, sell, buyers = a["buy_usd"], a["sell_usd"], a["distinct_buy_filings"]
    score = 0.0
    if buy > 50_000:
        score += min(buy / 2_000_000, 1.0) * (1.5 if buyers >= 2 else 1.0)
    if sell > 5_000_000 and buy == 0:
        score -= min(sell / 50_000_000, 0.4)
    return max(-1.0, min(1.0, score))


# ---------------------------------------------------------------- news
def news_activity(cfg, symbols: list[str], log=print) -> dict[str, dict]:
    """Headline sentiment + volume per symbol from Alpaca's news feed."""
    from alpaca.data.historical.news import NewsClient
    from alpaca.data.requests import NewsRequest
    client = NewsClient(cfg.api_key, cfg.api_secret)
    start = datetime.now(timezone.utc) - timedelta(days=NEWS_LOOKBACK_DAYS)
    out = {}
    for sym in symbols:
        api_sym = sym.replace("/", "")
        try:
            req = NewsRequest(symbols=api_sym, start=start, limit=50)
            arts = client.get_news(req).data.get("news", [])
        except Exception as e:
            log(f"  ! news {sym}: {e}")
            continue
        pos = neg = 0
        for art in arts:
            headline = (getattr(art, "headline", "") or "").lower()
            pos += sum(w in headline for w in POS_WORDS)
            neg += sum(w in headline for w in NEG_WORDS)
        if arts:
            out[sym] = {"articles": len(arts), "pos_hits": pos, "neg_hits": neg}
            log(f"  {sym}: {len(arts)} articles, sentiment hits +{pos}/-{neg}")
    return out


def news_score(n: dict) -> float:
    total = n["pos_hits"] + n["neg_hits"]
    if total == 0:
        return 0.0
    balance = (n["pos_hits"] - n["neg_hits"]) / total
    weight = min(total / 10.0, 1.0)
    return balance * weight


# ---------------------------------------------------------------- main
def run_scout(cfg, log=print) -> dict:
    stock_syms = cfg.stock_symbols
    all_syms = cfg.all_symbols

    log(f"Scout scanning {len(all_syms)} symbols…")
    log("Insider filings (SEC Form 4, ≤2 business days old):")
    insiders = insider_activity(stock_syms, log=log)
    log("News (last 3 days):")
    news = news_activity(cfg, all_syms, log=log)

    boosts = {}
    for sym in all_syms:
        i_s = insider_score(insiders[sym]) if sym in insiders else 0.0
        n_s = news_score(news[sym]) if sym in news else 0.0
        boost = max(-BOOST_CAP, min(BOOST_CAP, 0.06 * i_s + 0.04 * n_s))
        reasons = []
        if sym in insiders:
            a = insiders[sym]
            if a["buy_usd"]:
                reasons.append(f"insiders bought ${a['buy_usd']:,}")
            if a["sell_usd"] > 5_000_000:
                reasons.append(f"insiders sold ${a['sell_usd']:,}")
        if sym in news and (news[sym]["pos_hits"] or news[sym]["neg_hits"]):
            reasons.append(f"news +{news[sym]['pos_hits']}/-{news[sym]['neg_hits']}")
        if abs(boost) >= 0.005:
            boosts[sym] = {"boost": round(boost, 4), "reasons": reasons}

    payload = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "boosts": boosts,
        "raw": {"insiders": insiders, "news": news},
    }
    BOOST_FILE.write_text(json.dumps(payload, indent=1))
    log(f"\nBoosts written for {len(boosts)} symbols -> scout_boost.json")
    for sym, b in sorted(boosts.items(), key=lambda kv: -abs(kv[1]["boost"])):
        log(f"  {sym:>8}: {b['boost']:+.3f}  ({'; '.join(b['reasons'])})")
    if not boosts:
        log("  (nothing notable right now — that's a normal, honest result)")
    return payload


def load_boosts(max_age_hours: float = 48.0) -> dict[str, float]:
    """For the engine: symbol -> boost, empty if stale/missing."""
    try:
        data = json.loads(BOOST_FILE.read_text())
        age = datetime.now(timezone.utc) - datetime.fromisoformat(data["generated"])
        if age > timedelta(hours=max_age_hours):
            return {}
        return {s: b["boost"] for s, b in data.get("boosts", {}).items()}
    except Exception:
        return {}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()
    if args.show:
        print(json.dumps(load_boosts(), indent=1) or "no fresh boosts")
    else:
        from bot.config import Config
        cfg = Config.load()
        cfg.validate_keys()
        run_scout(cfg)
