# Multi-Signal Trading Bot (Paper Trading)

A real-time trading bot that trades **US stocks + crypto** through Alpaca's free
paper-trading API — real live market data, **fake money**. It runs a six-signal
technical ensemble with serious risk management, and shows everything it's
thinking on a live dashboard.

> **This starts in PAPER mode and should stay there until it has weeks of good
> results.** Nothing here is financial advice; even a good backtest doesn't
> guarantee live profits.

---

## 1. Setup (one time, ~5 minutes)

**a) Get free Alpaca paper keys**
1. Sign up at [alpaca.markets](https://alpaca.markets) (free, no deposit needed)
2. In the web dashboard, switch the account dropdown (top-left) to **Paper**
3. On the right side, find **API Keys** → Generate → copy both values

**b) Install**
```bash
# needs Python 3.10+  (python.org if you don't have it)
cd trading-bot
pip install -r requirements.txt
```

**c) Add your keys**
```bash
copy .env.example .env        # Windows   (mac/linux: cp .env.example .env)
```
…then open `.env` in any text editor and paste your two keys in.

## 2. Backtest first (recommended)

```bash
python run_backtest.py                # quick synthetic smoke test, no keys needed
python run_backtest.py SPY 30         # SPY, last 30 days of real data
python run_backtest.py BTC/USD 60     # Bitcoin, last 60 days
```
You get trades, win rate, return, max drawdown, Sharpe estimate, and profit
factor, plus full detail in `backtest_result.json`. If a symbol backtests
terribly, take it out of `config.yaml`.

## 3. Run it live (paper money)

```bash
python run_live.py
```
Then open **http://127.0.0.1:8050** — live equity, open positions with stops
and targets, every signal's current vote per symbol, and an activity feed.

Leave it running as long as you want. Crypto trades 24/7; stocks only during
market hours (8:30am–3pm your time). `Ctrl+C` stops the bot; positions stay in
your paper account and the bot re-adopts them on restart. Every trade is also
appended to `trades_log.jsonl` so you can review the full history.

## How the brain works

Six independent signal families each vote in **[-1, +1]** every cycle:

| Signal | What it looks at | Weight |
|---|---|---|
| Trend | EMA 20/50 separation + slope | 0.30 |
| Momentum | RSI regime + MACD histogram | 0.20 |
| Mean reversion | Bollinger %B stretched to an extreme | 0.15 |
| Volume | Volume surge confirming the move | 0.15 |
| Volatility regime | ATR expansion vs its baseline | 0.10 |
| Breakout | 20-bar Donchian channel break | 0.10 |

The weighted sum is the **composite score**. Above `+0.35` → buy; the position
is sized so a stop-out loses ~1% of equity, with an ATR-based stop-loss,
take-profit, and trailing stop. The dashboard shows every vote so you always
know *why* it traded.

## The safety net

All in `config.yaml` under `risk:` — max 1% equity risk per trade, max 10% in
any one position, max 6 open positions, a **daily kill switch** at -3% (no new
trades for the day), a **hard kill switch** at -10% drawdown (flattens
everything), and a 30-minute cooldown on any symbol that just stopped out.

## Tuning

Everything lives in `config.yaml`: symbols, bar size, how often it checks,
signal weights, entry threshold (lower = trades more), and all risk limits.
Change a value, restart the bot. **Backtest any change before running it.**

### The ML brain (the advanced part)

```bash
python train.py
```

This pulls ~2 years of history for a broad symbol set, builds 19 features per
bar, and trains a gradient-boosting model to predict the probability that
price rises past trading costs. It validates **walk-forward** — every test
fold is data the model never saw, with an embargo gap so labels can't leak —
and then gives one of three honest verdicts:

- **USE** — a real out-of-sample edge was found; `model.pkl` is saved and the
  bot automatically trades with the model on next restart
- **MARGINAL** — saved, but watch it skeptically
- **SKIP** — no reliable edge; nothing is saved and the hand-tuned ensemble
  stays in charge. A "no" from honest validation is worth more than a "yes"
  from a leaky one.

Retrain weekly-ish. Set `use_ml: false` in config.yaml to force the ensemble.
The dashboard works identically either way (the model's probability drives
the evidence meter).

### The smart-money scout

```bash
python scout.py          # run a scan manually (the live bot also runs it daily)
```

Scans the two public sources that are actually fresh and legal: **SEC Form 4
insider filings** (officers/directors must report their own-company trades
within 2 business days — cluster buying is a well-documented bullish signal)
and **news headlines** for every watched symbol. Findings become a small nudge
(capped at ±0.10) on each symbol's signal score — enough to tilt a close call,
never enough to overrule the strategy or risk manager. The dashboard shows a
"smart-$" tag on nudged symbols.

Deliberately NOT included: congressional trade copying (disclosures arrive up
to 45 days late by law — the free feeds are dead, and "instant Pelosi
tracking" is a myth) and social-media hype (mostly exit liquidity).

### Automatic tuning (recommended)

```bash
python tune.py SPY QQQ NVDA BTC/USD ETH/USD --days 120
```

This grid-searches timeframes, thresholds, and stop/target sizes on real
data — training on the first 70% and validating the winners on the last 30%
the search never saw (that out-of-sample check is what separates a real edge
from curve-fitting). The best config is saved to `tuned_config.yaml` for you
to review and copy over. If it tells you nothing works well in this period,
believe it — that honesty is the feature.

## Going live with real money — read this first

Not recommended until the paper account has performed well for at least a
month across different market conditions. If that day comes: fund a live
Alpaca account, generate *live* keys, and set `mode: live` in `config.yaml`.
The bot will make you type a confirmation before it starts. Start tiny.
