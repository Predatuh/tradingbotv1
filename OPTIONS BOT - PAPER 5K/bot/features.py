"""Feature engineering for the ML model.

Everything is computed vectorized from OHLCV bars. The same function feeds
training (train.py), backtesting, and the live engine — one code path, so the
model always sees features built exactly the way it was trained on.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import indicators as ta

FEATURES = [
    "ret_1", "ret_3", "ret_6", "ret_12", "ret_24",
    "rsi", "macd_hist", "bb_pctb", "vol_z", "vol_trend",
    "atr_ratio", "ema_sep_fast", "ema_sep_slow", "ema_slope",
    "donchian_pos", "regime", "hour_sin", "hour_cos", "dow",
]

LABEL_HORIZON = 12          # look this many bars ahead
LABEL_COST = 0.0015         # move must clear ~2x round-trip cost to count


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature matrix aligned to df's index. NaNs during warmup (~200 bars)."""
    close, volume = df["close"], df["volume"]
    f = pd.DataFrame(index=df.index)

    for lag in (1, 3, 6, 12, 24):
        f[f"ret_{lag}"] = close.pct_change(lag)

    f["rsi"] = (ta.rsi(close, 14) - 50.0) / 50.0
    _, _, hist = ta.macd(close)
    f["macd_hist"] = (hist / (close * 0.002)).clip(-3, 3)
    _, _, _, pct_b = ta.bollinger(close)
    f["bb_pctb"] = (pct_b - 0.5).clip(-1.5, 1.5)
    f["vol_z"] = ta.volume_zscore(volume).clip(-4, 4)
    f["vol_trend"] = (volume.rolling(5).mean() / volume.rolling(20).mean() - 1).clip(-2, 2)

    a = ta.atr(df, 14)
    f["atr_ratio"] = (a / a.rolling(60).mean() - 1).clip(-2, 3)

    e20, e50, e200 = ta.ema(close, 20), ta.ema(close, 50), ta.ema(close, 200)
    f["ema_sep_fast"] = ((e20 - e50) / e50 * 100).clip(-5, 5)
    f["ema_sep_slow"] = ((e50 - e200) / e200 * 100).clip(-10, 10)
    f["ema_slope"] = (ta.slope(e20, 10) * 1000).clip(-5, 5)

    upper, lower = ta.donchian(df, 20)
    rng = (upper - lower).replace(0, np.nan)
    f["donchian_pos"] = ((close - lower) / rng - 0.5).clip(-0.5, 0.5)

    f["regime"] = np.where(close >= e200, 1.0, -1.0)

    hours = df.index.hour + df.index.minute / 60.0
    f["hour_sin"] = np.sin(2 * np.pi * hours / 24)
    f["hour_cos"] = np.cos(2 * np.pi * hours / 24)
    f["dow"] = df.index.dayofweek / 6.0

    return f[FEATURES]


def make_labels(df: pd.DataFrame, horizon: int = LABEL_HORIZON,
                cost: float = LABEL_COST) -> pd.Series:
    """1 if price rises more than `cost` over the next `horizon` bars,
    0 if it falls more than `cost`. Small in-between moves become NaN and are
    dropped from training — they're noise nobody can profit from anyway."""
    fwd = df["close"].shift(-horizon) / df["close"] - 1
    y = pd.Series(np.nan, index=df.index)
    y[fwd > cost] = 1.0
    y[fwd < -cost] = 0.0
    return y


def training_rows(df: pd.DataFrame, symbol: str,
                  horizon: int = LABEL_HORIZON) -> pd.DataFrame:
    """Features + label + bookkeeping columns for one symbol, NaNs dropped."""
    X = build_features(df)
    y = make_labels(df, horizon)
    out = X.copy()
    out["y"] = y
    out["symbol"] = symbol
    out["fwd_ret"] = df["close"].shift(-horizon) / df["close"] - 1
    out = out.dropna(subset=FEATURES + ["y"])
    return out
