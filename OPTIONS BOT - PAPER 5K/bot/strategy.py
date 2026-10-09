"""Multi-signal ensemble strategy (v2).

Six independent signals each score the market in [-1, +1]; the composite is
their weighted sum. v2 changes, driven by real-market backtest results:

  * All signals are computed VECTORIZED over the whole history at once
    (evaluate_all), so backtesting/tuning is ~100x faster. The live engine
    uses the same code path via evaluate() — one implementation, no drift.
  * A hard REGIME GATE: longs only when price is above its 200-bar EMA,
    shorts only below. The ensemble votes, but it can't fight the tide.
  * The old "signal fade" exit closed winners the moment enthusiasm dipped;
    the exit threshold now defaults below zero so a position is only closed
    early when the ensemble actually turns against it. The trailing stop is
    the primary exit.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import indicators as ta


@dataclass
class Signal:
    composite: float
    parts: dict = field(default_factory=dict)
    atr: float = 0.0
    price: float = 0.0
    regime: int = 0          # +1 above 200-EMA, -1 below

    def to_dict(self) -> dict:
        return {
            "composite": round(self.composite, 4),
            "parts": {k: round(v, 4) for k, v in self.parts.items()},
            "atr": round(self.atr, 6),
            "price": round(self.price, 6),
            "regime": self.regime,
        }


PART_KEYS = ["trend", "momentum", "mean_reversion", "volume", "volatility_regime", "breakout"]


class MultiSignalStrategy:
    def __init__(self, cfg: dict):
        self.weights: dict = cfg.get("weights", {})
        self.entry_threshold: float = float(cfg.get("entry_threshold", 0.5))
        self.exit_threshold: float = float(cfg.get("exit_threshold", -0.05))
        self.allow_shorts: bool = bool(cfg.get("allow_shorts", False))
        self.use_regime_filter: bool = bool(cfg.get("regime_filter", True))

    # ---------- vectorized signal computation ----------
    def evaluate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute every signal for every bar. Returns a DataFrame with
        columns: composite, atr, price, regime, and one column per signal."""
        close, volume = df["close"], df["volume"]
        out = pd.DataFrame(index=df.index)

        # --- trend: EMA 20/50 separation + fast-EMA slope
        fast, slow = ta.ema(close, 20), ta.ema(close, 50)
        sep = (fast - slow) / slow
        slp = ta.slope(fast, 10)
        out["trend"] = (sep / 0.01 * 0.6 + slp / 0.001 * 0.4).clip(-1, 1)

        # --- momentum: RSI regime + normalized MACD histogram
        r = ta.rsi(close, 14)
        _, _, hist = ta.macd(close)
        h = (hist / (close * 0.002)).clip(-1, 1)
        out["momentum"] = (0.5 * (r - 50.0) / 30.0 + 0.5 * h).clip(-1, 1)

        # --- mean reversion: fade Bollinger extremes
        _, _, _, pct_b = ta.bollinger(close)
        mr = pd.Series(0.0, index=df.index)
        low_mask, high_mask = pct_b < 0.05, pct_b > 0.95
        mr[low_mask] = ((0.05 - pct_b[low_mask]) / 0.15).clip(0, 1)
        mr[high_mask] = (-(pct_b[high_mask] - 0.95) / 0.15).clip(-1, 0)
        out["mean_reversion"] = mr

        # --- volume: surge confirming the current move
        vz = ta.volume_zscore(volume)
        ret = close.pct_change()
        out["volume"] = (np.sign(ret.fillna(0)) *
                         (vz.abs() / 2.5).clip(0, 1)).clip(-1, 1)

        # --- volatility regime: ATR expansion with the 20-bar direction
        a = ta.atr(df, 14)
        baseline = a.rolling(60).mean()
        ratio = (a / baseline).replace([np.inf, -np.inf], np.nan)
        trend_dir = np.sign(close - close.shift(20))
        vr = pd.Series(0.0, index=df.index)
        expanding = (ratio > 1.1) & (ratio <= 2.0)
        vr[expanding] = (trend_dir[expanding] * (ratio[expanding] - 1.0)).clip(-1, 1)
        vr[ratio > 2.0] = -0.5
        out["volatility_regime"] = vr.fillna(0.0)

        # --- breakout: Donchian channel break (vs previous bar's channel)
        upper, lower = ta.donchian(df, 20)
        prev_u, prev_l = upper.shift(1), lower.shift(1)
        rng = (prev_u - prev_l)
        pos = ((close - prev_l) / rng.replace(0, np.nan)).clip(0, 1)
        bo = ((pos - 0.5) * 0.6).fillna(0.0)
        bo[close > prev_u] = 1.0
        bo[close < prev_l] = -1.0
        out["breakout"] = bo.clip(-1, 1)

        # --- composite, context
        out["composite"] = sum(self.weights.get(k, 0.0) * out[k] for k in PART_KEYS)
        out["atr"] = a
        out["price"] = close
        ema200 = ta.ema(close, 200)
        out["regime"] = np.where(close >= ema200, 1, -1)
        # not enough history -> neutral
        warm = min(len(df), 60)
        out.iloc[:warm, out.columns.get_loc("composite")] = 0.0
        return out

    # ---------- single-bar evaluation (live engine) ----------
    def evaluate(self, df: pd.DataFrame) -> Signal:
        if len(df) < 60:
            return Signal(composite=0.0, parts={}, atr=0.0,
                          price=float(df["close"].iloc[-1]) if len(df) else 0.0)
        row = self.evaluate_all(df).iloc[-1]
        return self.signal_from_row(row)

    @staticmethod
    def signal_from_row(row) -> Signal:
        return Signal(
            composite=float(row["composite"]),
            parts={k: float(row[k]) for k in PART_KEYS},
            atr=float(row["atr"]) if row["atr"] == row["atr"] else 0.0,
            price=float(row["price"]),
            regime=int(row["regime"]),
        )

    # ---------- decisions ----------
    def wants_entry(self, sig: Signal) -> str | None:
        if sig.composite >= self.entry_threshold:
            if self.use_regime_filter and sig.regime < 0:
                return None                      # don't buy into a downtrend
            return "long"
        if self.allow_shorts and sig.composite <= -self.entry_threshold:
            if self.use_regime_filter and sig.regime > 0:
                return None
            return "short"
        return None

    def wants_exit(self, sig: Signal, side: str) -> bool:
        if side == "long":
            return sig.composite < self.exit_threshold
        return sig.composite > -self.exit_threshold


class MLStrategy(MultiSignalStrategy):
    """Same interface as the ensemble, but the composite score comes from a
    trained model's probability that price rises past costs: score = 2*P - 1.

    Entry/exit thresholds are expressed as probabilities in config
    (ml_entry_prob / ml_exit_prob) and mapped onto the same score scale, so
    the engine, backtester, and dashboard need no special cases. The regime
    filter and all risk management still apply — the model proposes, the
    risk manager disposes.
    """

    def __init__(self, cfg: dict, model, meta: dict):
        super().__init__(cfg)
        self.model = model
        self.meta = meta or {}
        entry_p = float(cfg.get("ml_entry_prob", 0.58))
        exit_p = float(cfg.get("ml_exit_prob", 0.45))
        self.entry_threshold = 2 * entry_p - 1
        self.exit_threshold = 2 * exit_p - 1

    def evaluate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        from .features import FEATURES, build_features
        X = build_features(df)
        out = pd.DataFrame(index=df.index)
        valid = X.notna().all(axis=1)
        proba = np.full(len(df), 0.5)
        if valid.any():
            proba[valid.to_numpy()] = self.model.predict_proba(X[valid])[:, 1]
        out["composite"] = 2 * proba - 1
        out["ml_prob"] = proba
        out["atr"] = ta.atr(df, 14)
        out["price"] = df["close"]
        ema200 = ta.ema(df["close"], 200)
        out["regime"] = np.where(df["close"] >= ema200, 1, -1)
        for k in PART_KEYS:            # keep schema identical for the engine/dashboard
            out[k] = 0.0
        out.loc[~valid, "composite"] = 0.0
        return out

    @staticmethod
    def signal_from_row(row) -> Signal:
        sig = MultiSignalStrategy.signal_from_row(row)
        if "ml_prob" in row:
            sig.parts = {"ml_prob": float(row["ml_prob"])}
        return sig


def make_strategy(cfg: dict):
    """Factory: ML model if trained + enabled, otherwise the ensemble.

    strategy.use_ml: auto (default) -> use model.pkl when it exists
                     true           -> require it (error if missing)
                     false          -> ensemble only
    """
    use_ml = cfg.get("use_ml", "auto")
    if use_ml is False:
        return MultiSignalStrategy(cfg)
    from .ml import load_model
    model, meta = load_model()
    if model is not None:
        return MLStrategy(cfg, model, meta)
    if use_ml is True:
        raise SystemExit("strategy.use_ml is true but no model.pkl found — run: python train.py")
    return MultiSignalStrategy(cfg)
