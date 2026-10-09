"""ML model: gradient boosting with walk-forward validation.

The model predicts P(price rises more than trading costs over the next N bars)
from the features in features.py. Honesty rules baked in:

  * Walk-forward: every validation fold is tested on data STRICTLY AFTER
    everything it trained on. No shuffling, no leakage.
  * An embargo gap of LABEL_HORIZON bars between train and test, so labels
    that peek into the test window can't leak.
  * The final verdict compares the model against "always predict up" on the
    same bars — beating a coin, not just scoring well.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

from .features import FEATURES, LABEL_HORIZON

ROOT = Path(__file__).resolve().parent.parent
MODEL_FILE = ROOT / "model.pkl"
META_FILE = ROOT / "model_meta.json"


def _new_model() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_depth=4,
        min_samples_leaf=200, l2_regularization=1.0,
        early_stopping=True, validation_fraction=0.15, random_state=42,
    )


@dataclass
class FoldResult:
    fold: int
    auc: float
    n_test: int
    hit_rate_at_thr: float      # of bars the model liked, how many went up
    base_up_rate: float          # how many bars went up overall (the coin to beat)
    avg_fwd_ret_at_thr: float    # average forward return on model-liked bars
    avg_fwd_ret_all: float

    def edge(self) -> float:
        return self.hit_rate_at_thr - self.base_up_rate


@dataclass
class WalkForwardReport:
    folds: list = field(default_factory=list)
    threshold: float = 0.58

    def mean_auc(self) -> float:
        return float(np.mean([f.auc for f in self.folds])) if self.folds else 0.0

    def mean_edge(self) -> float:
        return float(np.mean([f.edge() for f in self.folds])) if self.folds else 0.0

    def verdict(self) -> str:
        auc, edge = self.mean_auc(), self.mean_edge()
        pos_folds = sum(1 for f in self.folds if f.edge() > 0)
        if auc >= 0.54 and edge >= 0.03 and pos_folds >= len(self.folds) * 0.6:
            return "USE"
        if auc >= 0.52 and edge > 0.01:
            return "MARGINAL"
        return "SKIP"


def walk_forward(rows: pd.DataFrame, n_folds: int = 5,
                 threshold: float = 0.58) -> WalkForwardReport:
    """rows: pooled training_rows() output from all symbols, sorted by time."""
    rows = rows.sort_index(kind="stable")
    n = len(rows)
    report = WalkForwardReport(threshold=threshold)
    fold_size = n // (n_folds + 1)

    for k in range(1, n_folds + 1):
        train_end = fold_size * k
        test_start = train_end + LABEL_HORIZON          # embargo gap
        test_end = min(train_end + fold_size, n)
        if test_start >= test_end:
            continue
        tr, te = rows.iloc[:train_end], rows.iloc[test_start:test_end]
        if tr["y"].nunique() < 2 or len(te) < 50:
            continue
        m = _new_model()
        m.fit(tr[FEATURES], tr["y"])
        p = m.predict_proba(te[FEATURES])[:, 1]
        liked = p >= threshold
        report.folds.append(FoldResult(
            fold=k,
            auc=float(roc_auc_score(te["y"], p)),
            n_test=len(te),
            hit_rate_at_thr=float(te["y"][liked].mean()) if liked.any() else float("nan"),
            base_up_rate=float(te["y"].mean()),
            avg_fwd_ret_at_thr=float(te["fwd_ret"][liked].mean()) if liked.any() else float("nan"),
            avg_fwd_ret_all=float(te["fwd_ret"].mean()),
        ))
    return report


def train_final(rows: pd.DataFrame, meta_extra: dict | None = None):
    """Train on ALL rows and persist. Only call after walk_forward looks good."""
    rows = rows.sort_index(kind="stable")
    m = _new_model()
    m.fit(rows[FEATURES], rows["y"])
    joblib.dump(m, MODEL_FILE)

    # permutation importance on the most recent 20% (what the model will face)
    tail = rows.iloc[-max(2000, len(rows) // 5):]
    imp = permutation_importance(m, tail[FEATURES], tail["y"],
                                 n_repeats=3, random_state=0, scoring="roc_auc")
    importance = sorted(zip(FEATURES, imp.importances_mean.round(5)),
                        key=lambda kv: -kv[1])
    meta = {
        "features": FEATURES,
        "n_rows": int(len(rows)),
        "symbols": sorted(rows["symbol"].unique().tolist()),
        "label_horizon": LABEL_HORIZON,
        "importance": importance,
        **(meta_extra or {}),
    }
    META_FILE.write_text(json.dumps(meta, indent=1, default=str))
    return m, meta


def load_model():
    """Returns (model, meta) or (None, None) if not trained yet."""
    if not MODEL_FILE.exists():
        return None, None
    meta = json.loads(META_FILE.read_text()) if META_FILE.exists() else {}
    return joblib.load(MODEL_FILE), meta
