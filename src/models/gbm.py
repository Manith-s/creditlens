"""Gradient-boosted models (XGBoost preferred, LightGBM or sklearn fallback).

Used by both workstreams: the credit "AUC lift over the logistic baseline" story
and the fraud "GBM usually beats the NN on tabular data" benchmark. Imbalance is
handled honestly with `scale_pos_weight` (computed from the TRAIN split only).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _pos_weight(y: np.ndarray) -> float:
    pos = max(int((y == 1).sum()), 1)
    neg = int((y == 0).sum())
    return neg / pos


def train_gbm(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    *,
    handle_imbalance: bool = False,
    random_state: int = 42,
):
    """Return a fitted GBM. Tries XGBoost, then LightGBM, then sklearn HistGBM.

    `handle_imbalance=True` sets scale_pos_weight (XGBoost) / is_unbalance
    (LightGBM) — use it for fraud, leave off for the credit baseline comparison.
    """
    spw = _pos_weight(y_train) if handle_imbalance else 1.0

    try:
        from xgboost import XGBClassifier

        model = XGBClassifier(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="aucpr" if handle_imbalance else "auc",
            scale_pos_weight=spw,
            random_state=random_state,
            n_jobs=-1,
            tree_method="hist",
        )
        model.fit(X_train, y_train)
        model._creditlens_backend = "xgboost"
        return model
    except ImportError:
        pass

    try:
        from lightgbm import LGBMClassifier

        model = LGBMClassifier(
            n_estimators=400,
            max_depth=-1,
            num_leaves=31,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            is_unbalance=handle_imbalance,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        )
        model.fit(X_train, y_train)
        model._creditlens_backend = "lightgbm"
        return model
    except ImportError:
        pass

    from sklearn.ensemble import HistGradientBoostingClassifier

    model = HistGradientBoostingClassifier(
        max_iter=400, max_depth=5, learning_rate=0.05, random_state=random_state
    )
    sample_weight = None
    if handle_imbalance:
        sample_weight = np.where(y_train == 1, spw, 1.0)
    model.fit(X_train, y_train, sample_weight=sample_weight)
    model._creditlens_backend = "sklearn-histgbm"
    return model


def predict_proba(model, X: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(X)[:, 1]
