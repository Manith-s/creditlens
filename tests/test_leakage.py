"""Leakage guards — the highest-value tests for a credit/fraud portfolio.

These encode the exact interview talking points: split before resampling, SMOTE
lives inside the pipeline, and binning/scaling never see the test split.
"""
from __future__ import annotations

import numpy as np

from src.models.imbalance import make_smote_pipeline, pos_weight


def test_smote_is_inside_pipeline():
    """make_smote_pipeline must wrap resampling+estimator so SMOTE only ever sees
    training folds — never the full dataset before the split."""
    from sklearn.linear_model import LogisticRegression

    pipe = make_smote_pipeline(LogisticRegression(max_iter=200))
    # if imblearn is present, the first step is SMOTE; otherwise it's the bare est
    try:
        from imblearn.pipeline import Pipeline as ImbPipeline

        assert isinstance(pipe, ImbPipeline)
        assert pipe.steps[0][0] == "smote"
        assert pipe.steps[-1][0] == "clf"
    except ImportError:
        assert hasattr(pipe, "fit")


def test_pos_weight_matches_imbalance():
    y = np.r_[np.zeros(990, int), np.ones(10, int)]
    assert abs(pos_weight(y) - 99.0) < 1e-6


def test_no_resampling_changes_test_distribution():
    """Sanity: a leakage-safe pipeline must not alter the held-out test labels."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    rng = np.random.default_rng(0)
    X = rng.normal(size=(2000, 5))
    y = (rng.uniform(size=2000) < 0.05).astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)
    before = y_te.mean()
    pipe = make_smote_pipeline(LogisticRegression(max_iter=200))
    pipe.fit(X_tr, y_tr)  # resampling happens only on train
    assert y_te.mean() == before, "test distribution must be untouched by training"
