"""Model smoke tests: metrics bundle, GBM training, scorecard, SHAP reason codes."""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split

from src.common.metrics import classification_metrics, gini, ks_statistic
from src.features.build_features import MODEL_FEATURES


def test_metrics_bundle_ranges():
    rng = np.random.default_rng(0)
    y = (rng.uniform(size=2000) < 0.1).astype(int)
    score = np.clip(y * 0.4 + rng.uniform(size=2000) * 0.6, 0, 1)
    m = classification_metrics(y, score)
    for k in ("auc", "auprc", "ks", "gini", "f1", "mcc", "recall", "precision"):
        assert k in m
    assert 0.0 <= m["auc"] <= 1.0
    assert -1.0 <= m["gini"] <= 1.0
    assert 0.0 <= m["ks"] <= 1.0


def test_ks_and_gini_consistent_with_auc():
    rng = np.random.default_rng(1)
    y = (rng.uniform(size=3000) < 0.2).astype(int)
    score = np.clip(y * 0.5 + rng.normal(0, 0.4, 3000), 0, 1)
    from sklearn.metrics import roc_auc_score

    assert abs(gini(y, score) - (2 * roc_auc_score(y, score) - 1)) < 1e-9
    assert 0.0 <= ks_statistic(y, score) <= 1.0


def test_gbm_beats_random(home_credit_features):
    from src.models.gbm import predict_proba, train_gbm

    feats = [f for f in MODEL_FEATURES if f in home_credit_features.columns]
    df = home_credit_features
    X = df[feats].fillna(df[feats].median(numeric_only=True))
    y = df["TARGET"].values
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)
    model = train_gbm(Xtr, ytr, random_state=0)
    m = classification_metrics(yte, predict_proba(model, Xte))
    assert m["auc"] > 0.65, f"GBM AUC {m['auc']:.3f} unexpectedly low on synthetic credit data"


def test_scorecard_runs_and_scores(home_credit_features):
    from src.models.scorecard import Scorecard

    feats = [f for f in MODEL_FEATURES if f in home_credit_features.columns][:8]
    df = home_credit_features
    Xtr, Xte = train_test_split(df, test_size=0.3, stratify=df["TARGET"], random_state=0)
    sc = Scorecard(features=feats).fit(Xtr, "TARGET")
    p = sc.predict_proba(Xte)
    pts = sc.score(Xte)
    assert ((p >= 0) & (p <= 1)).all()
    assert np.isfinite(pts).all()


def test_shap_reason_codes(home_credit_features):
    from src.explain.shap_codes import reason_codes_for
    from src.models.gbm import train_gbm

    feats = [f for f in MODEL_FEATURES if f in home_credit_features.columns]
    df = home_credit_features
    X = df[feats].fillna(df[feats].median(numeric_only=True)).reset_index(drop=True)
    model = train_gbm(X, df["TARGET"].values, random_state=0)
    codes = reason_codes_for(model, X, row_index=0, top_n=4)
    assert isinstance(codes, list)
    assert len(codes) <= 4  # Reg B notices cap the reasons disclosed
