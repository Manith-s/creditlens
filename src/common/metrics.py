"""Discrimination metrics shared by the credit and fraud workstreams.

Includes the credit-scoring staples (KS, Gini) and the imbalanced-classification
staples (AUPRC, MCC) so every run logs the same comparable numbers to MLflow.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)


def ks_statistic(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Kolmogorov-Smirnov: max separation between cumulative good/bad distributions."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    order = np.argsort(y_score)
    y = y_true[order]
    pos = np.cumsum(y) / max(y.sum(), 1)
    neg = np.cumsum(1 - y) / max((1 - y).sum(), 1)
    return float(np.max(np.abs(pos - neg)))


def gini(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Gini = 2 * AUC - 1."""
    return float(2 * roc_auc_score(y_true, y_score) - 1)


def classification_metrics(
    y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    """Full metric bundle. AUPRC/recall/MCC matter most under heavy imbalance."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    y_pred = (y_score >= threshold).astype(int)
    pos = y_pred == 1
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    recall = tp / max(int((y_true == 1).sum()), 1)
    precision = tp / max(int(pos.sum()), 1)
    return {
        "auc": float(roc_auc_score(y_true, y_score)),
        "auprc": float(average_precision_score(y_true, y_score)),
        "ks": ks_statistic(y_true, y_score),
        "gini": gini(y_true, y_score),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_pred)) > 1 else 0.0,
        "recall": float(recall),
        "precision": float(precision),
        "positive_rate": float(y_true.mean()),
    }
