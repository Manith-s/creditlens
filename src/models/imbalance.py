"""Leakage-safe imbalance handling for the fraud workstream.

The cardinal rule (and a classic interview trap): resample INSIDE the pipeline,
fit on TRAIN folds only, NEVER before the split. This module provides:
  * make_smote_pipeline(): an imblearn Pipeline (SMOTE -> estimator) that is
    leakage-safe by construction.
  * pos_weight(): scale_pos_weight / pos_weight for cost-sensitive learning.
  * class_weight_dict(): for Keras `class_weight=`.
"""
from __future__ import annotations

import numpy as np


def pos_weight(y: np.ndarray) -> float:
    y = np.asarray(y)
    pos = max(int((y == 1).sum()), 1)
    return int((y == 0).sum()) / pos


def class_weight_dict(y: np.ndarray) -> dict[int, float]:
    """Balanced class weights, normalized so the majority class ~= 1.0."""
    y = np.asarray(y)
    n = len(y)
    n_pos = max(int((y == 1).sum()), 1)
    n_neg = max(int((y == 0).sum()), 1)
    return {0: n / (2 * n_neg), 1: n / (2 * n_pos)}


def make_smote_pipeline(estimator, sampling_strategy: float = 0.1, random_state: int = 42):
    """SMOTE -> estimator as a single imblearn Pipeline (leakage-safe).

    `sampling_strategy=0.1` oversamples the minority to 10% of the majority — a
    moderate ratio that avoids the over-synthesis that hurts precision. Falls back
    to the bare estimator if imbalanced-learn isn't installed.
    """
    try:
        from imblearn.over_sampling import SMOTE
        from imblearn.pipeline import Pipeline as ImbPipeline

        return ImbPipeline(
            steps=[
                ("smote", SMOTE(sampling_strategy=sampling_strategy, random_state=random_state)),
                ("clf", estimator),
            ]
        )
    except ImportError:
        return estimator
