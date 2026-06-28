"""Points-based credit scorecard.

Primary path: optbinning's `Scorecard` (WOE binning + logistic regression + PDO
scaling) — the industry-standard build. Fallback path (no optbinning): a
WOE-transform + scikit-learn `LogisticRegression`, then manual PDO scaling, so the
scorecard story still runs and tests pass anywhere.

PDO scaling ("points to double the odds"):
    factor = PDO / ln(2)
    offset = target_points - factor * ln(target_odds)
    score  = offset + factor * ln(odds_good/bad)
Standard config here: 600 points at 50:1 odds, PDO 20 (matches the governance doc).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.features.woe import has_optbinning, optbinning_process


@dataclass
class ScorecardConfig:
    target_points: int = 600
    target_odds: float = 50.0
    pdo: int = 20


def pdo_params(cfg: ScorecardConfig) -> tuple[float, float]:
    factor = cfg.pdo / np.log(2)
    offset = cfg.target_points - factor * np.log(cfg.target_odds)
    return factor, offset


class Scorecard:
    """Thin wrapper that picks optbinning when available, else the sklearn fallback."""

    def __init__(self, features: list[str], cfg: ScorecardConfig | None = None):
        self.features = features
        self.cfg = cfg or ScorecardConfig()
        self.backend = "optbinning" if has_optbinning() else "sklearn-woe"
        self._sc = None  # optbinning Scorecard
        self._lr = None  # sklearn LR
        self._process = None  # optbinning BinningProcess (fallback path)

    # -- fit -----------------------------------------------------------------
    def fit(self, df: pd.DataFrame, target: str) -> Scorecard:
        if self.backend == "optbinning":
            self._fit_optbinning(df, target)
        else:
            self._fit_sklearn(df, target)
        return self

    def _fit_optbinning(self, df: pd.DataFrame, target: str) -> None:
        from optbinning import BinningProcess
        from optbinning import Scorecard as OBScorecard
        from sklearn.linear_model import LogisticRegression as LR

        binning = BinningProcess(variable_names=self.features)
        f, o = pdo_params(self.cfg)
        self._sc = OBScorecard(
            binning_process=binning,
            estimator=LR(max_iter=1000),
            scaling_method="pdo_odds",
            scaling_method_params={"pdo": self.cfg.pdo, "odds": self.cfg.target_odds,
                                   "scorecard_points": self.cfg.target_points},
        )
        self._sc.fit(df[self.features], df[target].values)

    def _fit_sklearn(self, df: pd.DataFrame, target: str) -> None:
        try:
            self._process, X_woe = optbinning_process(df, target, self.features)
        except ImportError:
            # last-resort: median-impute + standardize-free LR on raw features
            X_woe = df[self.features].fillna(df[self.features].median(numeric_only=True))
        self._lr = LogisticRegression(max_iter=1000)
        self._lr.fit(X_woe.values, df[target].values)
        self._woe_cols = list(X_woe.columns)

    # -- predict -------------------------------------------------------------
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        if self.backend == "optbinning":
            return self._sc.predict_proba(df[self.features])[:, 1]
        if self._process is not None:
            X = pd.DataFrame(
                self._process.transform(df[self.features].values, metric="woe"),
                columns=self.features,
            )
        else:
            X = df[self.features].fillna(df[self.features].median(numeric_only=True))
        return self._lr.predict_proba(X.values)[:, 1]

    def score(self, df: pd.DataFrame) -> np.ndarray:
        """Map probability of default to scorecard points (higher = lower risk)."""
        if self.backend == "optbinning":
            return self._sc.score(df[self.features])
        p = np.clip(self.predict_proba(df), 1e-6, 1 - 1e-6)
        odds_good_bad = (1 - p) / p
        factor, offset = pdo_params(self.cfg)
        return offset + factor * np.log(odds_good_bad)

    def table(self) -> pd.DataFrame | None:
        """Printable scorecard table (optbinning backend only)."""
        if self.backend == "optbinning":
            return self._sc.table(style="summary")
        return None
