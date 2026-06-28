"""Weight of Evidence (WOE) and Information Value (IV).

Primary path uses **optbinning** (monotonic, constrained binning) when installed.
A dependency-free pandas implementation is always available so unit tests (WOE
monotonicity, IV ranking) run in CI without the native optbinning build.

IV interpretation (industry rule of thumb):
    < 0.02  not predictive   |  0.02-0.1 weak  |  0.1-0.3 medium
    0.3-0.5 strong           |  > 0.5  suspiciously strong (check for leakage)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

EPS = 0.5  # Laplace smoothing so empty bins don't blow up the log


@dataclass
class WOEResult:
    iv: float
    table: pd.DataFrame  # bin, n, n_bad, woe, iv_contribution


def woe_iv_numeric(
    x: pd.Series, y: pd.Series, n_bins: int = 10
) -> WOEResult:
    """Quantile-binned WOE/IV for a numeric feature (pandas-only)."""
    df = pd.DataFrame({"x": x.values, "y": np.asarray(y)})
    # quantile bins; missing values get their own bin
    df["bin"] = pd.qcut(df["x"], q=n_bins, duplicates="drop")
    df["bin"] = df["bin"].cat.add_categories(["MISSING"]).fillna("MISSING")
    return _woe_from_bins(df)


def woe_iv_categorical(x: pd.Series, y: pd.Series) -> WOEResult:
    df = pd.DataFrame({"x": x.astype("object").fillna("MISSING").values, "y": np.asarray(y)})
    df["bin"] = df["x"]
    return _woe_from_bins(df)


def _woe_from_bins(df: pd.DataFrame) -> WOEResult:
    total_bad = max(df["y"].sum(), 1)
    total_good = max((1 - df["y"]).sum(), 1)
    rows = []
    for b, g in df.groupby("bin", observed=True):
        n = len(g)
        bad = g["y"].sum()
        good = n - bad
        dist_bad = (bad + EPS) / (total_bad + EPS)
        dist_good = (good + EPS) / (total_good + EPS)
        woe = float(np.log(dist_good / dist_bad))
        iv_c = (dist_good - dist_bad) * woe
        rows.append({"bin": str(b), "n": n, "n_bad": int(bad), "woe": woe, "iv_contribution": iv_c})
    table = pd.DataFrame(rows)
    return WOEResult(iv=float(table["iv_contribution"].sum()), table=table)


def information_value_ranking(
    df: pd.DataFrame, target: str, features: list[str] | None = None, iv_floor: float = 0.02
) -> pd.DataFrame:
    """Rank features by IV and flag those that pass the selection floor."""
    y = df[target]
    feats = features or [c for c in df.columns if c != target]
    out = []
    for f in feats:
        s = df[f]
        try:
            if pd.api.types.is_numeric_dtype(s) and s.nunique(dropna=True) > 10:
                res = woe_iv_numeric(s, y)
            else:
                res = woe_iv_categorical(s, y)
            out.append({"feature": f, "iv": res.iv, "selected": res.iv >= iv_floor})
        except Exception:  # noqa: BLE001 — a constant/degenerate column shouldn't kill the ranking
            out.append({"feature": f, "iv": 0.0, "selected": False})
    return pd.DataFrame(out).sort_values("iv", ascending=False).reset_index(drop=True)


def optbinning_process(df: pd.DataFrame, target: str, features: list[str]):
    """Fit optbinning's BinningProcess (monotonic WOE). Returns (process, X_woe).

    Raises ImportError if optbinning isn't installed — callers should fall back to
    the pandas WOE above.
    """
    from optbinning import BinningProcess

    X = df[features]
    y = df[target].values
    process = BinningProcess(variable_names=features)
    process.fit(X.values, y)
    X_woe = process.transform(X.values, metric="woe")
    return process, pd.DataFrame(X_woe, columns=features, index=df.index)


def has_optbinning() -> bool:
    try:
        import optbinning  # noqa: F401

        return True
    except ImportError:
        return False
