"""WOE/IV correctness: monotonicity on a monotonic signal, IV ordering, and the
Information Value ranking selection floor."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.woe import information_value_ranking, woe_iv_numeric


def test_woe_monotonic_on_monotonic_signal():
    """If P(bad) increases monotonically with x, WOE across ordered bins should be
    (close to) monotonically decreasing — WOE = log(good/bad)."""
    rng = np.random.default_rng(0)
    n = 8000
    x = rng.uniform(0, 1, n)
    y = (rng.uniform(size=n) < x).astype(int)  # higher x -> more likely bad
    res = woe_iv_numeric(pd.Series(x), pd.Series(y), n_bins=10)
    woe = res.table.sort_values("bin")["woe"].values
    # allow a couple of inversions from sampling noise; overall trend must be down
    diffs = np.diff(woe)
    assert (diffs <= 0).mean() >= 0.7, f"WOE not predominantly monotonic: {woe}"
    assert res.iv > 0.1, f"strong signal should yield meaningful IV, got {res.iv:.3f}"


def test_iv_zero_for_noise():
    rng = np.random.default_rng(1)
    n = 6000
    x = rng.normal(size=n)
    y = rng.integers(0, 2, n)  # independent of x
    res = woe_iv_numeric(pd.Series(x), pd.Series(y), n_bins=10)
    assert res.iv < 0.05, f"noise feature should have ~0 IV, got {res.iv:.3f}"


def test_information_value_ranking_selects_predictive(home_credit_features):
    feats = ["EXT_SOURCE_2", "EXT_SOURCE_3", "DTI", "AMT_INCOME_TOTAL"]
    ranking = information_value_ranking(home_credit_features, "TARGET", feats, iv_floor=0.02)
    assert set(ranking["feature"]) == set(feats)
    assert ranking["iv"].is_monotonic_decreasing  # sorted by IV desc
    # at least one strong predictor should clear the floor
    assert ranking["selected"].any()
