"""Schema + class-balance checks on the synthetic datasets."""
from __future__ import annotations


def test_home_credit_schema(home_credit_features):
    df = home_credit_features
    for col in ("SK_ID_CURR", "TARGET", "AMT_INCOME_TOTAL", "AMT_CREDIT", "EXT_SOURCE_2"):
        assert col in df.columns, f"missing {col}"
    # engineered features from the Spark/pandas pipeline
    for col in ("DTI", "CREDIT_INCOME_RATIO", "AGE_YEARS", "BUREAU_CNT", "PREV_REFUSED_RATE"):
        assert col in df.columns, f"missing engineered feature {col}"


def test_home_credit_class_balance(home_credit_features):
    rate = home_credit_features["TARGET"].mean()
    assert 0.04 < rate < 0.14, f"default rate {rate:.3f} not near the real ~8%"


def test_fraud_schema(fraud_df):
    assert "Class" in fraud_df.columns
    assert {f"V{i}" for i in range(1, 29)}.issubset(set(fraud_df.columns))
    assert "Amount" in fraud_df.columns and "Time" in fraud_df.columns


def test_fraud_extreme_imbalance(fraud_df):
    rate = fraud_df["Class"].mean()
    assert rate < 0.02, f"fraud rate {rate:.4f} should be a fraction of a percent"


def test_time_does_not_leak_class(fraud_df):
    """Regression guard: Time must NOT be assigned in class order (a past bug)."""
    corr = fraud_df[["Time", "Class"]].corr().loc["Time", "Class"]
    assert abs(corr) < 0.1, f"Time leaks class (corr={corr:.3f})"
