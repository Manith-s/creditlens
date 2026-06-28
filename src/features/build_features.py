"""Pandas equivalent of the Spark feature pipeline.

Produces the SAME applicant feature table as spark_features.py, but with no JVM, so
the modeling pipeline and CI run anywhere. Use Spark for the scale story; use this
for fast iteration and tests. The aggregation logic is intentionally identical.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.common.io import load_home_credit_application, load_home_credit_auxiliary
from src.config import CFG


def _bureau_aggregates(bureau: pd.DataFrame) -> pd.DataFrame:
    g = bureau.groupby("SK_ID_CURR")
    out = pd.DataFrame(
        {
            "BUREAU_CNT": g.size(),
            "BUREAU_ACTIVE_CNT": g["CREDIT_ACTIVE"].apply(lambda s: (s == "Active").sum()),
            "BUREAU_DEBT_MEAN": g["AMT_CREDIT_SUM_DEBT"].mean(),
            "BUREAU_OVERDUE_SUM": g["AMT_CREDIT_SUM_OVERDUE"].sum(),
            "BUREAU_MAX_DAYS_OVERDUE": g["CREDIT_DAY_OVERDUE"].max(),
        }
    )
    return out.reset_index()


def _prev_aggregates(prev: pd.DataFrame) -> pd.DataFrame:
    # recency window: most recent prior application per applicant (mirrors Spark row_number)
    prev = prev.sort_values(["SK_ID_CURR", "DAYS_DECISION"], ascending=[True, False])
    last = prev.groupby("SK_ID_CURR").first()["AMT_CREDIT"].rename("PREV_LAST_AMT_CREDIT")
    g = prev.groupby("SK_ID_CURR")
    out = pd.DataFrame(
        {
            "PREV_CNT": g.size(),
            "PREV_REFUSED_RATE": g["NAME_CONTRACT_STATUS"].apply(
                lambda s: (s == "Refused").mean()
            ),
        }
    )
    return out.join(last).reset_index()


def build_credit_features(save: bool = False) -> pd.DataFrame:
    app = load_home_credit_application()
    aux = load_home_credit_auxiliary()

    # SQL-equivalent derived ratios
    app = app.copy()
    app["DTI"] = app["AMT_ANNUITY"] / (app["AMT_INCOME_TOTAL"] / 12.0).replace(0, np.nan)
    app["CREDIT_INCOME_RATIO"] = app["AMT_CREDIT"] / app["AMT_INCOME_TOTAL"].replace(0, np.nan)
    app["AGE_YEARS"] = -app["DAYS_BIRTH"] / 365.25
    app["EMPLOYED_YEARS"] = (-app["DAYS_EMPLOYED"] / 365.25).clip(lower=0)

    feats = app
    if "bureau" in aux:
        feats = feats.merge(_bureau_aggregates(aux["bureau"]), on="SK_ID_CURR", how="left")
    if "previous_application" in aux:
        feats = feats.merge(_prev_aggregates(aux["previous_application"]), on="SK_ID_CURR", how="left")

    # fill the engineered aggregate gaps (applicants with no prior credit)
    fill_zero = [
        "BUREAU_CNT", "BUREAU_ACTIVE_CNT", "BUREAU_OVERDUE_SUM",
        "BUREAU_MAX_DAYS_OVERDUE", "PREV_CNT", "PREV_REFUSED_RATE",
    ]
    for c in fill_zero:
        if c in feats:
            feats[c] = feats[c].fillna(0)

    if save:
        out = CFG.processed / "credit_features.parquet"
        feats.to_parquet(out, index=False)
        print(f"Wrote {len(feats):,} x {feats.shape[1]} -> {out}")
    return feats


# Feature groups -> used by SHAP reason-code aggregation in src/explain/shap_codes.py
FEATURE_GROUPS: dict[str, list[str]] = {
    "External credit scores": ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"],
    "Debt-to-income": ["DTI", "AMT_ANNUITY", "CREDIT_INCOME_RATIO", "AMT_CREDIT"],
    "Income": ["AMT_INCOME_TOTAL"],
    "Age & employment": ["AGE_YEARS", "EMPLOYED_YEARS"],
    "Prior bureau delinquency": [
        "BUREAU_OVERDUE_SUM", "BUREAU_MAX_DAYS_OVERDUE", "BUREAU_DEBT_MEAN", "BUREAU_ACTIVE_CNT",
        "BUREAU_CNT",
    ],
    "Prior application history": ["PREV_REFUSED_RATE", "PREV_CNT", "PREV_LAST_AMT_CREDIT"],
    "Household": ["CNT_CHILDREN"],
    "Region": ["REGION_RATING_CLIENT"],
}

MODEL_FEATURES: list[str] = [f for cols in FEATURE_GROUPS.values() for f in cols]


if __name__ == "__main__":
    build_credit_features(save=True)
