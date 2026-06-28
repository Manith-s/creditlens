"""Generate small, schema-faithful SYNTHETIC versions of every dataset so the
whole portfolio runs offline with zero downloads.

These are NOT the real datasets. They mimic the column names, dtypes, class
balance, and (importantly) inject genuine signal so models reach realistic
metrics. Swap in the real data with `src/data/download.py` for the headline
numbers you put on a resume.

Usage:
    python -m src.data.make_synthetic --all
    python -m src.data.make_synthetic --home-credit --n 20000
    python -m src.data.make_synthetic --fraud --paysim
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.config import CFG, rel_to_root, set_global_seed


# ---------------------------------------------------------------------------
# Home Credit Default Risk  (anchor for the credit scorecard)
#   real application_train: 307,511 rows x 122 cols, TARGET ~8% positive.
# We generate a faithful-but-small subset + two auxiliary tables so the Spark
# feature-engineering story (joins, groupBy, window functions) has real inputs.
# ---------------------------------------------------------------------------
def make_home_credit(n: int = 20_000, seed: int = 42) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    sk_id = np.arange(100_001, 100_001 + n)

    # --- core driver: a latent creditworthiness score we never expose ---
    age_years = rng.normal(43, 11, n).clip(21, 69)
    income = rng.lognormal(mean=11.9, sigma=0.5, size=n).clip(25_000, 1_000_000)
    credit_amt = (income * rng.uniform(0.5, 6.0, n)).clip(45_000, 4_000_000)
    annuity = credit_amt / rng.uniform(10, 40, n)
    ext1 = rng.beta(2.5, 2.0, n)  # mimics EXT_SOURCE_* (strongest real predictors)
    ext2 = rng.beta(2.5, 2.0, n)
    ext3 = rng.beta(2.5, 2.0, n)
    employed_years = rng.exponential(6, n).clip(0, 45)
    n_children = rng.poisson(0.5, n).clip(0, 8)

    # latent log-odds of default. Calibrated so a WOE/logistic scorecard lands
    # ~0.74-0.78 AUC and a GBM ~0.80-0.84 (matching the resume's lift story).
    dti = annuity / (income / 12.0)
    dti_z = (dti - dti.mean()) / dti.std()
    ext_avg = (ext1 + ext2 + ext3) / 3.0

    # LINEAR signal (what a WOE-logistic scorecard can capture), standardized so
    # its strength is controllable independent of feature scales.
    lin = (
        -1.1 * (ext1 - 0.5)
        - 1.0 * (ext2 - 0.5)
        - 0.9 * (ext3 - 0.5)
        + 0.20 * dti_z.clip(-3, 3)  # clip the heavy tail so annuity isn't a giveaway
        - 0.010 * (age_years - 43)
        - 0.025 * employed_years
        + 0.07 * n_children
    )
    lin = (lin - lin.mean()) / lin.std()

    # NON-ADDITIVE (XOR-style) interaction between DTI and external scores. An
    # additive WOE-logistic scorecard literally cannot represent this term; a
    # tree-based GBM can -> the source of a genuine, honest AUC lift (~+0.04),
    # matching the resume's "0.78 -> 0.84" story without borrowing the number.
    ext_z = (ext_avg - ext_avg.mean()) / ext_avg.std()
    xor = 1.7 * np.sign(dti_z.clip(-3, 3)) * np.sign(ext_z)

    # noise tuned so scorecard ~0.78, GBM ~0.82 (real default is mostly noise)
    score = 2.0 * lin + xor + rng.normal(0, 1.3, n)

    # solve the intercept so prevalence ~= 8% (Bernoulli sampling, no hard cutoff)
    target_rate = 0.08
    lo, hi = -12.0, 6.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if (1 / (1 + np.exp(-(score + mid)))).mean() > target_rate:
            hi = mid
        else:
            lo = mid
    p_default = 1 / (1 + np.exp(-(score + (lo + hi) / 2)))
    target = (rng.uniform(size=n) < p_default).astype(int)

    app = pd.DataFrame(
        {
            "SK_ID_CURR": sk_id,
            "TARGET": target,
            "NAME_CONTRACT_TYPE": rng.choice(["Cash loans", "Revolving loans"], n, p=[0.9, 0.1]),
            "CODE_GENDER": rng.choice(["M", "F"], n),
            "FLAG_OWN_CAR": rng.choice(["Y", "N"], n, p=[0.34, 0.66]),
            "FLAG_OWN_REALTY": rng.choice(["Y", "N"], n, p=[0.69, 0.31]),
            "CNT_CHILDREN": n_children,
            "AMT_INCOME_TOTAL": income.round(2),
            "AMT_CREDIT": credit_amt.round(2),
            "AMT_ANNUITY": annuity.round(2),
            "AMT_GOODS_PRICE": (credit_amt * rng.uniform(0.8, 1.0, n)).round(2),
            "NAME_INCOME_TYPE": rng.choice(
                ["Working", "Commercial associate", "Pensioner", "State servant"],
                n, p=[0.52, 0.23, 0.18, 0.07],
            ),
            "NAME_EDUCATION_TYPE": rng.choice(
                ["Secondary / secondary special", "Higher education", "Incomplete higher"],
                n, p=[0.71, 0.24, 0.05],
            ),
            "NAME_FAMILY_STATUS": rng.choice(
                ["Married", "Single / not married", "Civil marriage", "Widow", "Separated"],
                n, p=[0.64, 0.18, 0.10, 0.04, 0.04],
            ),
            "DAYS_BIRTH": (-age_years * 365.25).round().astype(int),
            "DAYS_EMPLOYED": (-employed_years * 365.25).round().astype(int),
            "EXT_SOURCE_1": ext1.round(4),
            "EXT_SOURCE_2": ext2.round(4),
            "EXT_SOURCE_3": ext3.round(4),
            "REGION_RATING_CLIENT": rng.choice([1, 2, 3], n, p=[0.16, 0.74, 0.10]),
        }
    )
    # realistic missingness in EXT_SOURCE_1 (real col is ~56% null)
    miss = rng.uniform(size=n) < 0.45
    app.loc[miss, "EXT_SOURCE_1"] = np.nan

    bureau = _make_bureau(sk_id, target, rng)
    prev = _make_previous_application(sk_id, target, rng)
    return {"application_train": app, "bureau": bureau, "previous_application": prev}


def _make_bureau(sk_id: np.ndarray, target: np.ndarray, rng) -> pd.DataFrame:
    rows = []
    bureau_id = 600_000
    for sid, tgt in zip(sk_id, target):
        for _ in range(rng.integers(0, 9)):  # number of prior bureau credits
            # Overdue is mostly ZERO (as in real bureau data); defaulters are only
            # slightly more likely to carry a balance -> weak, capped, noisy signal.
            has_overdue = rng.uniform() < (0.10 if tgt else 0.07)
            overdue = min(rng.exponential(2_500), 30_000) if has_overdue else 0.0
            rows.append(
                {
                    "SK_ID_CURR": sid,
                    "SK_ID_BUREAU": bureau_id,
                    "CREDIT_ACTIVE": rng.choice(["Closed", "Active", "Sold"], p=[0.6, 0.37, 0.03]),
                    "DAYS_CREDIT": -int(rng.integers(30, 2900)),
                    "CREDIT_DAY_OVERDUE": int(max(0, rng.normal(9 if tgt else 5, 28))),
                    "AMT_CREDIT_SUM": round(float(rng.lognormal(11, 1.0)), 2),
                    "AMT_CREDIT_SUM_DEBT": round(float(rng.lognormal(9.5, 1.4)), 2),
                    "AMT_CREDIT_SUM_OVERDUE": round(float(overdue), 2),
                    "CREDIT_TYPE": rng.choice(
                        ["Consumer credit", "Credit card", "Car loan", "Mortgage"],
                        p=[0.55, 0.3, 0.1, 0.05],
                    ),
                }
            )
            bureau_id += 1
    return pd.DataFrame(rows)


def _make_previous_application(sk_id: np.ndarray, target: np.ndarray, rng) -> pd.DataFrame:
    rows = []
    prev_id = 800_000
    for sid, tgt in zip(sk_id, target):
        for _ in range(rng.integers(0, 6)):
            refused = rng.uniform() < (0.22 if tgt else 0.15)  # defaulters refused a bit more
            rows.append(
                {
                    "SK_ID_CURR": sid,
                    "SK_ID_PREV": prev_id,
                    "NAME_CONTRACT_STATUS": "Refused" if refused else
                    rng.choice(["Approved", "Canceled", "Unused offer"], p=[0.85, 0.1, 0.05]),
                    "AMT_APPLICATION": round(float(rng.lognormal(12, 0.8)), 2),
                    "AMT_CREDIT": round(float(rng.lognormal(12, 0.8)), 2),
                    "DAYS_DECISION": -int(rng.integers(30, 2900)),
                    "NAME_YIELD_GROUP": rng.choice(
                        ["low_normal", "middle", "high", "XNA"], p=[0.3, 0.3, 0.25, 0.15]
                    ),
                }
            )
            prev_id += 1
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# ULB credit-card fraud  (anchor for fraud detection)
#   real: 284,807 tx, 492 frauds (0.172%), cols Time, V1..V28, Amount, Class.
# ---------------------------------------------------------------------------
def make_ulb_fraud(n: int = 60_000, fraud_rate: float = 0.0035, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_fraud = max(20, int(n * fraud_rate))
    n_legit = n - n_fraud

    # 28 PCA-like components; a handful separate the classes (as in the real set)
    def block(count, shift):
        X = rng.normal(0, 1, size=(count, 28))
        for j in (0, 2, 3, 9, 10, 13, 16):  # mirror real high-importance Vs
            X[:, j] += shift * rng.uniform(0.8, 1.6)
        return X

    Xl = block(n_legit, shift=0.0)
    # Frauds pushed off the legit manifold with heavy marginal overlap (each V is
    # only weakly predictive on its own). A GBM combining them lands ~0.78 AUPRC /
    # ~0.98 AUC -- realistic ULB territory, and far below the trivially-separable
    # regime, so the imbalance metrics (AUPRC/recall) carry the story, not accuracy.
    Xf = block(n_fraud, shift=-1.4)

    X = np.vstack([Xl, Xf])
    y = np.r_[np.zeros(n_legit, int), np.ones(n_fraud, int)]
    amount = np.r_[
        rng.lognormal(3.0, 1.2, n_legit),
        rng.lognormal(2.0, 1.6, n_fraud),  # frauds skew to smaller, odd amounts
    ].clip(0, 25_000)

    df = pd.DataFrame(X, columns=[f"V{i}" for i in range(1, 29)])
    df["Amount"] = amount.round(2)
    df["Class"] = y
    # Shuffle FIRST, then assign a sorted Time column to the shuffled rows. This
    # keeps Time chronological (like the real ULB set) WITHOUT it leaking class
    # order -- assigning sorted Time before the shuffle would make fraud (always
    # appended last) carry the largest Time values, a perfect giveaway.
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    df.insert(0, "Time", np.sort(rng.uniform(0, 172_792, size=len(df))))
    return df


# ---------------------------------------------------------------------------
# PaySim  (synthetic mobile-money -- used only for the "scale + Spark" story)
#   real: 6,362,620 tx, 8,213 fraud (0.129%).
# ---------------------------------------------------------------------------
def make_paysim(n: int = 100_000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    types = rng.choice(
        ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"],
        n, p=[0.34, 0.22, 0.30, 0.04, 0.10],
    )
    amount = rng.lognormal(8.0, 1.3, n).clip(1, 1e7).round(2)
    old_org = rng.lognormal(9.0, 1.5, n).round(2)
    new_org = (old_org - amount).clip(0).round(2)
    old_dst = rng.lognormal(8.5, 1.6, n).round(2)
    new_dst = (old_dst + amount).round(2)

    # fraud only happens in TRANSFER/CASH_OUT and tends to drain the account
    is_fraud = np.zeros(n, int)
    risky = np.isin(types, ["TRANSFER", "CASH_OUT"])
    drain = (amount > old_org * 0.95) & risky
    flag = drain & (rng.uniform(size=n) < 0.25)
    is_fraud[flag] = 1

    return pd.DataFrame(
        {
            "step": rng.integers(1, 744, n),
            "type": types,
            "amount": amount,
            "nameOrig": [f"C{int(x)}" for x in rng.integers(1e8, 9e8, n)],
            "oldbalanceOrg": old_org,
            "newbalanceOrig": new_org,
            "nameDest": [f"C{int(x)}" for x in rng.integers(1e8, 9e8, n)],
            "oldbalanceDest": old_dst,
            "newbalanceDest": new_dst,
            "isFraud": is_fraud,
            "isFlaggedFraud": ((amount > 200_000) & risky).astype(int),
        }
    )


# ---------------------------------------------------------------------------
def _save(frames: dict[str, pd.DataFrame], subdir: str) -> None:
    out = CFG.raw / subdir
    out.mkdir(parents=True, exist_ok=True)
    for name, df in frames.items():
        path = out / f"{name}.csv"
        df.to_csv(path, index=False)
        pos = ""
        for tcol in ("TARGET", "Class", "isFraud"):
            if tcol in df.columns:
                pos = f"  (positive rate {df[tcol].mean():.4f})"
        print(f"  wrote {rel_to_root(path)}  [{len(df):,} x {df.shape[1]}]{pos}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate synthetic datasets.")
    ap.add_argument("--all", action="store_true", help="generate everything")
    ap.add_argument("--home-credit", action="store_true")
    ap.add_argument("--fraud", action="store_true", help="ULB credit-card fraud")
    ap.add_argument("--paysim", action="store_true")
    ap.add_argument("--n", type=int, default=None, help="override row count")
    ap.add_argument("--seed", type=int, default=CFG.seed)
    args = ap.parse_args()

    set_global_seed(args.seed)
    do_all = args.all or not (args.home_credit or args.fraud or args.paysim)

    if args.home_credit or do_all:
        print("Home Credit (synthetic):")
        _save(make_home_credit(n=args.n or 20_000, seed=args.seed), "home_credit")
    if args.fraud or do_all:
        print("ULB credit-card fraud (synthetic):")
        _save({"creditcard": make_ulb_fraud(n=args.n or 60_000, seed=args.seed)}, "fraud")
    if args.paysim or do_all:
        print("PaySim (synthetic):")
        _save({"paysim": make_paysim(n=args.n or 100_000, seed=args.seed)}, "paysim")

    print("\nDone. These are SYNTHETIC. See src/data/download.py for the real datasets.")


if __name__ == "__main__":
    main()
