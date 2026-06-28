"""Dataset loaders that transparently prefer REAL data and fall back to synthetic.

Both layouts share column names, so downstream code is identical whether the user
ran `make synth` or downloaded the genuine Kaggle files.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import CFG


def _first_existing(*candidates: Path) -> Path | None:
    for c in candidates:
        if c.exists():
            return c
    return None


def load_home_credit_application() -> pd.DataFrame:
    """application_train.csv — real or synthetic."""
    path = _first_existing(CFG.raw / "home_credit" / "application_train.csv")
    if path is None:
        raise FileNotFoundError(
            "No application_train.csv found. Run `python -m src.data.make_synthetic --home-credit` "
            "or `python -m src.data.download --home-credit --run`."
        )
    return pd.read_csv(path)


def load_home_credit_auxiliary() -> dict[str, pd.DataFrame]:
    """bureau + previous_application tables, whichever exist."""
    base = CFG.raw / "home_credit"
    out: dict[str, pd.DataFrame] = {}
    for name in ("bureau", "previous_application"):
        p = base / f"{name}.csv"
        if p.exists():
            out[name] = pd.read_csv(p)
    return out


def load_fraud() -> pd.DataFrame:
    """ULB credit-card fraud — real or synthetic. Target column is `Class`."""
    path = _first_existing(CFG.raw / "fraud" / "creditcard.csv")
    if path is None:
        raise FileNotFoundError(
            "No creditcard.csv found. Run `python -m src.data.make_synthetic --fraud` "
            "or `python -m src.data.download --ulb --run`."
        )
    return pd.read_csv(path)
