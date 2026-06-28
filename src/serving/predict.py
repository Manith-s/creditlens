"""Load the registered fraud champion from MLflow and score transactions.

Two modes:
  * In-process: `load_champion()` returns an mlflow pyfunc you can call directly.
  * Over HTTP: `score_via_rest()` posts to a running `mlflow models serve` endpoint,
    mirroring how a real service would call the deployed model.

Deployment workflow this demonstrates:
    train -> register -> promote (@champion) -> serve (/invocations) -> monitor
Containerization (optional, to show breadth):
    mlflow models build-docker -m "models:/fraud-champion@champion" -n fraud-svc
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.config import CFG

CHAMPION_URI = "models:/fraud-champion@champion"


def load_champion(model_uri: str = CHAMPION_URI):
    """Return an mlflow pyfunc model. Requires the registry-backed server running."""
    import mlflow

    mlflow.set_tracking_uri(CFG.mlflow_tracking_uri)
    return mlflow.pyfunc.load_model(model_uri)


def predict(df: pd.DataFrame, model_uri: str = CHAMPION_URI) -> np.ndarray:
    model = load_champion(model_uri)
    out = model.predict(df)
    return np.asarray(out).ravel()


def score_via_rest(df: pd.DataFrame, url: str = "http://127.0.0.1:5001/invocations") -> dict:
    """POST a dataframe_split payload to a running `mlflow models serve` endpoint."""
    import urllib.request

    payload = json.dumps(
        {"dataframe_split": {"columns": list(df.columns), "data": df.values.tolist()}}
    ).encode()
    req = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


if __name__ == "__main__":
    # demo: score the first few rows of the fraud set against the live endpoint
    from src.common.io import load_fraud

    sample = load_fraud().drop(columns=["Class"]).head(3)
    try:
        print(score_via_rest(sample))
    except Exception as e:  # noqa: BLE001
        print(f"No live endpoint ({e}). Start one with:\n"
              "  mlflow models serve -m 'models:/fraud-champion@champion' -p 5001 --no-conda")
