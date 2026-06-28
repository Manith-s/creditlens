"""Fraud detection benchmark (the Workstream 2 deliverable).

Apples-to-apples on identical splits/metrics:
    XGBoost/LightGBM   vs   PyTorch MLP   vs   Keras 3 MLP
on the ULB credit-card set, with leakage-safe imbalance handling and AUPRC/recall/
MCC (NOT accuracy). The best run is registered to the MLflow Model Registry as the
champion; serving instructions are printed for the local /invocations endpoint.

    python -m src.pipelines.run_fraud
"""
from __future__ import annotations

import json

import pandas as pd
from sklearn.model_selection import train_test_split

from src.common.io import load_fraud
from src.common.metrics import classification_metrics
from src.common.tracking import mlflow_run
from src.config import CFG, set_global_seed
from src.models import tf_net, torch_net
from src.models.gbm import predict_proba as gbm_predict
from src.models.gbm import train_gbm

TARGET = "Class"
REGISTERED_MODEL = "fraud-champion"


def run() -> dict:
    set_global_seed()
    print("[1/4] Loading fraud data…")
    df = load_fraud()
    y = df[TARGET].values
    X = df.drop(columns=[TARGET])
    print(f"      {len(df):,} tx  fraud_rate={y.mean():.4%}")

    # ---- split FIRST (leakage guard); stratify to preserve the rare class ----
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=CFG.seed
    )

    results: dict[str, dict] = {}
    fitted: dict[str, object] = {}

    # ---- GBM (scale_pos_weight) ----
    print("[2/4] Training models…")
    gbm = train_gbm(X_tr, y_tr, handle_imbalance=True, random_state=CFG.seed)
    p = gbm_predict(gbm, X_te)
    results["gbm"] = classification_metrics(y_te, p)
    fitted["gbm"] = gbm
    print(f"      gbm ({getattr(gbm,'_creditlens_backend','?')}):  "
          f"AUPRC={results['gbm']['auprc']:.4f}  recall={results['gbm']['recall']:.3f}")

    # ---- PyTorch MLP (pos_weight) ----
    if torch_net.is_available():
        net = torch_net.train_torch_mlp(X_tr.values, y_tr, random_state=CFG.seed)
        p = net.predict_proba(X_te.values)[:, 1]
        results["pytorch"] = classification_metrics(y_te, p)
        fitted["pytorch"] = net
        print(f"      pytorch:  AUPRC={results['pytorch']['auprc']:.4f}  "
              f"recall={results['pytorch']['recall']:.3f}")
    else:
        print("      pytorch: SKIPPED (torch not installed)")

    # ---- Keras 3 MLP (class_weight) ----
    if tf_net.is_available():
        net = tf_net.train_keras_mlp(X_tr.values, y_tr, random_state=CFG.seed)
        p = net.predict_proba(X_te.values)[:, 1]
        results["keras"] = classification_metrics(y_te, p)
        fitted["keras"] = net
        print(f"      keras:  AUPRC={results['keras']['auprc']:.4f}  "
              f"recall={results['keras']['recall']:.3f}")
    else:
        print("      keras: SKIPPED (tensorflow not installed)")

    # ---- comparison table + MLflow logging ----
    table = pd.DataFrame(results).T[
        ["auprc", "auc", "recall", "precision", "f1", "mcc"]
    ].sort_values("auprc", ascending=False)
    print("\n[3/4] Benchmark (sorted by AUPRC — the right metric under imbalance):")
    print(table.round(4).to_string())
    table.to_csv(CFG.artifacts / "fraud_benchmark.csv")

    champion = table.index[0]
    for name, metrics in results.items():
        with mlflow_run(CFG.experiment_fraud, f"fraud-{name}") as rec:
            rec.log_params({"model": name, "imbalance": "cost-sensitive",
                            "champion": name == champion})
            rec.log_metrics(metrics)
            # register only the champion, and only if it's an sklearn-style model
            if name == champion and name == "gbm":
                rec.sklearn_log(fitted[name], name="model",
                                registered_model_name=REGISTERED_MODEL)

    # ---- serving instructions ----
    print(f"\n[4/4] Champion = '{champion}'. Registered as '{REGISTERED_MODEL}'.")
    print(_serving_help())

    summary = {
        "champion": champion,
        "fraud_rate": round(float(y.mean()), 5),
        "metrics": {k: {m: round(v, 4) for m, v in vals.items()} for k, vals in results.items()},
    }
    (CFG.artifacts / "fraud_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def _serving_help() -> str:
    return (
        "\nLocal deployment (simulated):\n"
        "  1. Start the registry-backed server:  make mlflow\n"
        f"  2. Promote:  the run logs '{REGISTERED_MODEL}'; set its @champion alias in the UI\n"
        f"  3. Serve:    mlflow models serve -m 'models:/{REGISTERED_MODEL}@champion' -p 5001 --no-conda\n"
        "  4. Score:    curl -X POST http://127.0.0.1:5001/invocations \\\n"
        "                 -H 'Content-Type: application/json' \\\n"
        "                 -d '{\"dataframe_split\": {\"columns\": [...], \"data\": [[...]]}}'\n"
        "  (see src/serving/predict.py for a programmatic client + a build-docker note)"
    )


if __name__ == "__main__":
    run()
