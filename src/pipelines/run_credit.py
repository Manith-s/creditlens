"""End-to-end credit-risk pipeline (the Workstream 1 deliverable).

    features -> IV ranking -> WOE scorecard (baseline) -> GBM -> SHAP reason codes

Everything is logged to MLflow (or printed if no server). Leakage guardrails:
split BEFORE any binning/scaling; binning fit on TRAIN only; stratified split;
fixed seed.

    python -m src.pipelines.run_credit
"""
from __future__ import annotations

import json

import numpy as np
from sklearn.model_selection import train_test_split

from src.common.metrics import classification_metrics
from src.common.tracking import mlflow_run
from src.config import CFG, set_global_seed
from src.explain.shap_codes import adverse_action_notice
from src.features.build_features import MODEL_FEATURES, build_credit_features
from src.features.woe import information_value_ranking
from src.models.gbm import predict_proba, train_gbm
from src.models.scorecard import Scorecard, ScorecardConfig

TARGET = "TARGET"


def run() -> dict:
    set_global_seed()
    print("[1/6] Building features (pandas mirror of the Spark pipeline)…")
    df = build_credit_features(save=False)
    feats = [f for f in MODEL_FEATURES if f in df.columns]
    df = df[[TARGET, *feats]].copy()

    # ---- split FIRST (leakage guard) ----
    train, test = train_test_split(
        df, test_size=0.25, stratify=df[TARGET], random_state=CFG.seed
    )
    print(f"      train={len(train):,}  test={len(test):,}  default_rate={df[TARGET].mean():.3f}")

    # ---- IV ranking + selection (fit on train) ----
    print("[2/6] Information Value ranking…")
    iv = information_value_ranking(train, TARGET, feats, iv_floor=0.02)
    selected = iv.loc[iv["selected"], "feature"].tolist() or feats
    print(iv.head(10).to_string(index=False))
    iv_path = CFG.artifacts / "iv_ranking.csv"
    iv.to_csv(iv_path, index=False)

    results: dict[str, dict] = {}

    # ---- baseline: WOE scorecard / logistic ----
    print("[3/6] Fitting WOE scorecard (logistic baseline)…")
    sc = Scorecard(features=selected, cfg=ScorecardConfig())
    sc.fit(train, TARGET)
    p_sc = sc.predict_proba(test)
    m_sc = classification_metrics(test[TARGET].values, p_sc)
    results["scorecard"] = m_sc
    print(f"      scorecard backend={sc.backend}  AUC={m_sc['auc']:.4f}  KS={m_sc['ks']:.4f}")
    with mlflow_run(CFG.experiment_credit, "scorecard-logistic") as rec:
        rec.log_params({"backend": sc.backend, "n_features": len(selected),
                        "iv_floor": 0.02, "model": "woe-logistic-scorecard"})
        rec.log_metrics(m_sc)
        rec.log_artifact(str(iv_path))

    # ---- GBM ----
    print("[4/6] Training gradient boosting…")
    Xtr = train[selected].fillna(train[selected].median(numeric_only=True))
    Xte = test[selected].fillna(train[selected].median(numeric_only=True))
    gbm = train_gbm(Xtr, train[TARGET].values, handle_imbalance=False, random_state=CFG.seed)
    p_gbm = predict_proba(gbm, Xte)
    m_gbm = classification_metrics(test[TARGET].values, p_gbm)
    results["gbm"] = m_gbm
    backend = getattr(gbm, "_creditlens_backend", "gbm")
    print(f"      gbm backend={backend}  AUC={m_gbm['auc']:.4f}  KS={m_gbm['ks']:.4f}")

    lift = m_gbm["auc"] - m_sc["auc"]
    with mlflow_run(CFG.experiment_credit, f"gbm-{backend}") as rec:
        rec.log_params({"backend": backend, "n_features": len(selected),
                        "handle_imbalance": False, "model": "gradient-boosting"})
        rec.log_metrics(m_gbm)
        rec.log_metrics({"auc_lift_over_scorecard": lift})

    # ---- SHAP adverse-action notice on a high-risk applicant ----
    print("[5/6] Generating a sample adverse-action notice (SHAP)…")
    hi = int(np.argmax(p_gbm))  # most-likely-default applicant in the test set
    notice = adverse_action_notice(gbm, Xte.reset_index(drop=True), hi, applicant_id=f"TEST-{hi}")
    notice_path = CFG.artifacts / "sample_adverse_action_notice.txt"
    notice_path.write_text(notice, encoding="utf-8")

    # ---- summary ----
    print("[6/6] Summary")
    summary = {
        "scorecard_auc": round(m_sc["auc"], 4),
        "gbm_auc": round(m_gbm["auc"], 4),
        "auc_lift": round(lift, 4),
        "gbm_backend": backend,
        "n_selected_features": len(selected),
        "default_rate": round(float(df[TARGET].mean()), 4),
    }
    (CFG.artifacts / "credit_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print("\n--- sample adverse-action notice ---\n" + notice)
    return {"metrics": results, "summary": summary}


if __name__ == "__main__":
    run()
