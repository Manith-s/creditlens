"""SHAP-driven adverse-action reason codes (ECOA / Regulation B).

Pipeline:
  1. TreeExplainer on the GBM -> per-applicant feature contributions.
  2. Aggregate correlated features into reason GROUPS (so one underlying cause
     isn't double-counted) using FEATURE_GROUPS from build_features.
  3. Rank the groups that pushed the applicant TOWARD default (positive SHAP on a
     default=1 model) and map them to plain-language reasons.
  4. Render a sample adverse-action notice with up to N reasons.

Regulatory basis (cite in your write-up, note guidance is evolving):
  * CFPB Circular 2022-03: creditors may not use models so complex they cannot give
    specific, accurate adverse-action reasons.
  * CFPB Circular 2023-03: the sample-form reason checklist may not be used if those
    reasons don't accurately describe the principal reason(s).

A pure-magnitude fallback (no shap installed) keeps the demo runnable.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.features.build_features import FEATURE_GROUPS

# Plain-language adverse-action reasons keyed by feature group.
REASON_TEXT: dict[str, str] = {
    "External credit scores": "Credit bureau / external risk scores below approval threshold",
    "Debt-to-income": "Debt obligations are high relative to income",
    "Income": "Income is insufficient for the requested credit amount",
    "Age & employment": "Length of employment / credit history is limited",
    "Prior bureau delinquency": "Record of past-due balances on existing credit",
    "Prior application history": "Recent declined or withdrawn credit applications",
    "Household": "Number of dependents relative to income",
    "Region": "Higher-risk region rating for the application",
}


@dataclass
class ReasonCode:
    group: str
    reason: str
    contribution: float  # signed SHAP contribution toward default


def _shap_values(model, X: pd.DataFrame) -> np.ndarray:
    """Return an (n_samples, n_features) array of SHAP values for class=default."""
    try:
        import shap

        explainer = shap.TreeExplainer(model)
        vals = explainer.shap_values(X)
        if isinstance(vals, list):  # some versions return [class0, class1]
            vals = vals[1]
        vals = np.asarray(vals)
        if vals.ndim == 3:  # (n, features, classes)
            vals = vals[:, :, 1]
        return vals
    except Exception:  # noqa: BLE001 — fall back to a model-agnostic proxy
        # proxy: standardized feature deviation * model feature_importance sign-free
        Xz = (X - X.mean()) / (X.std(ddof=0).replace(0, 1))
        imp = getattr(model, "feature_importances_", np.ones(X.shape[1]))
        return Xz.values * imp


def reason_codes_for(
    model,
    X: pd.DataFrame,
    row_index: int,
    feature_groups: dict[str, list[str]] | None = None,
    top_n: int = 4,
) -> list[ReasonCode]:
    groups = feature_groups or FEATURE_GROUPS
    vals = _shap_values(model, X)
    row = vals[row_index]
    contrib = dict(zip(X.columns, row))

    grouped: list[ReasonCode] = []
    for group, cols in groups.items():
        present = [c for c in cols if c in contrib]
        if not present:
            continue
        total = float(np.sum([contrib[c] for c in present]))
        grouped.append(ReasonCode(group=group, reason=REASON_TEXT.get(group, group), contribution=total))

    # keep only groups pushing TOWARD default (positive), most influential first
    pushing = [g for g in grouped if g.contribution > 0]
    pushing.sort(key=lambda g: g.contribution, reverse=True)
    return pushing[:top_n]


def render_notice(
    reasons: list[ReasonCode],
    applicant_id: str | int = "—",
    decision: str = "denied",
) -> str:
    lines = [
        "NOTICE OF ADVERSE ACTION",
        "(Equal Credit Opportunity Act / Regulation B)",
        "",
        f"Applicant reference: {applicant_id}",
        f"Decision: Application {decision}.",
        "",
        "Principal reason(s) for the credit decision:",
    ]
    if not reasons:
        lines.append("  (No factor materially increased the assessed risk.)")
    for i, r in enumerate(reasons, 1):
        lines.append(f"  {i}. {r.reason}")
    lines += [
        "",
        "These reasons reflect the factors that most increased the assessed risk for",
        "this application, derived from the model's per-applicant explanations.",
        "You have the right to a statement of specific reasons; this notice provides it.",
    ]
    return "\n".join(lines)


def adverse_action_notice(model, X: pd.DataFrame, row_index: int, applicant_id="—", top_n=4) -> str:
    reasons = reason_codes_for(model, X, row_index, top_n=top_n)
    return render_notice(reasons, applicant_id=applicant_id)
