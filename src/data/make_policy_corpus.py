"""Generate a small SYNTHETIC internal-policy corpus + a golden Q&A set so the
RAG assistant runs offline without downloading the CFPB/Fed PDFs.

The text paraphrases real, public regulatory concepts (ECOA/Reg B adverse-action
reasons, TILA/Reg Z disclosures, fraud monitoring) into a fictional company
handbook. For the resume-grade build, also run:
    python -m src.data.download --policy --run
to add the genuine CFPB public-domain PDFs alongside these.

Usage:  python -m src.data.make_policy_corpus
"""
from __future__ import annotations

import json

from src.config import CFG, rel_to_root

DOCS: dict[str, str] = {
    "01_adverse_action_policy.md": """# Adverse Action Notice Policy (ECOA / Regulation B)

## Purpose
This policy governs how Northwind Bank issues adverse action notices when a credit
application is denied, an existing account is terminated, or terms are changed
unfavorably. It implements the Equal Credit Opportunity Act (ECOA) and Regulation B.

## Timing
A notice of adverse action must be sent within 30 days of receiving a completed
application. For incomplete applications, a notice of incompleteness or adverse
action must be sent within 30 days.

## Specific and accurate reasons
Each notice must state the specific principal reasons for the adverse action.
Generic statements such as "internal scoring" or "does not meet standards" are
prohibited. When a credit-scoring model drives the decision, the reasons disclosed
must reflect the actual factors that most reduced the applicant's score, ranked by
their contribution. If a model is too complex to extract specific reasons, it must
not be used for that decision. The checklist of sample reasons in the model forms
may only be used when those reasons accurately describe the principal reasons.

## Reason codes derived from model explanations
For machine-learning models, reason codes are produced from per-applicant feature
attributions (for example SHAP values). The top negative contributors are mapped
to plain-language reasons. Correlated features are grouped so a single underlying
cause is not double-counted. The maximum number of reasons disclosed is four.

## Record retention
Copies of adverse action notices and the supporting reason-code computations are
retained for 25 months for consumer credit and 12 months for business credit.
""",
    "02_credit_scoring_governance.md": """# Credit Scoring Model Governance

## Model lifecycle
All credit scoring models follow a documented lifecycle: development, validation,
approval, deployment, monitoring, and retirement. Every model has a registered
owner and a version recorded in the model registry.

## Validation requirements
Before approval, a model must demonstrate discrimination performance using ROC-AUC,
Kolmogorov-Smirnov (KS) statistic, and Gini. The minimum acceptable ROC-AUC for a
production application scorecard is 0.72. Calibration is assessed and must not drift
by more than 10 percent year over year.

## Information Value and feature selection
Candidate features are screened with Information Value (IV). Features with IV below
0.02 are considered non-predictive and are excluded. Weight of Evidence (WOE)
binning must be monotonic with respect to the target where a monotonic relationship
is expected; non-monotonic bins require documented justification.

## Scorecard scaling
Points-based scorecards use Points to Double the Odds (PDO) scaling. The standard
configuration is 600 points at 50:1 odds with a PDO of 20. Scaling parameters are
recorded with the model version.

## Fairness and disparate impact
Models are tested for disparate impact across protected classes. A model may not
use prohibited bases (race, color, religion, national origin, sex, marital status,
age) as inputs, and proxies for those bases are reviewed and removed.
""",
    "03_fraud_monitoring_standard.md": """# Transaction Fraud Monitoring Standard

## Scope
This standard covers real-time and near-real-time monitoring of card and
mobile-money transactions for fraud.

## Class imbalance and metrics
Fraud is rare (well under 1 percent of transactions). Accuracy is not an acceptable
performance metric because a model that predicts "not fraud" for every transaction
would appear over 99 percent accurate. Models are evaluated using the area under the
precision-recall curve (AUPRC), recall at a fixed alert budget, F1, and Matthews
correlation coefficient (MCC).

## Leakage controls
Resampling techniques such as SMOTE must be applied only within the training folds
of a cross-validation pipeline, never before the train/test split. Any feature that
encodes information unavailable at scoring time is prohibited.

## Champion / challenger
The production model is the champion. New candidates run as challengers in shadow
mode. Promotion from challenger to champion requires a statistically significant
improvement in AUPRC on a holdout period and sign-off by the model risk team.

## Alerting and thresholds
Alert thresholds are set against a fixed analyst review capacity. The score
threshold is tuned to maximize recall within that capacity rather than to a fixed
probability cutoff.
""",
    "04_tila_reg_z_disclosures.md": """# TILA / Regulation Z Disclosure Policy

## Purpose
This policy implements the Truth in Lending Act (TILA) and Regulation Z, which
require clear disclosure of the cost of consumer credit.

## Annual Percentage Rate
The Annual Percentage Rate (APR) must be disclosed as the cost of credit expressed
as a yearly rate. For closed-end credit, the APR, finance charge, amount financed,
and total of payments must be disclosed before consummation.

## Adjustable-rate mortgages
For adjustable-rate mortgage products, the consumer must receive the Consumer
Handbook on Adjustable-Rate Mortgages (the CHARM booklet) and a disclosure of the
index, margin, and how the rate and payment can change.

## Right of rescission
For certain transactions secured by the consumer's principal dwelling, the consumer
has a right to rescind within three business days.
""",
}

# Golden Q&A set for RAG evaluation (Recall@k, MRR, RAGAS faithfulness, etc.).
# `ground_truth` is a reference answer; `source` names the doc that should be retrieved.
GOLDEN_QA = [
    {
        "question": "Within how many days must an adverse action notice be sent after a completed application?",
        "ground_truth": "Within 30 days of receiving a completed application.",
        "source": "01_adverse_action_policy.md",
    },
    {
        "question": "Can a creditor use a credit model that is too complex to produce specific reasons for denial?",
        "ground_truth": "No. If a model is too complex to extract specific principal reasons, it must not be used for that decision under ECOA/Regulation B.",
        "source": "01_adverse_action_policy.md",
    },
    {
        "question": "What is the maximum number of reasons disclosed on an adverse action notice?",
        "ground_truth": "Up to four reasons.",
        "source": "01_adverse_action_policy.md",
    },
    {
        "question": "What is the minimum acceptable ROC-AUC for a production application scorecard?",
        "ground_truth": "0.72.",
        "source": "02_credit_scoring_governance.md",
    },
    {
        "question": "What Information Value threshold is used to exclude non-predictive features?",
        "ground_truth": "Features with IV below 0.02 are excluded.",
        "source": "02_credit_scoring_governance.md",
    },
    {
        "question": "What PDO scaling configuration is standard for the scorecard?",
        "ground_truth": "600 points at 50:1 odds with a PDO of 20.",
        "source": "02_credit_scoring_governance.md",
    },
    {
        "question": "Why is accuracy not used to evaluate fraud models?",
        "ground_truth": "Because fraud is extremely rare, a model predicting 'not fraud' for everything would look over 99% accurate; AUPRC, recall, F1, and MCC are used instead.",
        "source": "03_fraud_monitoring_standard.md",
    },
    {
        "question": "When may SMOTE be applied in the fraud modeling pipeline?",
        "ground_truth": "Only within the training folds of a cross-validation pipeline, never before the train/test split.",
        "source": "03_fraud_monitoring_standard.md",
    },
    {
        "question": "What must a creditor disclose as the yearly cost of credit under Regulation Z?",
        "ground_truth": "The Annual Percentage Rate (APR).",
        "source": "04_tila_reg_z_disclosures.md",
    },
    {
        "question": "What booklet must be provided for adjustable-rate mortgages?",
        "ground_truth": "The Consumer Handbook on Adjustable-Rate Mortgages (the CHARM booklet).",
        "source": "04_tila_reg_z_disclosures.md",
    },
]


def main() -> None:
    dest = CFG.policy_corpus
    dest.mkdir(parents=True, exist_ok=True)
    for name, text in DOCS.items():
        (dest / name).write_text(text, encoding="utf-8")
        print(f"  wrote {rel_to_root(dest / name)}")
    qa_path = CFG.raw / "policy_golden_qa.json"
    qa_path.write_text(json.dumps(GOLDEN_QA, indent=2), encoding="utf-8")
    print(f"  wrote {rel_to_root(qa_path)}  [{len(GOLDEN_QA)} golden Q&A]")


if __name__ == "__main__":
    main()
