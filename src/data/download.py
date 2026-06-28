"""Fetch the REAL public datasets. Synthetic generators (make_synthetic.py) let
everything run offline; use this when you want the headline, resume-grade numbers.

Nothing here is auto-run destructively: each dataset prints exact commands and,
where an API exists (Kaggle), can execute the download for you with --run.

    python -m src.data.download                 # list everything + instructions
    python -m src.data.download --home-credit --run
    python -m src.data.download --ulb --run
    python -m src.data.download --policy --run   # CFPB/Fed public-domain PDFs

Kaggle datasets require ~/.kaggle/kaggle.json (Account -> Create New API Token)
and accepting each competition's rules on the website first.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import urllib.request
from pathlib import Path

from src.config import CFG, rel_to_root

KAGGLE = {
    "home-credit": {
        "kind": "competition",
        "slug": "home-credit-default-risk",
        "dest": "home_credit",
        "note": "307,511 x 122, TARGET ~8%. Accept rules at the competition page first.",
    },
    "ulb": {
        "kind": "dataset",
        "slug": "mlg-ulb/creditcardfraud",
        "dest": "fraud",
        "note": "284,807 tx, 492 frauds (0.172%). cols Time, V1..V28, Amount, Class.",
    },
    "paysim": {
        "kind": "dataset",
        "slug": "ealaxi/paysim1",
        "dest": "paysim",
        "note": "6,362,620 tx, 8,213 fraud (0.129%). Synthetic mobile money.",
    },
    "ieee-cis": {
        "kind": "competition",
        "slug": "ieee-fraud-detection",
        "dest": "ieee_cis",
        "note": "590,540 tx, 20,663 frauds (3.5%), 431 features. Heavy FE + GBM-vs-NN.",
    },
    "give-me-credit": {
        "kind": "competition",
        "slug": "GiveMeSomeCredit",
        "dest": "give_me_credit",
        "note": "150,000 borrowers, 11 numeric features. Small WOE/IV auxiliary.",
    },
}

# CFPB / Federal Reserve documents are U.S. government PUBLIC DOMAIN — license-clean
# RAG content. URLs occasionally move; if one 404s, search the title on cfpb.gov.
POLICY_PDFS = {
    "cfpb_supervision_exam_manual_ecoa.pdf":
        "https://files.consumerfinance.gov/f/documents/cfpb_ecoa_narrative-and-procedures_2023-09.pdf",
    "cfpb_tila_reg_z_procedures.pdf":
        "https://files.consumerfinance.gov/f/documents/cfpb_tila_narrative-and-procedures_2023-03.pdf",
    "cfpb_charm_booklet.pdf":
        "https://files.consumerfinance.gov/f/documents/cfpb_charm-booklet.pdf",
    "cfpb_heloc_booklet.pdf":
        "https://files.consumerfinance.gov/f/documents/cfpb_heloc-booklet.pdf",
}

FREDDIE_MAC_GUIDE = """
Freddie Mac Single-Family Loan-Level Dataset  (the honest path to "3M+ records")
--------------------------------------------------------------------------------
~55 million mortgages originated 1999-01..2025-09; monthly performance rows in
the hundreds of millions. There is NO direct download API.

  1. Register (free) at: https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset
     (portal is operated via the EMBS site).
  2. Accept the academic/research/limited-use license. NOTE: commercial
     redistribution is NOT permitted — keep this data out of git.
  3. Download either the free "representative sample" (re-created each refresh)
     or full Standard Dataset quarters (one Origination + one Performance file
     each, from Q1 1999). Requires Java.
  4. Unzip into  data/raw/freddie_mac/  then point src/features/spark_features.py
     at it (it auto-detects either Home Credit or Freddie layout).

For a laptop: a few vintages/quarters already exceed 3M monthly performance rows,
which honestly satisfies the "processed 3M+ records with PySpark" claim.
"""


def _have_kaggle() -> bool:
    try:
        subprocess.run(["kaggle", "--version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def download_kaggle(key: str, run: bool) -> None:
    spec = KAGGLE[key]
    dest: Path = CFG.raw / spec["dest"]
    if spec["kind"] == "competition":
        cmd = ["kaggle", "competitions", "download", "-c", spec["slug"], "-p", str(dest)]
    else:
        cmd = ["kaggle", "datasets", "download", "-d", spec["slug"], "-p", str(dest), "--unzip"]
    print(f"\n[{key}] {spec['note']}")
    print("  " + " ".join(cmd))
    if run:
        if not _have_kaggle():
            print("  ! kaggle CLI not found / not configured. `pip install kaggle` and add kaggle.json.")
            return
        dest.mkdir(parents=True, exist_ok=True)
        subprocess.run(cmd, check=False)


def download_policy(run: bool) -> None:
    dest = CFG.policy_corpus
    print(f"\n[policy] CFPB/Fed public-domain PDFs -> {rel_to_root(dest)}")
    for name, url in POLICY_PDFS.items():
        print(f"  {name}  <-  {url}")
        if run:
            dest.mkdir(parents=True, exist_ok=True)
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=60) as r, open(dest / name, "wb") as f:
                    f.write(r.read())
                print(f"    saved {name}")
            except Exception as e:  # noqa: BLE001
                print(f"    ! failed ({e}); fetch manually from the URL above.")
    if not run:
        print("  (re-run with --run to download; or use src/data/make_policy_corpus.py for a synthetic corpus)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Download the real public datasets.")
    for k in KAGGLE:
        ap.add_argument(f"--{k}", action="store_true")
    ap.add_argument("--policy", action="store_true", help="CFPB/Fed policy PDFs for RAG")
    ap.add_argument("--freddie", action="store_true", help="print Freddie Mac instructions")
    ap.add_argument("--run", action="store_true", help="actually execute downloads")
    args = ap.parse_args()

    selected = [k for k in KAGGLE if getattr(args, k.replace("-", "_"))]
    if not selected and not args.policy and not args.freddie:
        print(__doc__)
        print("Available Kaggle datasets:")
        for k, v in KAGGLE.items():
            print(f"  --{k:16s} {v['note']}")
        print(FREDDIE_MAC_GUIDE)
        download_policy(run=False)
        sys.exit(0)

    for k in selected:
        download_kaggle(k, run=args.run)
    if args.policy:
        download_policy(run=args.run)
    if args.freddie:
        print(FREDDIE_MAC_GUIDE)


if __name__ == "__main__":
    main()
