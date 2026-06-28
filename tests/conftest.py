from __future__ import annotations
import os
import tempfile
os.environ.setdefault("CREDITLENS_DATA_DIR", tempfile.mkdtemp(prefix="creditlens-test-"))
import pytest  # noqa: E402
from src.data import make_synthetic  # noqa: E402
from src.data.make_policy_corpus import main as make_corpus  # noqa: E402

@pytest.fixture(scope="session", autouse=True)
def _bootstrap_data():
    make_synthetic.set_global_seed(42)
    hc = make_synthetic.make_home_credit(n=4000, seed=42)
    make_synthetic._save(hc, "home_credit")
    make_synthetic._save({"creditcard": make_synthetic.make_ulb_fraud(n=12000, seed=42)}, "fraud")
    make_corpus()
    yield

@pytest.fixture(scope="session")
def home_credit_features():
    from src.features.build_features import build_credit_features
    return build_credit_features(save=False)

@pytest.fixture(scope="session")
def fraud_df():
    from src.common.io import load_fraud
    return load_fraud()
