"""Functional fixtures use cached original UCI Adult observations."""
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture(scope="session")
def adult():
    from census_income.data import load_adult
    return load_adult(download=False)[0]


@pytest.fixture(scope="session")
def small_training(adult):
    from sklearn.model_selection import train_test_split
    from census_income.data import encode_target
    X = adult.drop(columns="income")
    y = encode_target(adult["income"])
    X, _, y, _ = train_test_split(X, y, train_size=700, random_state=42, stratify=y)
    return X, y


@pytest.fixture(scope="session")
def fitted_pipeline(small_training):
    from census_income.preprocessing import make_pipeline
    X, y = small_training
    return make_pipeline("Expanded + interaction", n_estimators=8, max_depth=5, n_jobs=1).fit(X, y)


@pytest.fixture
def valid_record(adult):
    from census_income.config import MODEL_INPUTS
    return adult.loc[:, MODEL_INPUTS].dropna().iloc[[0]].copy()

