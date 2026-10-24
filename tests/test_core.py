"""Regression checks for data provenance, leakage boundaries and measured metrics."""
import hashlib
import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from census_income.data import load_adult, split_data, encode_target, validate_schema
from census_income.features import engineer_features
from census_income.preprocessing import make_pipeline
from census_income.evaluate import metric_summary, threshold_diagnostics, subgroup_metrics


def test_original_schema_cleaning_and_target(adult):
    assert adult.shape[1] == 15
    assert 48000 < len(adult) < 48842
    assert set(adult.income.unique()) == {"<=50K", ">50K"}
    assert not adult.duplicated().any()
    assert adult.isna().sum().sum() > 0
    assert not adult.select_dtypes(exclude="number").apply(lambda col: col.dropna().str.contains(r"^ | $|^\?$", regex=True).any()).any()
    _, audit = load_adult(download=False)
    assert isinstance(audit, dict) and audit


def test_target_encoding_matches_cleaned_uci_labels(adult):
    encoded = encode_target(adult.income)
    assert set(encoded.unique()) == {0, 1}
    np.testing.assert_array_equal(encoded, adult.income.eq(">50K").astype(int))
    labels = adult.income.iloc[:3].copy()
    labels.iloc[0] = "invalid label"
    with pytest.raises(ValueError):
        encode_target(labels)


def test_schema_rejects_missing_column_and_malformed_numeric(adult):
    with pytest.raises(ValueError):
        validate_schema(adult.drop(columns="age"))
    corrupted = adult.iloc[:3].astype(object)
    corrupted.loc[corrupted.index[0], "age"] = "not numeric"
    with pytest.raises(ValueError):
        validate_schema(corrupted)


def test_split_is_reproducible_stratified_and_row_disjoint(adult):
    first = split_data(adult)
    second = split_data(adult)
    for left, right in zip(first, second):
        if isinstance(left, pd.DataFrame):
            pd.testing.assert_frame_equal(left, right)
        else:
            pd.testing.assert_series_equal(left, right)
    Xdev, Xtest, ydev, ytest = first
    assert not set(Xdev.index) & set(Xtest.index)
    assert len(Xdev) + len(Xtest) == len(adult)
    assert len(Xtest) == int(np.ceil(len(adult) * .2))
    assert "income" not in Xdev
    assert Xdev.index.equals(ydev.index) and Xtest.index.equals(ytest.index)
    assert abs(ydev.mean() - ytest.mean()) < .002


def test_engineering_is_record_local_and_does_not_mutate(adult):
    raw = adult.drop(columns="income").iloc[:12].copy()
    before = raw.copy(deep=True)
    batch = engineer_features(raw)
    single = pd.concat([engineer_features(raw.iloc[[i]]) for i in range(len(raw))])
    pd.testing.assert_frame_equal(batch, single)
    pd.testing.assert_frame_equal(raw, before)
    assert len(set(batch.columns) - set(raw.columns)) >= 2
    np.testing.assert_array_equal(batch.hours_age_interaction, raw.age * raw["hours-per-week"])
    expected = pd.cut(raw["education-num"], [0, 9, 13, 16], labels=["HS or less", "College to Bachelors", "Postgraduate"]).astype(object)
    pd.testing.assert_series_equal(batch.education_bin, expected, check_names=False)


def test_pipeline_imputes_and_encodes_unseen_categories(fitted_pipeline, small_training):
    X, _ = small_training
    probe = X.iloc[:3].copy()
    probe.loc[probe.index[0], "age"] = np.nan
    probe.loc[probe.index[1], "occupation"] = "A category absent in training"
    probe.loc[probe.index[2], "workclass"] = np.nan
    probabilities = fitted_pipeline.predict_proba(probe)
    assert probabilities.shape == (3, 2)
    assert np.isfinite(probabilities).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1)


def test_forest_seed_reproduces_predictions(small_training):
    X, y = small_training
    a = make_pipeline("Expanded + interaction", n_estimators=8, max_depth=5, n_jobs=1).fit(X, y)
    b = make_pipeline("Expanded + interaction", n_estimators=8, max_depth=5, n_jobs=1).fit(X, y)
    np.testing.assert_array_equal(a.predict_proba(X.iloc[:20]), b.predict_proba(X.iloc[:20]))


def test_imputers_learn_only_the_passed_training_fold(small_training):
    X, y = small_training
    fold = X.iloc[:300].copy()
    fold.loc[fold.index[:10], "age"] = np.nan
    model = make_pipeline("Expanded + interaction", n_estimators=4, max_depth=3, n_jobs=1)
    model.fit(fold, y.loc[fold.index])
    transformed = engineer_features(fold)
    numeric = model.named_steps["preprocess"].named_transformers_["numeric"]
    numeric_columns = next(cols for name, transformer, cols in model.named_steps["preprocess"].transformers_ if name == "numeric")
    np.testing.assert_allclose(numeric.statistics_, transformed[numeric_columns].median())
    before = numeric.statistics_.copy()
    model.predict(X.iloc[300:])
    np.testing.assert_array_equal(numeric.statistics_, before)


def test_metric_summary_matches_independent_sklearn_arithmetic(fitted_pipeline, small_training):
    X, y = small_training
    pred = fitted_pipeline.predict(X)
    score = fitted_pipeline.predict_proba(X)[:, 1]
    metrics = metric_summary(y, pred, score)
    expected = dict(accuracy=accuracy_score(y, pred), precision=precision_score(y, pred), recall=recall_score(y, pred), f1=f1_score(y, pred), roc_auc=roc_auc_score(y, score))
    for key, value in expected.items():
        assert metrics[key] == pytest.approx(value)


def test_inference_import_does_not_import_training():
    module = importlib.import_module("census_income.inference")
    assert "train" not in module.__dict__


def test_threshold_confusion_counts_reconcile_with_rows(fitted_pipeline, small_training):
    X, y = small_training
    score = fitted_pipeline.predict_proba(X)[:, 1]
    result = threshold_diagnostics(y, score)
    assert set(("threshold", "tn", "fp", "fn", "tp", "selection_rate")) <= set(result)
    for row in result.itertuples():
        assert row.tn + row.fp + row.fn + row.tp == len(y)
        assert row.tp + row.fp == int((score > row.threshold).sum())
        assert row.selection_rate == pytest.approx((row.tp + row.fp) / len(y))


def test_subgroup_denominators_reconcile_with_actual_data(fitted_pipeline, small_training):
    X, y = small_training
    result = subgroup_metrics(y, fitted_pipeline.predict(X), fitted_pipeline.predict_proba(X)[:, 1], X[["sex", "race"]])
    for row in result.itertuples():
        mask = X[row.attribute].eq(row.group)
        assert row.n == int(mask.sum())
        assert row.positives == int(y[mask].sum())
        assert row.negatives + row.positives == row.n
        assert row.tn + row.fp + row.fn + row.tp == row.n
        assert 0 <= row.selection_rate <= 1


@pytest.mark.workspace
def test_preserved_root_files_match_recorded_hashes():
    project = Path(__file__).resolve().parents[1]
    manifest = json.loads((project / "reports/preservation_manifest.json").read_text())
    assert manifest
    for name, details in manifest.items():
        path = project.parent / name
        assert path.stat().st_size == details["bytes"], name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == details["sha256"], name

