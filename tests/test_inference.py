"""Prediction contract and pre-deserialization artifact integrity checks."""
import numpy as np
import pandas as pd
import pytest

from census_income import inference


@pytest.fixture
def local_model(monkeypatch, fitted_pipeline):
    monkeypatch.setattr(inference, "load_validated_model", lambda: fitted_pipeline)


def test_prediction_contract_and_direct_pipeline_agree(local_model, fitted_pipeline, valid_record):
    result = inference.predict_records(valid_record)
    assert set(("predicted_class", "income_category", "model_score", "input_warnings")) <= set(result)
    assert len(result) == 1
    assert result.predicted_class.iloc[0] in (0, 1)
    assert result.income_category.iloc[0] in ("<=50K", ">50K")
    assert result.model_score.iloc[0] == pytest.approx(fitted_pipeline.predict_proba(valid_record)[0, 1])
    pd.testing.assert_frame_equal(result, inference.predict_records(valid_record.to_dict("records")))


@pytest.mark.parametrize("column,value", [("age", 16), ("age", 91), ("age", "39"), ("age", True), ("hours-per-week", 0), ("hours-per-week", 100), ("education-num", 0), ("education-num", 17), ("capital-gain", -1), ("capital-loss", 4357), ("age", np.inf), ("occupation", 42)])
def test_malformed_inputs_are_rejected(local_model, valid_record, column, value):
    probe = valid_record.astype(object)
    probe.loc[:, column] = value
    with pytest.raises((ValueError, TypeError)):
        inference.predict_records(probe)


def test_missing_required_column_is_rejected(local_model, valid_record):
    with pytest.raises(ValueError):
        inference.predict_records(valid_record.drop(columns="education-num"))


def test_missing_values_use_saved_imputers(local_model, valid_record):
    valid_record["age"] = np.nan
    valid_record["occupation"] = None
    result = inference.predict_records(valid_record)
    assert np.isfinite(result.model_score.iloc[0])


def test_unknown_categories_return_warnings(local_model, valid_record):
    valid_record.loc[:, "occupation"] = "Previously unseen occupation"
    result = inference.predict_records(valid_record)
    assert np.isfinite(result.model_score.iloc[0])
    assert result.input_warnings.iloc[0]


def test_missing_artifact_fails_actionably(tmp_path, monkeypatch):
    monkeypatch.setattr(inference, "MODEL_PATH", tmp_path / "model.joblib")
    monkeypatch.setattr(inference, "MANIFEST_PATH", tmp_path / "model_manifest.json")
    with pytest.raises((FileNotFoundError, ValueError), match="(?i)model|artifact|manifest"):
        inference.load_validated_model()


def test_saved_pipeline_roundtrip_and_corruption_rejected(tmp_path, monkeypatch, fitted_pipeline, valid_record):
    from census_income.train import save_model
    save_model(fitted_pipeline, {"purpose": "functional test using cached UCI observations"}, artifact_dir=tmp_path)
    monkeypatch.setattr(inference, "MODEL_PATH", tmp_path / "model.joblib")
    monkeypatch.setattr(inference, "MANIFEST_PATH", tmp_path / "model_manifest.json")
    loaded = inference.load_validated_model()
    np.testing.assert_array_equal(loaded.predict_proba(valid_record), fitted_pipeline.predict_proba(valid_record))
    with (tmp_path / "model.joblib").open("ab") as handle:
        handle.write(b"corrupted")
    with pytest.raises(ValueError, match="(?i)hash|checksum|integrity"):
        inference.load_validated_model()


def test_dependency_mismatch_rejected_before_deserialization(tmp_path, monkeypatch, fitted_pipeline):
    from census_income.train import save_model
    save_model(fitted_pipeline, {}, artifact_dir=tmp_path)
    manifest_path = tmp_path / "model_manifest.json"
    original = manifest_path.read_text()
    import sklearn
    assert sklearn.__version__ in original
    manifest_path.write_text(original.replace(sklearn.__version__, "0.0.invalid"))
    monkeypatch.setattr(inference, "MODEL_PATH", tmp_path / "model.joblib")
    monkeypatch.setattr(inference, "MANIFEST_PATH", manifest_path)
    def forbidden_load(*args, **kwargs):
        pytest.fail("Artifact was deserialized before package compatibility validation")
    monkeypatch.setattr(inference.joblib, "load", forbidden_load)
    with pytest.raises(ValueError, match="(?i)version|compatib|dependenc"):
        inference.load_validated_model()


def test_project_validated_artifact_predicts_raw_inputs(valid_record):
    if not inference.MODEL_PATH.exists():
        pytest.skip("Notebook has not generated the validated project artifact yet")
    model = inference.load_validated_model()
    result = inference.predict_records(valid_record)
    np.testing.assert_allclose(result.model_score, model.predict_proba(valid_record)[:, 1])
