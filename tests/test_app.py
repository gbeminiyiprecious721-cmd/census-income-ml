"""Exercise actual page routing and honest unavailable-artifact states."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("page", ["Research Overview", "Data Explorer", "Model Performance", "Model Explainability", "Responsible AI Assessment", "Prediction Sandbox"])
def test_pages_render_without_exceptions(page):
    app = AppTest.from_file(str(ROOT / "app/main.py"), default_timeout=90).run()
    app.sidebar.radio[0].set_value(page).run()
    assert not app.exception
    assert app.title


def test_missing_reports_are_actionable(monkeypatch):
    monkeypatch.setenv("CENSUS_APP_REPORTS_DIR", str(ROOT / "tests" / "deliberately_absent_reports"))
    app = AppTest.from_file(str(ROOT / "app/main.py"), default_timeout=90).run()
    assert not app.exception
    assert any("Generate" in item.value for item in app.warning)
    assert not app.metric


def test_sandbox_prediction_or_actionable_artifact_error():
    app = AppTest.from_file(str(ROOT / "app/main.py"), default_timeout=90).run()
    app.sidebar.radio[0].set_value("Prediction Sandbox").run()
    app.button(key='classify_record').click().run()
    assert not app.exception
    assert app.error or any(m.label == 'Model score · >$50K' for m in app.metric)
    assert not app.success
    assert any('uncalibrated' in item.value.lower() for item in app.caption) or app.error


@pytest.mark.skipif(not (ROOT / "artifacts/model.joblib").exists(), reason="Fresh fitted research artifact is not yet generated")
def test_sandbox_fitted_pipeline_with_unknown_category():
    app = AppTest.from_file(str(ROOT / "app/main.py"), default_timeout=90).run()
    app.sidebar.radio[0].set_value("Prediction Sandbox").run()
    app.selectbox(key='hyp_input_country').set_value('Other / unseen category')
    app.text_input(key='hyp_input_other_country').set_value('Hypothetical-unseen-country')
    app.button(key='classify_record').click().run()
    assert not app.exception
    assert not app.success and not app.error
    assert any(m.label == 'Model score · >$50K' for m in app.metric)
    assert any("unknown" in item.value.lower() or "unseen" in item.value.lower() for item in app.warning)


@pytest.mark.skipif(not (ROOT / "reports/summary.json").exists(), reason="Fresh research reports are not yet generated")
@pytest.mark.parametrize("view", ["Confusion matrix", "ROC curve", "Precision–recall curve", "Calibration", "Threshold sensitivity"])
def test_fresh_performance_diagnostics(view):
    app = AppTest.from_file(str(ROOT / "app/main.py"), default_timeout=90).run()
    app.sidebar.radio[0].set_value("Model Performance").run()
    app.selectbox(key='performance_view').set_value(view).run()
    assert not app.exception
    assert not app.warning


@pytest.mark.skipif(not (ROOT / "reports/subgroup_metrics.csv").exists(), reason="Fresh subgroup report is not yet generated")
def test_subgroup_false_negative_rate_view():
    app = AppTest.from_file(str(ROOT / "app/main.py"), default_timeout=90).run()
    app.sidebar.radio[0].set_value("Responsible AI Assessment").run()
    app.selectbox(key='demographic_rate').set_value('fnr').run()
    assert not app.exception
    assert len(app.get("plotly_chart")) == 2
