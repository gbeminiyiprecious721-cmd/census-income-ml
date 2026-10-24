"""UI boundary tests: report identity, explicit evaluation and recovery behavior."""
from pathlib import Path
import json

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def app_store(tmp_path, monkeypatch):
    from census_income.run_store import publish_existing
    publish_existing(ROOT, tmp_path)
    monkeypatch.setenv('CENSUS_APP_STORE_DIR', str(tmp_path))
    return tmp_path


def start():
    return AppTest.from_file(str(ROOT/'app/main.py'), default_timeout=60).run()


def test_overview_separates_cv_test_and_dates(app_store):
    app = start()
    assert not app.exception
    captions = ' '.join(item.value for item in app.caption)
    assert '1994' in captions and 'Published' in captions and 'Loaded' in captions
    assert any('Development comparison' in item.value for item in app.subheader)
    assert any('Held-out test' in item.value for item in app.subheader)
    metrics = {item.label: item.value for item in app.metric}
    assert metrics['Test accuracy'] == '83.22%'
    assert metrics['Test recall'] == '81.38%'
    assert metrics['Test precision'] == '61.26%'
    assert metrics['Test F1'] == '0.699'
    assert app.get('plotly_chart') and not app.warning


def test_rejected_update_keeps_last_valid_ui_and_recovers(app_store):
    app = start()
    before = {item.label: item.value for item in app.metric}
    pointer = (app_store/'current.json').read_bytes()
    (app_store/'current.json').write_text(json.dumps({'broken': True}))
    app.run()
    assert not app.exception
    assert any('last validated' in item.value.lower() for item in app.warning)
    assert {item.label: item.value for item in app.metric} == before
    (app_store/'current.json').write_bytes(pointer)
    app.run()
    assert not app.exception and not app.warning


def test_no_snapshot_never_shows_unvalidated_metrics(tmp_path, monkeypatch):
    monkeypatch.setenv('CENSUS_APP_STORE_DIR', str(tmp_path))
    app = start()
    assert not app.exception
    assert app.warning and not app.metric
    assert any('publish_run.py' in item.value for item in app.warning)


def test_incomplete_overview_update_keeps_renderable_last_valid_snapshot(app_store):
    from test_run_store import rewrite
    app = start()
    before = {item.label: item.value for item in app.metric}
    def edit(path):
        value = json.loads(path.read_text())
        value.pop('limitations')
        path.write_text(json.dumps(value))
    rewrite(app_store, 'reports/summary.json', edit)
    app.run()
    assert not app.exception
    assert any('last validated' in item.value.lower() for item in app.warning)
    assert {item.label: item.value for item in app.metric} == before
