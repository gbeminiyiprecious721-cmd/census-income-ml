"""A fresh source checkout must open with validated research on first visit."""
from pathlib import Path
import shutil

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


def test_fresh_checkout_initializes_saved_evidence(tmp_path, monkeypatch):
    for name in ('app', 'src', 'reports', 'data', 'artifacts', '.streamlit'):
        shutil.copytree(ROOT / name, tmp_path / name,
                        ignore=shutil.ignore_patterns('research_store', '__pycache__'))
    monkeypatch.delenv('CENSUS_APP_STORE_DIR', raising=False)
    monkeypatch.delenv('CENSUS_APP_REPORTS_DIR', raising=False)
    app = AppTest.from_file(str(tmp_path / 'app/main.py'), default_timeout=120).run()
    assert not app.exception
    assert (tmp_path / 'artifacts/research_store/current.json').exists()
    assert any(item.label == 'Test accuracy' and item.value == '83.22%'
               for item in app.metric)
    pointer = (tmp_path / 'artifacts/research_store/current.json').read_bytes()
    app.run()
    assert not app.exception
    assert (tmp_path / 'artifacts/research_store/current.json').read_bytes() == pointer
