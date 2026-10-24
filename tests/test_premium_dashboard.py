"""Core product behavior for the premium analytical workspace."""
from pathlib import Path
from streamlit.testing.v1 import AppTest

ROOT=Path(__file__).resolve().parents[1]


def start():return AppTest.from_file(str(ROOT/'app/main.py'),default_timeout=90).run()


def metric(app,label):return next(m.value for m in app.metric if m.label==label)


def test_overview_threshold_population_and_reset_change_real_metrics():
    app=start()
    assert not app.exception
    assert metric(app,'Test accuracy')=='83.22%'
    assert metric(app,'Evaluated samples')=='9,758'
    assert len(app.get('plotly_chart'))>=5
    app.slider(key='analysis_threshold').set_value(.8).run()
    assert not app.exception
    assert metric(app,'Test precision')=='82.25%'
    app.button(key='analysis_reset').click().run()
    assert metric(app,'Test precision')=='61.26%'
    app.selectbox(key='analysis_segment').set_value('Female').run()
    assert metric(app,'Evaluated samples')=='3,326'


def test_baseline_selection_and_context_survive_navigation():
    app=start()
    app.selectbox(key='analysis_model').set_value('Majority baseline').run()
    assert metric(app,'Test recall')=='0.00%'
    app.sidebar.radio[0].set_value('Model Performance').run()
    assert not app.exception
    app.sidebar.radio[0].set_value('Research Overview').run()
    assert app.selectbox(key='analysis_model').value=='Majority baseline'
    assert metric(app,'Test recall')=='0.00%'


def test_reset_threshold_remains_reset_after_refresh():
    app=start()
    app.slider(key='analysis_threshold').set_value(.8).run()
    app.button(key='analysis_reset').click().run()
    assert metric(app,'Test precision')=='61.26%'
    app.button(key='analysis_refresh').click().run()
    assert app.slider(key='analysis_threshold').value==.5
    assert metric(app,'Test precision')=='61.26%'
