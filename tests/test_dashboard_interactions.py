"""User-facing chart/filter contracts for the six-page redesign."""
from pathlib import Path
from streamlit.testing.v1 import AppTest

ROOT=Path(__file__).resolve().parents[1]


def start(page):
    app=AppTest.from_file(str(ROOT/'app/main.py'),default_timeout=60).run()
    app.sidebar.radio[0].set_value(page).run()
    return app


def test_explorer_filters_update_cohort_and_reset():
    app=start('Data Explorer')
    assert len(app.get('plotly_chart')) >= 2
    before=next(m.value for m in app.metric if m.label=='Cohort records')
    app.selectbox(key='explorer_sex').set_value('Female').run()
    assert not app.exception
    assert next(m.value for m in app.metric if m.label=='Cohort records') == '16,176'
    assert next(m.value for m in app.metric if m.label=='Cohort records') != before
    app.button(key='reset_explorer').click().run()
    assert next(m.value for m in app.metric if m.label=='Cohort records') == '48,790'


def test_explorer_empty_cohort_is_honest_and_resettable():
    app=start('Data Explorer')
    app.slider(key='explorer_age').set_value((17,17))
    app.selectbox(key='explorer_education').set_value('Postgraduate').run()
    assert not app.exception
    assert next(m.value for m in app.metric if m.label=='Cohort records') == '0'
    assert any('No records' in item.value for item in app.info)
    app.button(key='reset_explorer').click().run()
    assert next(m.value for m in app.metric if m.label=='Cohort records')=='48,790'


def test_explainability_top_n_controls_ranking():
    app=start('Model Explainability')
    app.select_slider(key='importance_top_n').set_value(5).run()
    assert not app.exception
    assert len(app.get('plotly_chart')) >= 2
    assert any('shuffle' in c.value.lower() for c in app.caption)


def test_responsible_group_and_rate_controls():
    app=start('Responsible AI Assessment')
    app.selectbox(key='demographic_attribute').set_value('race').run()
    app.selectbox(key='demographic_rate').set_value('fnr').run()
    assert not app.exception
    assert len(app.get('plotly_chart')) >= 2
    assert any('actual positives' in c.value.lower() for c in app.caption)


def test_sandbox_country_labels_and_neutral_result():
    app=start('Prediction Sandbox')
    assert len(app.get('plotly_chart')) >= 1
    app.button(key='classify_record').click().run()
    assert not app.exception and not app.error
    assert any(m.label=='Model score · >$50K' for m in app.metric)
    assert not app.success


def test_sandbox_submitted_profile_survives_navigation_with_result():
    app=start('Prediction Sandbox')
    app.number_input(key='hyp_input_age').set_value(67)
    app.button(key='classify_record').click().run()
    score=next(m.value for m in app.metric if m.label=='Model score · >$50K')
    app.sidebar.radio[0].set_value('Research Overview').run()
    app.sidebar.radio[0].set_value('Prediction Sandbox').run()
    assert not app.exception
    assert app.number_input(key='hyp_input_age').value == 67
    assert next(m.value for m in app.metric if m.label=='Model score · >$50K') == score
    assert any('Attribute' in f.value and f.value.loc[f.value.Attribute.eq('Age'),'Value'].tolist()==['67'] for f in app.dataframe)
    app.button(key='reset_hypothetical').click().run()
    assert app.number_input(key='hyp_input_age').value == 35
    assert not any(m.label=='Model score · >$50K' for m in app.metric)


def test_legacy_result_without_profile_is_discarded_safely():
    app=start('Prediction Sandbox')
    signature=app.session_state['research_snapshot_session'].snapshot.signature
    app.session_state['hypothetical_result']={'signature':signature,'model_score':.4,
        'income_category':'<=50K','input_warnings':''}
    app.run()
    assert not app.exception
    assert not any(m.label=='Model score · >$50K' for m in app.metric)
