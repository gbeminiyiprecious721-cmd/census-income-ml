"""Six-page research dashboard. Read-only snapshots; no model training."""
from __future__ import annotations

import os
from pathlib import Path
import sys

import streamlit as st

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'app'))
from census_income.run_store import SnapshotSession
from ui import apply_style
from live_evaluation import score_snapshot
from overview import render_overview
import explorer
import performance
import explainability
import responsible
import sandbox

STORE=Path(os.environ.get('CENSUS_APP_STORE_DIR',ROOT/'artifacts/research_store'))
PAGES=['Research Overview','Data Explorer','Model Performance','Model Explainability','Responsible AI Assessment','Prediction Sandbox']

st.set_page_config(page_title='CensusIncome · Research Intelligence',page_icon=':material/analytics:',layout='wide')
apply_style()

if st.session_state.get('research_snapshot_store')!=str(STORE):
    st.session_state.research_snapshot_session=SnapshotSession()
    st.session_state.research_snapshot_store=str(STORE)
    st.session_state.rendered_snapshot_status=None


@st.cache_data(max_entries=2,show_spinner='Reading validated source snapshot…')
def frozen_source(signature,_snapshot):
    return _snapshot.source_data()


@st.cache_data(max_entries=2,show_spinner='Scoring validated inputs — no model training…')
def scored_data(signature,_snapshot):
    return score_snapshot(_snapshot)


def source_data():
    snapshot=st.session_state.research_snapshot_session.snapshot
    return frozen_source(snapshot.signature,snapshot)


def status(session):
    return(session.snapshot.signature if session.snapshot else None,session.error,session.stale)


@st.fragment(run_every='60s')
def refresh_status():
    session=st.session_state.research_snapshot_session
    force=st.button('Refresh evidence',key='refresh_snapshot',icon=':material/sync:',width='stretch',help='Validate the local research publication. Active sessions also check for a newly published bundle every 60 seconds.')
    if st.session_state.pop('analysis_refresh_requested',False):
        force=True
        scored_data.clear()
    override=os.environ.get('CENSUS_APP_REPORTS_DIR')
    if override and Path(override).resolve()!=(ROOT/'reports').resolve():
        session.error='Loose report-directory overrides are unsupported; publish a complete bundle and set CENSUS_APP_STORE_DIR.'
        session.stale=session.snapshot is not None
    else:session.refresh(STORE,force=force)
    previous=st.session_state.get('rendered_snapshot_status')
    if previous is not None and previous!=status(session):
        st.session_state.rendered_snapshot_status=status(session)
        st.rerun()
    st.caption('Validated local evidence · checks every 60s' if session.snapshot else 'No validated evidence loaded')
    with st.popover('Run provenance',icon=':material/fingerprint:',width='stretch'):
        if session.snapshot:
            m=session.snapshot.manifest
            st.caption(f"Research: {m['research_execution']['timestamp_utc']} (model creation; exact completion time unavailable)")
            st.caption(f"Published locally: {m['published_utc']}")
            st.caption(f'Loaded: {session.loaded_utc}')
            st.caption(f"Run {m['run_id']} · dataset source year 1994")
        else:st.caption('No model or reports have been activated.')


with st.sidebar:
    st.markdown('## :material/insights: CensusIncomeML')
    st.caption('INCOME INTELLIGENCE · RESEARCH')
    st.divider()
    page=st.radio('Navigate',PAGES,key='research_page',label_visibility='collapsed',format_func=lambda p:f'{PAGES.index(p)+1:02d}   {p}')
    st.divider()
    refresh_status()
    st.caption('UCI ADULT / 1994\n\nHistorical research.\nEducational use only.')

session=st.session_state.research_snapshot_session
if session.error:
    if session.snapshot:st.warning('Update rejected. Showing the last validated research snapshot; all reports, source data and predictions remain on that snapshot.')
    else:st.warning('Generate and validate a complete local bundle with `.venv/bin/python scripts/publish_run.py`, then select Refresh evidence. No training is performed by this command.')
    with st.popover('Evidence loading details'):st.caption(session.error)

with st.container(key='dashboard_workspace',gap='small'):
    if session.snapshot:
        snapshot=session.snapshot
        if page==PAGES[0]:render_overview(snapshot,session.loaded_utc,scored_data(snapshot.signature,snapshot))
        elif page==PAGES[1]:explorer.render(snapshot,source_data)
        elif page==PAGES[2]:performance.render(snapshot,scored_data(snapshot.signature,snapshot))
        elif page==PAGES[3]:explainability.render(snapshot)
        elif page==PAGES[4]:responsible.render(snapshot,scored_data(snapshot.signature,snapshot))
        else:sandbox.render(snapshot)
    else:st.title(page)

st.session_state.rendered_snapshot_status=status(session)
with st.container(key='dashboard_footer'):
    st.caption('CENSUS INCOME ML · 1994 historical sample · validated research snapshot · research and education only')
