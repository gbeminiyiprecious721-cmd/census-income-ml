"""Persistent model/evaluation context and real export/reset/refresh actions."""
from dataclasses import asdict
import json

import streamlit as st
from live_evaluation import AnalysisContext,MODELS,SEGMENTS,SPLITS,METRICS,export_analysis

FIELDS=['model','metric','split','segment','threshold','comparison']


def context() -> AnalysisContext:
    return AnalysisContext(**st.session_state.setdefault('analysis_context',asdict(AnalysisContext())))


def save(field: str) -> None:
    st.session_state.analysis_context[field]=st.session_state['analysis_'+field]


def reset() -> None:
    defaults=asdict(AnalysisContext())
    # Set live widget state too: deleting a key leaves its frontend value able to
    # repopulate the state on the next unrelated rerun (for example Refresh).
    for field in FIELDS:st.session_state['analysis_'+field]=defaults[field]
    st.session_state.analysis_context=defaults


def refresh() -> None:
    st.session_state.analysis_refresh_requested=True


def controls() -> AnalysisContext:
    current=context()
    with st.container(border=True,key='analysis_controls',gap='small'):
        columns=st.columns([1.6,1,1.6,1.25,1.2,1.7],gap='small')
        with columns[0]:st.selectbox('Model',MODELS,index=MODELS.index(current.model),key='analysis_model',on_change=save,args=('model',))
        with columns[1]:st.selectbox('Evaluation metric',list(METRICS),index=list(METRICS).index(current.metric),key='analysis_metric',on_change=save,args=('metric',))
        with columns[2]:st.selectbox('Dataset split',list(SPLITS),format_func=SPLITS.get,index=list(SPLITS).index(current.split),key='analysis_split',on_change=save,args=('split',))
        with columns[3]:st.selectbox('Population',SEGMENTS,index=SEGMENTS.index(current.segment),key='analysis_segment',on_change=save,args=('segment',))
        with columns[4]:st.slider('Scenario threshold',.05,.95,float(current.threshold),.05,key='analysis_threshold',on_change=save,args=('threshold',),help='Changes local diagnostics only; the validated artifact retains its original0.50 rule.')
        with columns[5]:st.selectbox('Comparison scope',['Matched development CV','Current evaluation'],index=['Matched development CV','Current evaluation'].index(current.comparison),key='analysis_comparison',on_change=save,args=('comparison',))
    return context()


def actions(snapshot, result: dict, current: AnalysisContext, prefix='analysis') -> None:
    a,b,c,d=st.columns([2.9,1.2,1.1,1.1],vertical_alignment='center')
    with a:
        scope='Held-out test' if current.split=='test' else 'In-sample development — optimistic, not CV'
        st.caption(f'{scope} · {current.segment} · strict score >{current.threshold:.2f}')
    with b:
        st.download_button('Export results',data=json.dumps(export_analysis(result,current,snapshot.signature),indent=2),file_name='census_analysis.json',mime='application/json',key=prefix+'_export',icon=':material/download:',width='stretch')
    with c:st.button('Reset filters',on_click=reset,key=prefix+'_reset',icon=':material/restart_alt:',width='stretch')
    with d:st.button('Refresh analysis',on_click=refresh,key=prefix+'_refresh',icon=':material/sync:',width='stretch')
