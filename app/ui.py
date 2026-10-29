"""Shared corporate research identity and compact, accessible presentation."""
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

BLUE='#19816E'
TEAL='#35A7A0'
PURPLE='#8F7CB7'
AMBER='#E7AB56'
NAVY='#4A6254'
SLATE='#9CA99A'
INK='#27332F'
CORAL='#E78168'
MAGENTA='#B7739B'
PALETTE=[BLUE,TEAL,CORAL,MAGENTA,AMBER,PURPLE]
INCOME_COLORS={'<=50K': BLUE, '>50K': PURPLE}
FEATURE_LABELS={'marital-status':'Marital status','capital-gain':'Capital gain',
    'capital-loss':'Capital loss','hours_age_interaction':'Age × weekly hours',
    'education_bin':'Education group','occupation':'Occupation','age':'Age',
    'hours-per-week':'Weekly hours','workclass':'Workclass','relationship':'Relationship',
    'native-country':'Native country','education-num':'Education level'}


def apply_style():
    st.html('<style>'+Path(__file__).with_name('style.css').read_text()+'</style>')


def header(title, subtitle, number):
    with st.container(key='page_heading',gap='small'):
        st.caption(f'CENSUS INCOME ML  /  EVIDENCE WORKSPACE  /  {number:02d}')
        st.title(title)
        st.caption(subtitle)


def kpis(items):
    with st.container(horizontal=True,key='kpi_row'):
        for label,value,help_text in items:
            slug=label.lower().replace(' ','_').replace('-','_')
            with st.container(key='kpi_'+slug):
                st.metric(label,value,help=help_text,border=True)


def plot(fig, alt, key, height=300, selectable=False, on_select=None, *, legend=None, margin=None):
    fig.update_layout(height=height, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Arial, sans-serif',size=12,color=INK),
        margin=margin if margin is not None else dict(l=20,r=20,t=18,b=30),
        legend=legend if legend is not None else dict(title_text='',orientation='h',y=1.13,x=0),
        hoverlabel=dict(bgcolor='white',font_size=12),
        colorway=PALETTE, transition_duration=150)
    fig.update_xaxes(automargin=True,gridcolor='#EDF0E9',zeroline=False,title_font_size=11)
    fig.update_yaxes(automargin=True,gridcolor='#EDF0E9',zeroline=False,title_font_size=11)
    options = dict(on_select=on_select or 'rerun',selection_mode='points') if selectable else {}
    return st.plotly_chart(fig, width='stretch', theme=None, key=key, alt=alt,
        config={'displayModeBar':'hover','displaylogo':False,'responsive':True,
                'modeBarButtonsToRemove':['lasso2d']}, **options)


def matrix(frame, normalize=False):
    values=frame.pivot(index='actual',columns='predicted',values='count').reindex(index=[0,1],columns=[0,1])
    raw=values.to_numpy()
    import numpy as np
    z=np.divide(raw,raw.sum(axis=1,keepdims=True),out=np.zeros_like(raw,dtype=float),where=raw.sum(axis=1,keepdims=True)!=0) if normalize else raw
    text=[[f'{z[i,j]:.1%}<br>n={raw[i,j]:,}' if normalize else f'{raw[i,j]:,}' for j in range(2)] for i in range(2)]
    fig=go.Figure(go.Heatmap(z=z,x=['≤$50K','>$50K'],y=['≤$50K','>$50K'],
        text=text,texttemplate='%{text}',textfont_size=15,
        colorscale=[[0,'#EDF5EE'],[.5,'#8CC9AF'],[1,BLUE]],showscale=False,
        hovertemplate='Actual %{y}<br>Predicted %{x}<br>%{text}<extra></extra>'))
    fig.update_layout(xaxis_title='Predicted income category',yaxis_title='Actual income category')
    fig.update_yaxes(autorange='reversed')
    return fig


@st.dialog('Evidence & definitions', width='large')
def evidence_dialog(snapshot, table_name=None):
    summary=snapshot.report('summary.json')
    st.caption('SAVED RESEARCH ARTIFACT · original0.50 policy · separate from current dashboard scenarios')
    st.caption('1994 US Census · unweighted historical sample · research and education only')
    if table_name:
        data=snapshot.report(table_name)
        if isinstance(data,dict):st.json(data)
        else:st.dataframe(data,hide_index=True,height=300,width='stretch',alt='Full-precision evidence for the selected dashboard view')
    st.markdown('**Accuracy:** all correctly classified records. **Precision:** true positives / positive predictions. **Recall:** true positives / actual positives. **F1:** harmonic mean of precision and recall. **ROC-AUC:** ranking discrimination across thresholds.')
    st.markdown('**Frozen rule:** score >0.50 predicts >$50K, with class 0 on ties. Scores are uncalibrated. CV comparisons concern development selection; test results are a historical reproduction.')
    st.caption(' · '.join(summary['limitations']))
    m=snapshot.manifest
    st.caption(f"Run {m['run_id']} · model creation {m['research_execution']['timestamp_utc']} · Published locally {m['published_utc']} · Loaded {st.session_state.research_snapshot_session.loaded_utc}")


def detail_button(snapshot, report=None, key='view_evidence'):
    if st.button('Evidence & definitions',key=key,icon=':material/info:',type='tertiary'):
        evidence_dialog(snapshot,report)
