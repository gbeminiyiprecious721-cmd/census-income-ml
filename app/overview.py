"""Executive evidence workspace with real, snapshot-bound analytical controls."""
from dataclasses import replace

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from analysis_controls import controls,actions
from live_evaluation import evaluate_context,threshold_curve,curve_data,METRICS,MODELS
from ui import header,kpis,plot,matrix,detail_button,BLUE,TEAL,PURPLE,AMBER,CORAL,SLATE,NAVY,FEATURE_LABELS


def render_overview(snapshot, loaded_utc, scored=None):
    if scored is None:
        from live_evaluation import score_snapshot
        scored=score_snapshot(snapshot)
    header('Research overview','UCI Adult · investigate model evidence, error trade-offs and the limits of historical income classification.',1)
    current=controls();result=evaluate_context(scored,current)
    actions(snapshot,result,current)
    metrics=result['metrics'];prefix='Test' if current.split=='test' else 'In-sample'
    fmt=lambda name: '—' if metrics[name] is None else f"{metrics[name]:.3f}" if name=='f1' else f"{metrics[name]:.4f}" if name=='roc_auc' else f"{metrics[name]:.2%}"
    kpis([(f'{prefix} accuracy',fmt('accuracy'),'Correct classifications / evaluated records'),
          (f'{prefix} precision',fmt('precision'),'True positives / positive predictions'),
          (f'{prefix} recall',fmt('recall'),'True positives / actual positives'),
          (f'{prefix} F1',fmt('f1'),'Harmonic mean of positive-class precision and recall'),
          (f'{prefix} ROC-AUC',fmt('roc_auc'),'Ranking discrimination; unavailable for a one-class sample'),
          ('Evaluated samples',f"{result['n']:,}",'Current split and population; not a training-history count')])
    left,right=st.columns([1.1,1],gap='small')
    key=METRICS[current.metric]
    with left,st.container(border=True,key='panel_overview_1'):
        st.subheader('Development comparison' if current.comparison=='Matched development CV' else 'Model comparison',icon=':material/bar_chart:')
        if current.comparison=='Matched development CV':
            s=snapshot.report('summary.json');table=snapshot.report('model_comparison.csv')
            fixed=table[(table.model==s['selected_feature_group']) & table.evaluation.str.startswith('fixed')].iloc[0]
            base=table[table.model=='Majority-class baseline'].iloc[0]
            data=pd.DataFrame({'Model':['Majority baseline','Fixed forest · CV','F1-selected forest · CV'],'Value':[base[key],fixed[key],s['cv_metrics'][key]]})
            st.caption('Global five-fold CV · same folds · population/threshold controls do not reinterpret CV')
        else:
            rows=[]
            for model in MODELS:
                value=evaluate_context(scored,replace(current,model=model))['metrics'][key]
                rows.append({'Model':model,'Value':value})
            data=pd.DataFrame(rows)
            st.caption('Same selected split, population and scenario threshold · dashboard computation')
        fig=px.bar(data,x='Value',y='Model',orientation='h',color='Model',text='Value',color_discrete_sequence=[SLATE,TEAL,PURPLE])
        fig.update_traces(texttemplate='%{x:.3f}' if key in ['f1','roc_auc'] else '%{x:.1%}',textposition='outside',cliponaxis=False)
        fig.update_layout(showlegend=False,xaxis=dict(title=current.metric,range=[0,1.12],tickvals=[0,.25,.5,.75,1]),yaxis=dict(title=None,autorange='reversed'),margin=dict(l=120))
        plot(fig,'Matched model comparison using the explicitly selected comparison scope','overview_cv',150)
    with right,st.container(border=True,key='panel_overview_2'):
        st.subheader('Held-out test · confusion matrix' if current.split=='test' else 'In-sample confusion matrix',icon=':material/grid_view:')
        a,b=st.columns([1,2])
        with a:mode=st.segmented_control('Display',['Count','Percentage'],default='Count',key='overview_matrix_mode',label_visibility='collapsed')
        with b:st.caption(f"{current.model} · scenario threshold {current.threshold:.2f} · n={result['n']:,}")
        plot(matrix(result['confusion'],mode=='Percentage'),'Actual versus predicted income for the selected evaluation context','overview_test',140)
    a,b,c=st.columns([1.1,1.15,.85],gap='small')
    with a,st.container(border=True,key='panel_overview_3'):
        st.subheader('Global feature importance',icon=':material/insights:')
        st.caption('Saved forest permutation evidence · full historical test sample')
        if current.model==MODELS[0]:
            data=snapshot.report('importance_comparison.csv').nlargest(5,'f1_decrease_mean').copy()
            data['Feature']=data.feature.map(lambda v:FEATURE_LABELS.get(v,v))
            fig=px.bar(data,x='f1_decrease_mean',y='Feature',orientation='h',error_x='f1_shuffle_sd',color='Feature',color_discrete_sequence=[BLUE,TEAL,CORAL,PURPLE,AMBER])
            fig.update_layout(showlegend=False,xaxis_title='Mean F1 decrease ± shuffle SD',yaxis=dict(title=None,autorange='reversed'))
            plot(fig,'Top five saved global permutation importance values; not local explanations','overview_importance',145)
        else:st.info('Not available: a constant majority baseline has no fitted feature importance.')
    with b,st.container(border=True,key='panel_overview_4'):
        st.subheader('Score & threshold diagnostics',icon=':material/show_chart:')
        view=st.segmented_control('Diagnostic',['ROC','Precision–recall','Threshold'],default='ROC',key='overview_curve_kind',label_visibility='collapsed') or 'ROC'
        if view=='Threshold':
            data=threshold_curve(result['actual'],result['scores'])
            fig=px.line(data,x='threshold',y=['precision','recall','f1'],color_discrete_sequence=[TEAL,PURPLE,CORAL],labels={'value':'Score','variable':'Metric','threshold':'Scenario threshold'})
            fig.add_vline(x=current.threshold,line_dash='dot',line_color=AMBER)
            fig.update_yaxes(range=[0,1])
        else:
            data=curve_data(result,view)
            if data.empty:fig=go.Figure().add_annotation(text='Both target classes are required',showarrow=False)
            else:
                x,y=list(data.columns)
                fig=px.line(data,x=x,y=y,color_discrete_sequence=[BLUE if view=='ROC' else PURPLE])
                if view=='ROC':fig.add_shape(type='line',x0=0,y0=0,x1=1,y1=1,line=dict(dash='dot',color=SLATE))
                fig.update_xaxes(range=[0,1]);fig.update_yaxes(range=[0,1])
        plot(fig,'Score diagnostic computed from the selected frozen model and evaluation population','overview_curves',130)
    with c,st.container(border=True,key='panel_overview_5'):
        st.subheader('Prediction distribution',icon=':material/donut_large:')
        st.caption('Scenario classifications · not actual income composition')
        counts=result['counts'];negative=counts['tn']+counts['fn'];positive=counts['tp']+counts['fp']
        fig=go.Figure(go.Pie(labels=['≤$50K','>$50K'],values=[negative,positive],hole=.72,marker_colors=[TEAL,PURPLE],textinfo='percent',textposition='inside',textfont_size=11,sort=False))
        fig.add_annotation(text=f"<b>{result['n']:,}</b>",x=.5,y=.5,showarrow=False,font_size=13)
        plot(fig,'Predicted income class counts under the current scenario threshold','overview_distribution',220,
             legend=dict(title_text='',orientation='h',y=-.12,yanchor='top',x=.5,xanchor='center'),
             margin=dict(l=12,r=12,t=12,b=48))
    counts=result['counts']
    st.caption(f"RESEARCH INSIGHT · This scenario finds {counts['tp']:,} of {counts['tp']+counts['fn']:,} actual positives and flags {counts['fp']:,} actual negatives. Scores are uncalibrated; threshold changes are diagnostics, not a recommended policy.")
