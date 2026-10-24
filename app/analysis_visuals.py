"""Reusable visualizations for the explicitly selected evaluation population."""
from dataclasses import replace

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from live_evaluation import MODELS,METRICS,evaluate_context,curve_data,threshold_curve
from ui import BLUE,TEAL,PURPLE,CORAL,AMBER,SLATE


def comparison(snapshot,scored,current):
    key=METRICS[current.metric]
    if current.comparison=='Matched development CV':
        s=snapshot.report('summary.json');table=snapshot.report('model_comparison.csv')
        fixed=table[(table.model==s['selected_feature_group']) & table.evaluation.str.startswith('fixed')].iloc[0]
        base=table[table.model=='Majority-class baseline'].iloc[0]
        data=pd.DataFrame({'Model':['Majority baseline','Fixed forest · CV','F1-selected forest · CV'],'Value':[base[key],fixed[key],s['cv_metrics'][key]]})
        caption='Global five-fold CV · same folds · unchanged by scenario filters'
    else:
        data=pd.DataFrame([{'Model':model,'Value':evaluate_context(scored,replace(current,model=model))['metrics'][key]} for model in MODELS])
        caption='Same selected split, population and scenario threshold'
    fig=px.bar(data,x='Value',y='Model',orientation='h',color='Model',text='Value',color_discrete_sequence=[SLATE,TEAL,PURPLE])
    fig.update_traces(texttemplate='%{x:.3f}' if key in ['f1','roc_auc'] else '%{x:.1%}',textposition='outside',cliponaxis=False)
    fig.update_layout(showlegend=False,xaxis=dict(title=current.metric,range=[0,1.12],tickvals=[0,.25,.5,.75,1],tickformat='.2f' if key in ['f1','roc_auc'] else '.0%'),yaxis=dict(title=None,autorange='reversed'))
    return fig,caption


def diagnostic(result,kind,threshold):
    if kind=='Threshold':
        data=threshold_curve(result['actual'],result['scores'])
        fig=px.line(data,x='threshold',y=['precision','recall','f1'],markers=True,color_discrete_sequence=[TEAL,PURPLE,CORAL],labels={'value':'Score','variable':'Metric','threshold':'Scenario threshold'})
        fig.add_vline(x=threshold,line_dash='dot',line_color=AMBER)
        fig.update_yaxes(range=[0,1])
        return fig
    data=curve_data(result,kind)
    if data.empty:return go.Figure().add_annotation(text='Both target classes are required',showarrow=False)
    x,y=list(data.columns)
    fig=px.line(data,x=x,y=y,color_discrete_sequence=[BLUE if kind=='ROC' else PURPLE])
    if kind=='ROC':fig.add_shape(type='line',x0=0,y0=0,x1=1,y1=1,line=dict(dash='dot',color=SLATE))
    fig.update_xaxes(range=[0,1]);fig.update_yaxes(range=[0,1])
    return fig
