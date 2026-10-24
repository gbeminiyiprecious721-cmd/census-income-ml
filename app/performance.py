"""Computed model evaluation workspace; saved research remains unchanged."""
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from analysis_controls import controls,actions
from analysis_visuals import comparison,diagnostic
from live_evaluation import evaluate_context,class_report
from census_income.evaluate import calibration_diagnostics
from ui import header,kpis,plot,matrix,BLUE,TEAL,PURPLE,AMBER,CORAL


def render(snapshot,scored):
    header('Model performance','Evaluate the selected model and population; compare errors, ranking and threshold trade-offs.',3)
    current=controls();result=evaluate_context(scored,current);metrics=result['metrics']
    actions(snapshot,result,current,prefix='performance')
    fmt=lambda k:'—' if metrics[k] is None else f'{metrics[k]:.2%}'
    kpis([('Accuracy',fmt('accuracy'),'Correct / evaluated records'),('Precision',fmt('precision'),'True positives / positive predictions'),
          ('Recall',fmt('recall'),'True positives / actual positives'),('Evaluated samples',f"{result['n']:,}",'Current selected split and population')])
    left,right=st.columns([1.1,1],gap='small')
    with left,st.container(border=True,key='panel_performance_1'):
        a,b=st.columns([2,1])
        with a:view=st.selectbox('Evaluation diagnostic',['Confusion matrix','ROC curve','Precision–recall curve','Calibration','Threshold sensitivity'],key='performance_view')
        with b:normalized=st.selectbox('Confusion display',['Records','Within actual class'],disabled=view!='Confusion matrix',key='confusion_mode')
        if view=='Confusion matrix':
            fig=matrix(result['confusion'],normalized!='Records');note='Rows are actual income; percentages use actual-class denominators.'
        elif view=='Calibration':
            if result['n']:
                data,values=calibration_diagnostics(result['actual'],result['scores'])
                fig=px.scatter(data,x='mean_score',y='observed_fraction',size='count',color_discrete_sequence=[TEAL],hover_data=['count'],labels={'mean_score':'Mean uncalibrated score','observed_fraction':'Observed positive share'})
                fig.add_shape(type='line',x0=0,y0=0,x1=1,y1=1,line=dict(dash='dot',color=AMBER));fig.update_xaxes(range=[0,1]);fig.update_yaxes(range=[0,1])
                note=f"Brier {values['brier_score']:.3f} · ECE {values['ece_10_equal_width_bins']:.3f}. No calibrator fitted."
            else:fig=matrix(result['confusion']);note='No observations available.'
        else:
            kind={'ROC curve':'ROC','Precision–recall curve':'Precision–recall','Threshold sensitivity':'Threshold'}[view]
            fig=diagnostic(result,kind,current.threshold);note='Threshold scenarios do not select a policy or overwrite the original0.50 rule.'
        plot(fig,f'{view} computed from the current model, split, segment and threshold','performance_diagnostic',270)
        st.caption(note)
    with right,st.container(border=True,key='panel_performance_2'):
        st.subheader('Evaluation detail',icon=':material/query_stats:')
        detail=st.segmented_control('View',['Models','Classification report','Errors'],default='Models',key='performance_detail',label_visibility='collapsed') or 'Models'
        if detail=='Models':
            fig,note=comparison(snapshot,scored,current)
            plot(fig,'Model comparison with explicitly matched comparison scope','performance_cv',277)
            st.caption(note)
        elif detail=='Classification report':
            report=class_report(result)
            st.dataframe(report.style.format({'Precision':'{:.2%}','Recall':'{:.2%}','F1':'{:.3f}','Support':'{:,.0f}'}),hide_index=True,height=160,width='stretch',alt='Classification precision, recall, F1 and support for each income class')
            st.caption('Each class is treated as positive in its row. Zero predicted support produces zero precision under the documented zero-division convention.')
            st.download_button('Export classification report',report.to_csv(index=False),'classification_report.csv','text/csv',key='class_report_export',icon=':material/download:')
        else:
            counts=result['counts'];data=pd.DataFrame({'Error':['False positives','Missed positives'],'Records':[counts['fp'],counts['fn']]})
            fig=px.bar(data,x='Error',y='Records',color='Error',text='Records',color_discrete_sequence=[CORAL,PURPLE]);fig.update_traces(texttemplate='%{y:,}',textposition='outside',cliponaxis=False);fig.update_layout(showlegend=False,xaxis_title=None)
            plot(fig,'False-positive and false-negative counts for the selected evaluation context','performance_errors',245)
            with st.popover('Inspect public-source error records'):
                mask=result['prediction']!=result['actual']
                data=result['records'].loc[mask,['age','education','occupation','sex','race','income']].copy()
                data['Scenario score']=result['scores'][mask]
                data['Predicted income']=np.where(result['prediction'][mask]==1,'>50K','<=50K')
                st.dataframe(data.head(50),hide_index=True,height=260,alt='Up to50 misclassified historical public UCI records')
                st.download_button('Export current errors',data.to_csv(index=False),'census_errors.csv','text/csv',key='errors_export')
    st.caption('Computed dashboard diagnostic · development scores are in-sample, not CV · historical test data has already been inspected · no new independent validation claim.')
