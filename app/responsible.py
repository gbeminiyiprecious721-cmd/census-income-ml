"""Linked subgroup uncertainty and error composition with visible denominators."""
import plotly.express as px
import streamlit as st

from dashboard_data import interval_view
from analysis_controls import context,controls
from live_evaluation import evaluate_context
from census_income.evaluate import subgroup_metrics
import pandas as pd
from ui import header,kpis,plot,detail_button,BLUE,PURPLE,AMBER,TEAL,SLATE


def render(snapshot,scored):
    current=context()
    result=evaluate_context(scored,current)
    records=result['records']
    raw=subgroup_metrics(pd.Series(result['actual'],index=records.index),result['prediction'],result['scores'],records[['sex','race']])
    header('Responsible AI assessment','Compare descriptive error disparities with sample denominators and Wilson uncertainty. This does not establish fairness.',5)
    controls()
    st.caption(f"Dashboard scenario · {current.split} · {current.segment} · threshold {current.threshold:.2f} · n={result['n']:,}")
    a,b,c,d=st.columns([1.4,2,2,1.7],vertical_alignment='bottom')
    with a:attribute=st.selectbox('Demographic attribute',['sex','race'],format_func=str.title,key='demographic_attribute')
    with b:rate=st.selectbox('Error rate',['fpr','fnr'],format_func=lambda x:'False-positive rate' if x=='fpr' else 'False-negative rate',key='demographic_rate')
    data=raw[raw.attribute==attribute].copy()
    groups=data.group.tolist()
    # A group selection is page-local. Change of attribute safely resets membership.
    if st.session_state.get('demographic_group') not in groups:st.session_state.demographic_group=groups[0]
    with c:group=st.selectbox('Inspect group',groups,key='demographic_group')
    with d:detail_button(snapshot,'subgroup_metrics.csv',key='responsible_details')
    row=data[data.group==group].iloc[0]
    kpis([('Selected group records',f'{int(row.n):,}',f'{attribute}: {group}'),
          ('Actual positives',f'{int(row.positives):,}','Denominator for recall and false-negative rate'),
          ('Actual negatives',f'{int(row.negatives):,}','Denominator for false-positive rate'),
          ('Selected error rate',f'{row[rate]:.2%}','Descriptive rate at the selected scenario threshold')])
    left,right=st.columns([1.3,1],gap='medium')
    with left,st.container(border=True,key='panel_responsible_1'):
        st.subheader('Subgroup error rates')
        denominator='actual positives' if rate=='fnr' else 'actual negatives'
        st.caption(f'Wilson 95% intervals · denominators are {denominator} · common 0–100% scale')
        view=interval_view(data,rate)
        view['Label']=view.apply(lambda v:f"{v['group']} · n={int(v.Denominator):,}",axis=1)
        view['Selection']=view.group.map(lambda v:'Selected' if v==group else 'Context')
        fig=px.scatter(view,x='Rate',y='Label',error_x='Upper',error_x_minus='Lower',color='Selection',
            color_discrete_map={'Selected':PURPLE if rate=='fnr' else AMBER,'Context':SLATE},
            custom_data=['Denominator','Low','High'])
        fig.update_traces(marker_size=11,hovertemplate='%{y}<br>Rate %{x:.1%}<br>95% interval %{customdata[1]:.1%}–%{customdata[2]:.1%}<extra></extra>')
        fig.update_layout(showlegend=False,yaxis_title=None,xaxis_title='False-negative rate' if rate=='fnr' else 'False-positive rate')
        fig.update_xaxes(range=[0,1],tickformat='.0%')
        plot(fig,'Subgroup error rates with Wilson intervals and visible actual-class denominators','subgroup_intervals',260)
        st.caption('Intervals are marginal, not simultaneous; small groups have less precise estimates.')
    with right,st.container(border=True,key='panel_responsible_2'):
        st.subheader(f'Error composition · {group}')
        st.caption(f'Selected group n={int(row.n):,} · same scenario threshold rule')
        parts={'Correct negatives':int(row.tn),'False positives':int(row.fp),'Missed positives':int(row.fn),'Correct positives':int(row.tp)}
        breakdown=pd.DataFrame({'Outcome':list(parts),'Records':list(parts.values())})
        fig=px.bar(breakdown,x='Records',y='Outcome',orientation='h',text='Records',color='Outcome',color_discrete_map={'Correct negatives':BLUE,'False positives':AMBER,'Missed positives':PURPLE,'Correct positives':TEAL})
        fig.update_traces(texttemplate='%{x:,}',textposition='outside',cliponaxis=False)
        fig.update_layout(showlegend=False,yaxis=dict(title=None,autorange='reversed'),xaxis=dict(range=[0,max(parts.values())*1.22],title='Records'))
        plot(fig,'Selected demographic group confusion counts, linked to group selection','group_confusion',260)
        st.caption('FPR = false positives / actual negatives · FNR = missed positives / actual positives.')
    st.download_button('Export subgroup results',data.assign(scenario_threshold=current.threshold,model=current.model,dataset_split=current.split,population=current.segment,snapshot_signature=snapshot.signature).to_csv(index=False),'subgroup_scenario.csv','text/csv',key='subgroup_export',icon=':material/download:')
    observed=interval_view(data,rate)
    gap=observed.Rate.max()-observed.Rate.min() if len(data)>1 else None
    st.caption(f'Observed group range: {gap:.1%} points; descriptive, not a universal fairness score.' if gap is not None else 'No cross-group comparator in the selected population.')
    st.caption('Historical sex/race coding · unweighted observations · incomplete intersectional analysis · disparities do not establish discrimination, explain causes or demonstrate fairness.')
