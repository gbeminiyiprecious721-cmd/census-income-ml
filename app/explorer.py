"""Linked cohort exploration using only the selected snapshot's full source."""
import json
import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard_data import cohort,feature_distribution,selected_income,EDUCATION_GROUPS
from ui import header,kpis,plot,detail_button,INCOME_COLORS,FEATURE_LABELS,BLUE,TEAL,PURPLE

DEFAULTS={'age':(17,90),'sex':'All','education':'All'}


def save_filter(name,key):
    st.session_state.explorer_filters[name]=st.session_state[key]


def reset_filters():
    st.session_state.explorer_filters=DEFAULTS.copy()
    st.session_state.explorer_income='All'
    st.session_state.explorer_epoch=st.session_state.get('explorer_epoch',0)+1
    for key in ['explorer_age','explorer_sex','explorer_education']:
        st.session_state.pop(key,None)


def income_selected(key):
    st.session_state.explorer_income=selected_income(st.session_state[key])



@st.dialog('Filtered public-source records',width='large')
def records_dialog(frame, signature, filter_context):
    st.caption('UCI Adult · CC BY4.0 · public historical records · no person identifiers. Filters match the source Explorer, not model evaluation.')
    search=st.text_input('Search records',key='records_search',placeholder='Search displayed categories or values…')
    visible=frame
    if search:
        mask=frame.astype('string').apply(lambda column:column.str.contains(search,case=False,regex=False,na=False)).any(axis=1)
        visible=frame.loc[mask]
    a,b=st.columns(2)
    with a:size=st.selectbox('Rows per page',[25,50,100],key='records_page_size')
    pages=max(1,(len(visible)+size-1)//size)
    if st.session_state.get('records_page',1)>pages:st.session_state.records_page=1
    with b:number=st.number_input('Page',1,pages,1,key='records_page')
    st.caption(f'{len(visible):,} matching records · page {number}/{pages}')
    st.dataframe(visible.iloc[(number-1)*size:number*size],hide_index=True,height=310,width='stretch',alt='Paginated public UCI source records matching the active Explorer and search filters')
    st.download_button('Download matching records',visible.to_csv(index=False),'census_filtered_source.csv','text/csv',key='source_records_export',icon=':material/download:')
    metadata={'source':'UCI Adult (1996), DOI10.24432/C5XW20','license':'CC BY4.0','source_year':1994,'snapshot_signature':signature,'filters':filter_context,'search':search,'rows':len(visible)}
    st.download_button('Download filter metadata',json.dumps(metadata,indent=2),'census_source_context.json','application/json',key='source_context_export')

def render(snapshot, source_data):
    frame,_=source_data()
    header('Data explorer','Filter the historical sample, then select an income bar to focus the linked feature profile.',2)
    filters=st.session_state.setdefault('explorer_filters',DEFAULTS.copy())
    a,b,c,d=st.columns([2.2,1.3,2,1],vertical_alignment='bottom')
    with a: age=st.slider('Age range',17,90,value=filters['age'],key='explorer_age',on_change=save_filter,args=('age','explorer_age'))
    with b: sex=st.selectbox('Sex',['All','Female','Male'],index=['All','Female','Male'].index(filters['sex']),key='explorer_sex',on_change=save_filter,args=('sex','explorer_sex'))
    with c: education=st.selectbox('Education group',['All',*EDUCATION_GROUPS],index=['All',*EDUCATION_GROUPS].index(filters['education']),key='explorer_education',on_change=save_filter,args=('education','explorer_education'))
    with d: st.button('Reset filters',key='reset_explorer',on_click=reset_filters,icon=':material/restart_alt:',width='stretch')
    base=cohort(frame,age,sex,education,'All')
    income=st.session_state.get('explorer_income','All')
    selected=cohort(frame,age,sex,education,income)
    n=len(selected); positive=int(selected.income.eq('>50K').sum())
    missing=int(selected.isna().any(axis=1).sum())
    kpis([('Cohort records',f'{n:,}','Current filters and income selection'),
          ('Share of full sample',f'{n/len(frame):.1%}','Denominator: all 48,790 cleaned records'),
          ('Income >$50K share',f'{positive/n:.1%}' if n else '—','Unweighted historical sample share; not a population estimate'),
          ('Records with missing inputs',f'{missing:,}','At least one missing source predictor')])
    left,right=st.columns([.8,1.4],gap='medium')
    with left,st.container(border=True,key='panel_explorer_1'):
        st.subheader('Income composition')
        st.caption('Click a bar to filter the feature profile; click blank space to clear.')
        counts=base.income.value_counts().reindex(['<=50K','>50K'],fill_value=0).rename_axis('Income').reset_index(name='Records')
        counts['Label']=counts.Income.replace({'<=50K':'≤$50K','>50K':'>$50K'})
        fig=px.bar(counts,x='Records',y='Label',color='Income',orientation='h',text='Records',custom_data=['Income'],color_discrete_map=INCOME_COLORS)
        fig.update_traces(texttemplate='%{x:,}',textposition='outside',cliponaxis=False)
        fig.update_layout(showlegend=False,yaxis_title=None,xaxis=dict(title='Records before income selection',range=[0,max(1,counts.Records.max()*1.2)]))
        key=f"explorer_income_chart_{st.session_state.get('explorer_epoch',0)}"
        plot(fig,'Income category counts before income selection; selecting one filters the linked profile',key,300,selectable=True,on_select=lambda:income_selected(key))
        st.caption(f'Base cohort n={len(base):,} · selected income: {income}')
    with right,st.container(border=True,key='panel_explorer_2'):
        st.subheader('Explore the selected sample',icon=':material/query_stats:')
        view=st.segmented_control('Explore',['Distributions','Missingness','Associations'],default='Distributions',key='explorer_view',label_visibility='collapsed') or 'Distributions'
        if view=='Distributions':
            a,b=st.columns([1,1])
            with a: feature=st.selectbox('Break down by',['age','hours-per-week','education','occupation','workclass','marital-status','race','sex'],format_func=lambda x:FEATURE_LABELS.get(x,x.replace('-',' ').title()),key='explorer_feature')
            with b: measure=st.selectbox('Display',['Records','Within-group income share'],key='explorer_measure')
            data=feature_distribution(selected,feature)
            if data.empty:
                st.info('No records match these filters. Reset filters or widen the selection.')
            else:
                totals=data.groupby('Category').Records.sum()
                shown=totals.nlargest(8).index if len(totals)>8 else totals.index
                data=data[data.Category.isin(shown)]
                axis='Share' if measure!='Records' else 'Records'
                fig=px.bar(data,x=axis,y='Category',color='Income',orientation='h',barmode='stack',
                    color_discrete_map=INCOME_COLORS,custom_data=['Records','Denominator','Share'])
                fig.update_traces(hovertemplate='%{y}<br>Records: %{customdata[0]:,}<br>Group n=%{customdata[1]:,}<br>Group share: %{customdata[2]:.1%}<extra>%{fullData.name}</extra>')
                fig.update_layout(yaxis_title=None,xaxis_title='Within-group share' if axis=='Share' else 'Records')
                if axis=='Share':fig.update_xaxes(range=[0,1],tickformat='.0%')
                plot(fig,'Income distribution by selected feature for the current intersected cohort','explorer_feature_chart',248)
            st.caption('Up to 8 categories shown · each share uses that category’s cohort denominator · unweighted sample')
        elif selected.empty:
            st.info('No records match these filters. Reset or widen your selection.')
        elif view=='Missingness':
            missing=selected.drop(columns='income').isna().sum().rename_axis('Feature').reset_index(name='Missing records')
            missing=missing[missing['Missing records']>0].sort_values('Missing records',ascending=False)
            if missing.empty:st.info('No missing source predictors in this cohort.')
            else:
                missing['Share']=missing['Missing records']/len(selected)
                fig=px.bar(missing,x='Share',y='Feature',orientation='h',text='Missing records',color_discrete_sequence=[TEAL])
                fig.update_xaxes(tickformat='.0%',range=[0,min(1,max(missing.Share)*1.3)])
                fig.update_layout(yaxis_title=None,xaxis_title='Share of cohort with missing input')
                plot(fig,'Source missingness by predictor for the current filtered cohort','explorer_missingness',248)
            st.caption('Per-field missingness counts can overlap; denominator is the selected cohort.')
        else:
            numeric=['age','education-num','hours-per-week','capital-gain','capital-loss']
            corr=selected[numeric].corr(method='spearman')
            corr=corr.rename(index=FEATURE_LABELS,columns=FEATURE_LABELS)
            fig=px.imshow(corr,zmin=-1,zmax=1,color_continuous_scale=[(0,PURPLE),(.5,'#FBFBF8'),(1,TEAL)],text_auto='.2f',aspect='auto')
            fig.update_layout(coloraxis_showscale=False)
            plot(fig,'Spearman rank correlations among numeric source inputs in the selected cohort','explorer_associations',248)
            st.caption('Spearman rank association · descriptive, not causal or model importance · blank values indicate insufficient variation.')
    a,b,c=st.columns([4,1.2,1.2])
    with a:st.caption(f'ACTIVE FILTERS · age {age[0]}–{age[1]} · sex {sex} · education {education} · income {income}. Public records are available in Browse & export.')
    with b:
        if st.button('Browse & export',key='explorer_records',icon=':material/table_view:'):
            records_dialog(selected,snapshot.signature,{'age':list(age),'sex':sex,'education':education,'income':income})
    with c:detail_button(snapshot,'eda_summary.json',key='explorer_details')
