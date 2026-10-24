"""Compact hypothetical input grid; inference uses the frozen snapshot only."""
import json
import pandas as pd

import plotly.graph_objects as go
import streamlit as st

from analysis_controls import context,save
from census_income.snapshot_inference import predict_snapshot_records
from ui import header,plot,detail_button,BLUE,AMBER,SLATE,INK

EDUCATION={1:'Preschool',2:'Grades 1–4',3:'Grades 5–6',4:'Grades 7–8',5:'Grade 9',6:'Grade 10',7:'Grade 11',8:'Grade 12',9:'High school graduate',10:'Some college',11:'Associate — vocational',12:'Associate — academic',13:"Bachelor’s degree",14:"Master’s degree",15:'Professional school',16:'Doctorate'}
OTHER='Other / unseen category'


def reset_inputs():
    for key in list(st.session_state):
        if str(key).startswith('hyp_input_') or key=='hypothetical_result':
            del st.session_state[key]


def render(snapshot):
    header('Prediction sandbox','Explore a hypothetical profile with the frozen historical pipeline. No profile is written to disk or sent elsewhere.',6)
    a,b,c,d=st.columns([2,2,1,1],vertical_alignment='bottom')
    with a:st.caption('RESEARCH DEMONSTRATION · Validated Random Forest · original policy0.50 · educational use only')
    with b:scenario=st.slider('Scenario threshold',.05,.95,float(context().threshold),.05,key='sandbox_threshold',on_change=lambda:st.session_state.analysis_context.update(threshold=st.session_state.sandbox_threshold))
    with c:st.button('Reset inputs',key='reset_hypothetical',on_click=reset_inputs,icon=':material/restart_alt:')
    with d:detail_button(snapshot,None,key='sandbox_details')
    manifest=json.loads(snapshot.bytes('artifacts/model_manifest.json'))
    levels=manifest['categorical_levels']
    previous=st.session_state.get('hypothetical_result',{})
    if previous and 'inputs' not in previous:
        st.session_state.pop('hypothetical_result',None)
        previous={}
    saved=previous.get('inputs',{})
    saved_country=saved.get('native-country','United-States')
    country_options=levels['native-country']+[OTHER]
    country_default=saved_country if saved_country in levels['native-country'] else OTHER
    left,right=st.columns([1.5,1],gap='medium')
    with left,st.container(border=True,key='panel_sandbox_1'):
        st.subheader('Hypothetical attributes')
        with st.form('hypothetical_record',border=False):
            inputs={}
            a,b,c=st.columns(3)
            with a:inputs['age']=st.number_input('Age',17,90,int(saved.get('age',35)),key='hyp_input_age')
            with b:inputs['hours-per-week']=st.number_input('Hours per week',1,99,int(saved.get('hours-per-week',40)),key='hyp_input_hours')
            with c:inputs['education-num']=st.selectbox('Education level',list(EDUCATION),index=list(EDUCATION).index(int(saved.get('education-num',10))),format_func=EDUCATION.get,key='hyp_input_education')
            a,b,c=st.columns(3)
            with a:inputs['capital-gain']=st.number_input('Capital gain ($)',0,99999,int(saved.get('capital-gain',0)),key='hyp_input_gain',help='Historical source dollars, not current income')
            with b:inputs['capital-loss']=st.number_input('Capital loss ($)',0,4356,int(saved.get('capital-loss',0)),key='hyp_input_loss')
            with c:inputs['workclass']=st.selectbox('Workclass',levels['workclass'],index=levels['workclass'].index(saved.get('workclass','Private')),format_func=lambda x:x.replace('-',' '),key='hyp_input_workclass')
            a,b,c=st.columns(3)
            for column,field,label,default in [(a,'occupation','Occupation','Prof-specialty'),(b,'marital-status','Marital status','Never-married'),(c,'relationship','Relationship','Not-in-family')]:
                with column:inputs[field]=st.selectbox(label,levels[field],index=levels[field].index(saved.get(field,default)),format_func=lambda x:x.replace('-',' '),key='hyp_input_'+field)
            a,b=st.columns([1.2,1])
            with a:country=st.selectbox('Native country',country_options,index=country_options.index(country_default),format_func=lambda x:x.replace('-',' '),key='hyp_input_country')
            with b:unknown=st.text_input('Other country (optional)',value=saved_country if country_default==OTHER else '',max_chars=80,key='hyp_input_other_country',help='Used only when Native country is Other / unseen category. Unseen inputs produce an encoder warning.')
            inputs['native-country']=unknown.strip() if country==OTHER else country
            st.caption('Historical UCI categories are retained; labels describe the source, not all current identities.')
            submitted=st.form_submit_button('Classify hypothetical record',key='classify_record',type='primary',width='stretch',icon=':material/play_arrow:')
        if submitted:
            try:
                if country==OTHER and not unknown.strip():raise ValueError('Enter an Other country value or select a known country.')
                value=predict_snapshot_records(snapshot,[inputs]).iloc[0].to_dict()
                st.session_state.hypothetical_result={'signature':snapshot.signature,'inputs':dict(inputs),**value}
            except (ValueError,TypeError,OSError) as exc:
                st.session_state.pop('hypothetical_result',None)
                st.error(f'Prediction unavailable: {exc}')
    with right,st.container(border=True,key='panel_sandbox_2'):
        st.subheader('Last submitted result')
        result=st.session_state.get('hypothetical_result')
        if result and result['signature']!=snapshot.signature:
            st.session_state.pop('hypothetical_result',None);result=None
        st.caption('Saved model only · no fitting or threshold selection in this interface')
        score=float(result['model_score']) if result else None
        fig=go.Figure()
        if score is not None:
            fig.add_trace(go.Bar(x=[score],y=[''],orientation='h',marker_color=BLUE,text=[f'{score:.3f}'],textposition='outside',cliponaxis=False))
        else:
            fig.add_annotation(x=.5,y=.5,xref='paper',yref='paper',text='Submit a profile to inspect its score',showarrow=False,font=dict(color=SLATE,size=13))
        fig.add_vline(x=scenario,line_dash='dot',line_color=AMBER)
        fig.update_xaxes(range=[0,1.08],tickvals=[0,.25,.5,.75,1],title='Uncalibrated score for >$50K')
        fig.update_yaxes(showticklabels=False,showgrid=False)
        fig.update_layout(showlegend=False)
        plot(fig,'Uncalibrated historical model score and selected scenario decision boundary','sandbox_score',230)
        if result:
            st.metric('Model score · >$50K',f'{score:.3f}',help='A model output, not a calibrated probability of current income')
            label='>$50K' if score>scenario else '≤$50K'
            st.markdown(f'Scenario classification at {scenario:.2f}: **{label}**')
            st.caption(f"Saved0.50-policy classification: {result['income_category']}. The scenario does not alter the fitted model.")
            with st.popover('Submitted profile',icon=':material/person:'):
                labels={'age':'Age','hours-per-week':'Hours per week','education-num':'Education',
                        'capital-gain':'Capital gain (historical $)','capital-loss':'Capital loss (historical $)',
                        'workclass':'Workclass','occupation':'Occupation','marital-status':'Marital status',
                        'relationship':'Relationship','native-country':'Native country'}
                submitted_inputs=result['inputs']
                rows=[{'Attribute':label,'Value':str(EDUCATION[value]) if key=='education-num' else str(value)} for key,label in labels.items() for value in [submitted_inputs[key]]]
                st.dataframe(pd.DataFrame(rows),hide_index=True,height=280,width='stretch',alt='Exact hypothetical attributes submitted for the displayed result')
                st.caption('Submitted values stay in this session until reset; no profile is written to disk.')
            payload={'snapshot_signature':snapshot.signature,'run_id':snapshot.manifest['run_id'],'model':'Validated Random Forest','input':result['inputs'],'uncalibrated_score':score,'scenario_threshold':scenario,'scenario_class':label,'validated_class_at_0_50':result['income_category'],'local_explanation':'Not available; no validated SHAP/local contribution artifact'}
            st.download_button('Export submitted result',json.dumps(payload,indent=2),'hypothetical_result.json','application/json',key='sandbox_export',icon=':material/download:')
            st.caption('Local contributions are not available; global permutation importance cannot explain this individual output.')
            st.caption('This classification is neutral; neither outcome represents a desirable or recommended result.')
            if result['input_warnings']:st.warning(result['input_warnings'])
        else:st.caption('No classification yet. Values reflect a hypothetical profile, not a prediction of current income.')
    st.caption('Do not use for employment, lending, eligibility or other consequential decisions. Historical 1994 data and uncalibrated scores cannot support those uses.')
