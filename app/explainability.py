"""Global saved importance evidence; view controls never imply a model refit."""
import plotly.express as px
import streamlit as st

from ui import header,kpis,plot,detail_button,BLUE,TEAL,PURPLE,AMBER,SLATE,FEATURE_LABELS

MEASURES={'f1_decrease_mean':'F1 score decrease','roc_auc_decrease_mean':'ROC-AUC decrease','impurity_importance':'Training impurity credit'}


def highlight_selection():
    event=st.session_state.importance_ranking
    points=event.get('selection',{}).get('points',[])
    if len(points)==1:
        st.session_state.importance_feature=points[0]['customdata'][0]


def render(snapshot):
    data=snapshot.report('importance_comparison.csv')
    header('Model explainability','Inspect saved model reliance. Rankings describe this model and sample; they do not identify causes of income.',4)
    a,b,c,d=st.columns([2,1.5,2,1.8],vertical_alignment='bottom')
    with a:measure=st.selectbox('Importance measure',list(MEASURES),format_func=MEASURES.get,key='importance_measure')
    with b:top=st.select_slider('Features shown',options=[5,8,10],value=8,key='importance_top_n')
    with c:feature=st.selectbox('Highlight feature',['All',*data.feature.tolist()],format_func=lambda x:FEATURE_LABELS.get(x,x),key='importance_feature')
    with d:detail_button(snapshot,'importance_comparison.csv',key='importance_details')
    ordered=data.sort_values(measure,ascending=False).head(top).copy()
    leader=ordered.iloc[0]
    kpis([('Largest measured reliance',FEATURE_LABELS.get(leader.feature,leader.feature),'Top input under the chosen importance method'),
          ('Inputs ranked',str(len(ordered)),'Display selection; full results available in Evidence & definitions'),
          ('Evaluation sample','9,758','Held-out sample for permutation; training split credit for impurity'),
          ('Interpretation','Model reliance','Correlation and derived inputs affect importance; no causal claim')])
    left,right=st.columns([1.35,1],gap='medium')
    with left,st.container(border=True,key='panel_explainability_1'):
        st.subheader('Feature ranking')
        st.caption('Click a bar to highlight its source input. Error bars show shuffle SD, not a sampling confidence interval.')
        ordered['Label']=ordered.feature.map(lambda v:FEATURE_LABELS.get(v,v))
        ordered['Emphasis']=ordered.feature.map(lambda v:'Selected' if feature=='All' or v==feature else 'Context')
        sd='f1_shuffle_sd' if measure=='f1_decrease_mean' else 'roc_auc_shuffle_sd' if measure=='roc_auc_decrease_mean' else None
        fig=px.bar(ordered,x=measure,y='Label',orientation='h',error_x=sd,
            color='Emphasis',color_discrete_map={'Selected':BLUE,'Context':SLATE},custom_data=['feature'])
        fig.update_layout(showlegend=False,xaxis_title=MEASURES[measure],yaxis=dict(title=None,categoryorder='array',categoryarray=ordered.Label.tolist(),autorange='reversed'))
        if sd:fig.update_traces(hovertemplate='%{y}<br>Mean decrease: %{x:.4f}<extra></extra>')
        plot(fig,'Selected importance ranking with available shuffle standard deviation','importance_ranking',300,selectable=True,on_select=highlight_selection)
        if feature!='All':
            row=data[data.feature==feature].iloc[0]
            st.caption(f"{FEATURE_LABELS.get(feature,feature)} · F1 decrease {row.f1_decrease_mean:.4f} · shuffle SD {row.f1_shuffle_sd:.4f}")
        else:st.caption('Impurity credit and held-out score decreases have different scales and meanings.')
    with right,st.container(border=True,key='panel_explainability_2'):
        st.subheader('Coherent source-input check')
        st.caption('Shuffling age or hours also recomputes their deterministic interaction.')
        raw=snapshot.report('coherent_source_permutation.csv')
        grouped=raw.groupby('source').agg(Mean=('f1_decrease','mean'),SD=('f1_decrease','std')).reset_index()
        grouped['Input']=grouped.source.map(lambda x:FEATURE_LABELS.get(x,x))
        fig=px.bar(grouped,x='Input',y='Mean',error_y='SD',color='Input',color_discrete_sequence=[PURPLE,TEAL])
        fig.update_layout(showlegend=False,xaxis_title=None,yaxis_title='Mean F1 decrease')
        fig.update_yaxes(range=[0,max(grouped.Mean+grouped.SD)*1.3])
        plot(fig,'Coherent source permutations of age and weekly hours, mean F1 decrease with shuffle SD','coherent_importance',300)
        st.caption('A supplementary joint-source diagnostic; not an interchangeable single-column ranking.')
    a,b=st.columns([1,3])
    with a:st.download_button('Export importance',ordered.drop(columns=['Emphasis','Label']).to_csv(index=False),'global_permutation_importance.csv','text/csv',key='importance_export',icon=':material/download:')
    with b:
        with st.popover('Global versus local explanations'):
            st.write('These saved permutation and impurity diagnostics explain global model reliance. They do not explain one individual prediction. No validated SHAP or local contribution artifact is available in this project; no local attribution is fabricated.')
    st.caption('GLOBAL EVIDENCE · full saved evaluation sample · no cohort filters or new permutation computations · feature importance is not an individual explanation.')
