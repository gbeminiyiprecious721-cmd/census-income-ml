"""Read-only dashboard diagnostics. Never fit or save a model/research output."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import math

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve

from census_income.config import MODEL_INPUTS
from census_income.data import split_data
from census_income.snapshot_inference import predict_snapshot_records

MODELS=['Validated Random Forest','Majority baseline']
SEGMENTS=['All individuals','Female','Male']
SPLITS={'test':'Held-out test','development':'Development · in-sample'}
METRICS={'Recall':'recall','Precision':'precision','Accuracy':'accuracy','F1':'f1','ROC-AUC':'roc_auc'}


@dataclass(frozen=True)
class AnalysisContext:
    model: str='Validated Random Forest'
    metric: str='Recall'
    split: str='test'
    segment: str='All individuals'
    threshold: float=.5
    comparison: str='Matched development CV'

    def __post_init__(self):
        if not math.isfinite(self.threshold) or not 0<=self.threshold<=1:
            raise ValueError('Threshold must be finite and in [0,1]')
        if self.model not in MODELS or self.metric not in METRICS or self.split not in SPLITS or self.segment not in SEGMENTS:
            raise ValueError('Unknown analysis context')
        if self.comparison not in ['Matched development CV','Current evaluation']:
            raise ValueError('Unknown comparison scope')


def score_snapshot(snapshot) -> pd.DataFrame:
    """Score all cleaned source records once; retain the original seeded split."""
    frame,_=snapshot.source_data()
    development,test,_,_=split_data(frame)
    scored=frame.copy()
    scored['actual']=scored.income.eq('>50K').astype(int)
    scored['score']=predict_snapshot_records(snapshot,scored[MODEL_INPUTS]).model_score
    scored['split']='development'
    scored.loc[test.index,'split']='test'
    assert len(development)+len(test)==len(scored)
    return scored


def metrics_at(actual: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    if not math.isfinite(threshold) or not 0<=threshold<=1:raise ValueError('Invalid threshold')
    y=np.asarray(actual,dtype=int);scores=np.asarray(scores,dtype=float)
    if len(y)!=len(scores) or not np.isfinite(scores).all() or not np.isin(y,[0,1]).all() or not ((scores>=0)&(scores<=1)).all():
        raise ValueError('Invalid target/score vectors')
    prediction=(scores>threshold).astype(int)
    tn=int(((y==0)&(prediction==0)).sum());fp=int(((y==0)&(prediction==1)).sum())
    fn=int(((y==1)&(prediction==0)).sum());tp=int(((y==1)&(prediction==1)).sum())
    n=len(y)
    ratio=lambda a,b:float(a/b) if b else 0.0
    metrics={'accuracy':ratio(tn+tp,n),'precision':ratio(tp,tp+fp),'recall':ratio(tp,tp+fn),
        'f1':ratio(2*tp,2*tp+fp+fn),'roc_auc':float(roc_auc_score(y,scores)) if len(np.unique(y))==2 else None}
    if not n:metrics={key:None for key in metrics}
    confusion=pd.DataFrame([{'actual':a,'predicted':b,'count':count} for a,b,count in [(0,0,tn),(0,1,fp),(1,0,fn),(1,1,tp)]])
    return {'n':n,'metrics':metrics,'counts':{'tn':tn,'fp':fp,'fn':fn,'tp':tp},'prediction':prediction,'actual':y,'scores':scores,'confusion':confusion}


def evaluate_context(scored: pd.DataFrame, context: AnalysisContext) -> dict:
    frame=scored[scored.split.eq(context.split)]
    if context.segment!='All individuals':frame=frame[frame.sex.eq(context.segment)]
    scores=frame.score.to_numpy() if context.model==MODELS[0] else np.zeros(len(frame))
    return {**metrics_at(frame.actual.to_numpy(),scores,context.threshold),'records':frame.copy()}


def threshold_curve(actual: np.ndarray, scores: np.ndarray) -> pd.DataFrame:
    rows=[]
    for threshold in np.round(np.arange(.05,1,.05),2):
        result=metrics_at(actual,scores,float(threshold))
        rows.append({'threshold':float(threshold),**result['metrics'],'predicted_positive':result['counts']['fp']+result['counts']['tp']})
    return pd.DataFrame(rows)


def curve_data(result: dict, kind: str) -> pd.DataFrame:
    if len(np.unique(result['actual']))<2:return pd.DataFrame()
    if kind=='ROC':
        fpr,tpr,_=roc_curve(result['actual'],result['scores'])
        return pd.DataFrame({'False-positive rate':fpr,'True-positive rate':tpr})
    precision,recall,_=precision_recall_curve(result['actual'],result['scores'])
    return pd.DataFrame({'Recall':recall,'Precision':precision})


def class_report(result: dict) -> pd.DataFrame:
    counts=result['counts'];tn,fp,fn,tp=(counts[k] for k in ['tn','fp','fn','tp'])
    divide=lambda a,b: a/b if b else 0.0
    return pd.DataFrame([
        {'Class':'≤$50K','Precision':divide(tn,tn+fn),'Recall':divide(tn,tn+fp),'F1':divide(2*tn,2*tn+fp+fn),'Support':tn+fp},
        {'Class':'>$50K','Precision':divide(tp,tp+fp),'Recall':divide(tp,tp+fn),'F1':divide(2*tp,2*tp+fp+fn),'Support':tp+fn}])


def export_analysis(result: dict, context: AnalysisContext, signature: str) -> dict:
    return {'generated_utc':datetime.now(timezone.utc).isoformat(),
        'scope':'dashboard scenario; original research artifacts unchanged',
        'snapshot_signature':signature,'context':asdict(context),'evaluated_rows':result['n'],
        'metrics':result['metrics'],'confusion_counts':result['counts'],
        'interpretation':'Uncalibrated scores; strict score > threshold rule. Development scores are in-sample, not CV. Test scenarios are diagnostics, not policy selection or independent validation.'}
