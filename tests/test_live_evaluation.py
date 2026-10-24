"""Dashboard scenario arithmetic and scope; independent hand-calculated fixtures."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'app'))
from live_evaluation import AnalysisContext, metrics_at, evaluate_context, threshold_curve, export_analysis


def fixture_scores():
    return pd.DataFrame({'age':[20,30,40,50], 'sex':['Female','Male','Female','Male'],
        'income':['<=50K','>50K','>50K','<=50K'], 'actual':[0,1,1,0],
        'score':[.1,.8,.5,.6], 'split':['test']*4})


def test_strict_threshold_tie_and_confusion_arithmetic():
    result=metrics_at(np.array([0,1,1,0]),np.array([.1,.8,.5,.6]),.5)
    assert result['counts']=={'tn':1,'fp':1,'fn':1,'tp':1}
    assert result['metrics']['accuracy']==.5
    assert result['metrics']['precision']==.5
    assert result['metrics']['recall']==.5
    assert result['metrics']['f1']==.5
    higher=metrics_at(np.array([0,1,1,0]),np.array([.1,.8,.5,.6]),.7)
    assert higher['counts']=={'tn':2,'fp':0,'fn':1,'tp':1}
    assert higher['metrics']['precision']==1
    assert higher['metrics']['recall']==.5


def test_context_filters_sample_and_baseline_scores():
    frame=fixture_scores()
    result=evaluate_context(frame,AnalysisContext(segment='Female'))
    assert result['n']==2 and result['counts']=={'tn':1,'fp':0,'fn':1,'tp':0}
    baseline=evaluate_context(frame,AnalysisContext(model='Majority baseline'))
    assert baseline['counts']=={'tn':2,'fp':0,'fn':2,'tp':0}
    assert baseline['metrics']['roc_auc']==.5
    assert len(frame)==4


def test_empty_and_one_class_are_explicit_not_fake_auc():
    result=evaluate_context(fixture_scores(),AnalysisContext(split='development'))
    assert result['n']==0 and all(v is None for v in result['metrics'].values())
    result=metrics_at(np.array([0,0]),np.array([.1,.2]),.5)
    assert result['metrics']['roc_auc'] is None


def test_threshold_counts_are_monotonic_and_export_records_context():
    frame=fixture_scores()
    curve=threshold_curve(frame.actual.to_numpy(),frame.score.to_numpy())
    assert (np.diff(curve.predicted_positive)<=0).all()
    context=AnalysisContext(segment='Female',threshold=.7)
    report=export_analysis(evaluate_context(frame,context),context,'snapshot-123')
    assert report['context']['threshold']==.7 and report['context']['segment']=='Female'
    assert report['snapshot_signature']=='snapshot-123' and report['evaluated_rows']==2
    assert report['scope']=='dashboard scenario; original research artifacts unchanged'


@pytest.mark.parametrize('threshold',[-.01,1.01,float('nan')])
def test_invalid_threshold_is_rejected(threshold):
    with pytest.raises(ValueError):AnalysisContext(threshold=threshold)


def test_scored_default_matches_original_validated_results():
    from census_income.run_store import load_snapshot
    from live_evaluation import score_snapshot
    root=Path(__file__).resolve().parents[1]
    snapshot=load_snapshot(root/'artifacts/research_store')
    scored=score_snapshot(snapshot)
    result=evaluate_context(scored,AnalysisContext())
    assert result['n']==9758
    assert result['counts']=={'tn':6220,'fp':1202,'fn':435,'tp':1901}
    for key,value in snapshot.report('summary.json')['test_metrics'].items():
        assert result['metrics'][key]==pytest.approx(value,abs=1e-12)
