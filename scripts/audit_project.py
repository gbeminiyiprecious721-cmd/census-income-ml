"""Read-only scientific release audit of saved artifacts; no estimator fitting.

Writes independent audit evidence to --output, never to the original reports.
"""
from pathlib import Path
import argparse,hashlib,json,re,sys
import numpy as np
import pandas as pd
from scipy.stats import rankdata
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from census_income.data import load_adult,split_data,split_independence
from census_income.config import MODEL_INPUTS
from census_income.inference import load_validated_model,predict_records
from census_income.evaluate import calibration_diagnostics,subgroup_metrics,threshold_diagnostics


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'docs/portfolio_audit/frozen')
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    summary=json.loads((ROOT/'reports/summary.json').read_text())
    frame,audit=load_adult();dev,test,yd,yt=split_data(frame)
    model=load_validated_model();score=model.predict_proba(test)[:,1];pred=model.predict(test)
    assert np.array_equal(pred,(score>.5).astype(int))
    truth=yt.to_numpy();tn=int(((truth==0)&(pred==0)).sum());fp=int(((truth==0)&(pred==1)).sum())
    fn=int(((truth==1)&(pred==0)).sum());tp=int(((truth==1)&(pred==1)).sum())
    npos=int(truth.sum());nneg=len(truth)-npos
    metrics={'accuracy':(tn+tp)/len(truth),'precision':tp/(tp+fp),'recall':tp/(tp+fn),
             'f1':2*tp/(2*tp+fp+fn),
             'roc_auc':float((rankdata(score)[truth==1].sum()-npos*(npos+1)/2)/(npos*nneg))}
    for key,value in metrics.items():assert np.isclose(value,summary['test_metrics'][key],atol=1e-12,rtol=0),key
    for key,value in summary['selected_parameters'].items():assert model['forest'].get_params()[key.replace('forest__','')]==value,key
    direct=predict_records(test[MODEL_INPUTS])
    np.testing.assert_allclose(direct.model_score,score,atol=1e-12,rtol=0)
    np.testing.assert_array_equal(direct.predicted_class,pred)
    _,calibration=calibration_diagnostics(yt,score)
    original=json.loads((ROOT/'reports/calibration_metrics.json').read_text())
    for k,v in calibration.items():
        if isinstance(v,float):assert np.isclose(v,original[k],atol=1e-12,rtol=0),k
    groups=subgroup_metrics(yt,pred,score,test[['sex','race']])
    saved=pd.read_csv(ROOT/'reports/subgroup_metrics.csv')
    for key in ['attribute','group']:assert groups[key].tolist()==saved[key].tolist()
    numeric=list(groups.select_dtypes('number'))
    np.testing.assert_allclose(groups[numeric],saved[numeric],atol=1e-12,rtol=0,equal_nan=True)
    for row in groups.itertuples():
        mask=test[row.attribute].eq(row.group).to_numpy()
        assert row.n==mask.sum() and row.positives==truth[mask].sum()
        assert np.isclose(row.fpr,((pred[mask]==1)&(truth[mask]==0)).sum()/row.negatives)
    thresholds=threshold_diagnostics(yt,score)
    np.testing.assert_allclose(thresholds.select_dtypes('number'),pd.read_csv(ROOT/'reports/threshold_diagnostics.csv').select_dtypes('number'),atol=1e-12,rtol=0)
    importance=pd.read_csv(ROOT/'reports/importance_comparison.csv').set_index('feature')
    for metric in ['f1','roc_auc']:
        repeats=pd.read_csv(ROOT/f'reports/permutation_{metric}_repeats.csv')
        for col in repeats:
            assert np.isclose(repeats[col].mean(),importance.loc[col,metric+'_decrease_mean'])
            assert np.isclose(repeats[col].std(ddof=0),importance.loc[col,metric+'_shuffle_sd'])
    sensitive=test[['sex','race']].copy();sensitive['sex']=sensitive['sex']+' × '+sensitive['race']
    intersections=subgroup_metrics(yt,pred,score,sensitive)
    intersections=intersections[intersections.attribute=='sex'].assign(attribute='sex × race')
    intersections.to_csv(out/'intersectional_diagnostics.csv',index=False)
    groups.to_csv(out/'subgroup_verified.csv',index=False)
    nb=json.loads((ROOT/'notebooks/01_income_classification_research.ipynb').read_text(encoding='utf-8-sig'))
    codes=[c for c in nb['cells'] if c['cell_type']=='code']
    assert [c['execution_count'] for c in codes]==list(range(1,len(codes)+1))
    assert not any(o.get('output_type')=='error' for c in codes for o in c.get('outputs',[]))
    codehash=hashlib.sha256('\n'.join(''.join(c['source']) for c in codes).encode()).hexdigest()
    assert codehash==summary['notebook_execution']['code_sha256']
    record={'saved_artifact_metrics_verified':metrics,'confusion':dict(tn=tn,fp=fp,fn=fn,tp=tp),
            'rows':len(frame),'test_rows':len(test),'split':split_independence(frame,dev,test),
            'calibration':calibration,'subgroup_rows_verified':len(groups),
            'intersectional_groups':len(intersections),'minimum_intersection_n':int(intersections.n.min()),
            'threshold_rows_verified':len(thresholds),'permutation_summary_verified':True,
            'notebook_code_sha256':codehash,'notebook_code_cells':len(codes),
            'interpretation':'Saved model reproduction, not independent validation. Intersectional estimates are descriptive with small denominators; no fairness certification.',
            'training_performed':False}
    (out/'verification.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))


if __name__=='__main__':main()
