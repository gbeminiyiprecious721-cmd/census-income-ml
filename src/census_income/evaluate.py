"""Evaluation diagnostics only: no threshold fitting or model selection."""
import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, brier_score_loss, log_loss


def metric_summary(y, prediction, probability):
    return {'accuracy':float(accuracy_score(y,prediction)),
            'precision':float(precision_score(y,prediction,zero_division=0)),
            'recall':float(recall_score(y,prediction,zero_division=0)),
            'f1':float(f1_score(y,prediction,zero_division=0)),
            'roc_auc':float(roc_auc_score(y,probability))}


def threshold_diagnostics(y, probability, thresholds=(.2,.35,.5,.65,.8)):
    rows = []
    for threshold in thresholds:
        if not 0 <= threshold <= 1: raise ValueError('Threshold must be in [0,1]')
        pred = (np.asarray(probability)>threshold).astype(int)
        tn,fp,fn,tp = confusion_matrix(y,pred,labels=[0,1]).ravel()
        rows.append({'threshold':threshold,**{k:v for k,v in metric_summary(y,pred,probability).items() if k!='roc_auc'},
            'selection_rate':float(pred.mean()),'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp)})
    return pd.DataFrame(rows)


def _ratio(n,d): return float(n/d) if d else float('nan')


def wilson_interval(num,den):
    if not den: return (float('nan'),float('nan'))
    z=norm.ppf(.975); p=num/den; divisor=1+z*z/den
    centre=(p+z*z/(2*den))/divisor
    half=z*np.sqrt(p*(1-p)/den+z*z/(4*den*den))/divisor
    return float(max(0,centre-half)),float(min(1,centre+half))


def subgroup_metrics(y, prediction, probability, sensitive):
    if isinstance(y,pd.Series) and not y.index.equals(sensitive.index):
        raise ValueError('Sensitive attributes must align exactly with target indices')
    y=np.asarray(y); prediction=np.asarray(prediction); probability=np.asarray(probability)
    if not len(y)==len(prediction)==len(probability)==len(sensitive): raise ValueError('Inconsistent subgroup input lengths')
    rows=[]
    for attribute in ['sex','race']:
        for group in sorted(sensitive[attribute].dropna().unique()):
            mask=sensitive[attribute].eq(group).to_numpy()
            truth=y[mask]; pred=prediction[mask]; prob=probability[mask]
            tn,fp,fn,tp=map(int,confusion_matrix(truth,pred,labels=[0,1]).ravel())
            positives=tp+fn; negatives=tn+fp; n=int(mask.sum())
            row={'attribute':attribute,'group':str(group),'n':n,'positives':positives,'negatives':negatives,
                 'tn':tn,'fp':fp,'fn':fn,'tp':tp,'prevalence':_ratio(positives,n),
                 'selection_rate':float(pred.mean()),'accuracy':_ratio(tn+tp,n),
                 'precision':_ratio(tp,tp+fp),'recall_tpr':_ratio(tp,positives),'fpr':_ratio(fp,negatives),
                 'fnr':_ratio(fn,positives),'f1':_ratio(2*tp,2*tp+fp+fn),
                 'roc_auc':float(roc_auc_score(truth,prob)) if positives and negatives else float('nan')}
            for metric,num,den in [('recall_tpr',tp,positives),('fpr',fp,negatives),('precision',tp,tp+fp)]:
                row[metric+'_ci_low'],row[metric+'_ci_high']=wilson_interval(num,den)
            rows.append(row)
    return pd.DataFrame(rows)


def calibration_diagnostics(y, probability, bins=10):
    """Fixed-bin diagnostic on the inspected test set; never fits a calibrator."""
    y=np.asarray(y); prob=np.asarray(probability)
    bucket=np.minimum((prob*bins).astype(int),bins-1)
    rows=[]
    for i in range(bins):
        mask=bucket==i
        if mask.any(): rows.append({'bin':i,'lower':i/bins,'upper':(i+1)/bins,'count':int(mask.sum()),
            'mean_score':float(prob[mask].mean()),'observed_fraction':float(y[mask].mean())})
    table=pd.DataFrame(rows)
    metrics={'brier_score':float(brier_score_loss(y,prob)),
             'log_loss':float(log_loss(y,prob,labels=[0,1])),
             'ece_10_equal_width_bins':float(np.average(abs(table.mean_score-table.observed_fraction),weights=table['count'])),
             'interpretation':'Descriptive test calibration diagnostic; no calibrator fitted and no calibrated probabilities claimed.'}
    return table,metrics


def export_evaluation(y,prediction,probability,sensitive,report_dir):
    from pathlib import Path
    import json
    out=Path(report_dir); out.mkdir(parents=True,exist_ok=True)
    thresholds=threshold_diagnostics(y,probability); thresholds.to_csv(out/'threshold_diagnostics.csv',index=False)
    groups=subgroup_metrics(y,prediction,probability,sensitive); groups.to_csv(out/'subgroup_metrics.csv',index=False)
    fpr,tpr,roc_threshold=roc_curve(y,probability)
    pd.DataFrame({'fpr':fpr,'tpr':tpr,'threshold':roc_threshold}).to_csv(out/'roc_curve.csv',index=False)
    precision,recall,pr_threshold=precision_recall_curve(y,probability)
    pd.DataFrame({'precision':precision,'recall':recall,'threshold':np.r_[pr_threshold,np.nan]}).to_csv(out/'precision_recall_curve.csv',index=False)
    cm=confusion_matrix(y,prediction,labels=[0,1])
    pd.DataFrame([{'actual':a,'predicted':p,'count':int(cm[a,p])} for a in range(2) for p in range(2)]).to_csv(out/'confusion_matrix.csv',index=False)
    calibration,calibration_metrics=calibration_diagnostics(y,probability)
    calibration.to_csv(out/'calibration.csv',index=False)
    (out/'calibration_metrics.json').write_text(json.dumps(calibration_metrics,indent=2),encoding='utf-8')
    tn,fp,fn,tp=cm.ravel()
    metrics=metric_summary(y,prediction,probability)
    assert np.isclose(metrics['accuracy'],(tn+tp)/cm.sum())
    assert np.isclose(metrics['f1'],2*tp/(2*tp+fp+fn))
    truth=np.asarray(y); npos=int(truth.sum()); nneg=len(truth)-npos
    rank_auc=(rankdata(probability)[truth==1].sum()-npos*(npos+1)/2)/(npos*nneg)
    assert np.isclose(rank_auc,metrics['roc_auc'])
    for attribute in ['sex','race']:
        assert np.array_equal(groups[groups.attribute==attribute][['tn','fp','fn','tp']].sum().to_numpy(),cm.ravel())
    return metrics,thresholds,groups,calibration,calibration_metrics
