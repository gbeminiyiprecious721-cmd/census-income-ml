"""Held-out permutation diagnostics, not feature selection or causal attribution."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score,roc_auc_score
from .config import FEATURE_SETS,SEED
from .features import engineer_features


def aggregated_impurity(model):
    pre=model['preprocess']; values=model['forest'].feature_importances_
    rows=[]; offset=0
    for col in pre.transformers_[0][2]:
        rows.append({'feature':col,'impurity_importance':float(values[offset])}); offset+=1
    encoder=pre.named_transformers_['categorical']['encode']
    for col,categories in zip(pre.transformers_[1][2],encoder.categories_):
        width=len(categories)
        rows.append({'feature':col,'impurity_importance':float(values[offset:offset+width].sum())}); offset+=width
    assert offset==len(values) and np.isclose(sum(r['impurity_importance'] for r in rows),1)
    return pd.DataFrame(rows)


def permutation_diagnostics(model,X_test,y_test,feature_group,report_dir,n_repeats=10):
    """Reproduce the reviewed transformed-column perturbation, plus coherent raw-source perturbations.

    The complete saved pipeline remains the inference object. The temporary
    inspection Pipeline starts at preprocessing to permit individual interaction
    perturbations without automatically recomputing that engineered column.
    """
    out=Path(report_dir); out.mkdir(exist_ok=True,parents=True)
    engineered=engineer_features(X_test)
    columns=FEATURE_SETS[feature_group]
    inspection_model=model[1:]
    results=permutation_importance(inspection_model,engineered[columns],y_test,n_repeats=n_repeats,
        random_state=SEED,n_jobs=1,scoring={'f1':'f1','roc_auc':'roc_auc'})
    table=pd.DataFrame({'feature':columns}).merge(aggregated_impurity(model),on='feature',validate='one_to_one')
    for metric,result in results.items():
        table[metric+'_decrease_mean']=result.importances_mean
        table[metric+'_shuffle_sd']=result.importances_std
        pd.DataFrame(result.importances.T,columns=columns).to_csv(out/f'permutation_{metric}_repeats.csv',index=False)
    table['impurity_rank']=table.impurity_importance.rank(ascending=False,method='min')
    table['f1_permutation_rank']=table.f1_decrease_mean.rank(ascending=False,method='min')
    table=table.sort_values('f1_decrease_mean',ascending=False)
    table.to_csv(out/'importance_comparison.csv',index=False)
    original=model.predict_proba(X_test)[:,1]
    original_f1=f1_score(y_test,model.predict(X_test)); original_auc=roc_auc_score(y_test,original)
    coherent=[]
    if 'hours_age_interaction' in columns:
        for source in ['age','hours-per-week']:
            rng=np.random.default_rng(SEED)
            for repeat in range(n_repeats):
                probe=X_test.copy()
                probe[source]=X_test[source].to_numpy()[rng.permutation(len(X_test))]
                prob=model.predict_proba(probe)[:,1]
                coherent.append({'source':source,'repeat':repeat+1,
                    'f1_decrease':float(original_f1-f1_score(y_test,model.predict(probe))),
                    'roc_auc_decrease':float(original_auc-roc_auc_score(y_test,prob))})
    coherent=pd.DataFrame(coherent); coherent.to_csv(out/'coherent_source_permutation.csv',index=False)
    return table,coherent
