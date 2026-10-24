"""Development-only model selection and trusted local artifact creation."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import importlib.metadata as metadata
import json
import platform
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate, RandomizedSearchCV
from . import __version__
from .config import SEED, FEATURE_SETS, SCORING, ARTIFACTS_DIR, REPORTS_DIR, MODEL_INPUTS
from .preprocessing import make_pipeline


def stratified_cv():
    return StratifiedKFold(n_splits=5,shuffle=True,random_state=SEED)


def compare_features(X_dev,y_dev):
    rows,folds=[],[]
    for name in FEATURE_SETS:
        print('Cross-validating',name,flush=True)
        result=cross_validate(make_pipeline(name,n_estimators=100,max_depth=16),X_dev,y_dev,
            cv=stratified_cv(),scoring=SCORING,n_jobs=1,return_train_score=True,error_score='raise')
        rows.append({'model':name,**{m:float(result['test_'+m].mean()) for m in SCORING},
                     'f1_std':float(result['test_f1'].std(ddof=1)), 'train_f1':float(result['train_f1'].mean())})
        folds.extend([{'model':name,'fold':i+1,**{m:float(result['test_'+m][i]) for m in SCORING}} for i in range(5)])
    return pd.DataFrame(rows).sort_values('f1',ascending=False),pd.DataFrame(folds)


def search_model(X_dev,y_dev,feature_group,n_iter=12):
    search=RandomizedSearchCV(make_pipeline(feature_group),param_distributions={
        'forest__n_estimators':[100,200,300], 'forest__max_depth':[8,12,16,20,None],
        'forest__max_features':['sqrt',.5,1.0], 'forest__min_samples_split':[2,5,10],
        'forest__min_samples_leaf':[1,2,4,8], 'forest__class_weight':[None,'balanced','balanced_subsample']},
        n_iter=n_iter,scoring=SCORING,refit='f1',cv=stratified_cv(),random_state=SEED,n_jobs=1,
        verbose=2,return_train_score=True,error_score='raise')
    search.fit(X_dev,y_dev)
    return search


def dependency_versions():
    return {name:metadata.version(name) for name in ['scikit-learn','numpy','pandas','scipy','joblib']}


def save_model(model, model_metadata, artifact_dir=ARTIFACTS_DIR):
    # Refuse artifacts that the public inference contract cannot consume, before
    # touching an existing trusted release. Research comparisons may use sex/race;
    # the final educational inference interface deliberately does not accept them.
    if not np.array_equal(model.classes_,[0,1]):
        raise ValueError('Saved inference requires target classes [0, 1]')
    pre=model['preprocess']
    required={column for _,transformer,columns in pre.transformers_
              if not (isinstance(transformer,str) and transformer=='drop')
              for column in columns}
    available=set(MODEL_INPUTS)|{'education_bin','hours_age_interaction'}
    unsupported=required-available
    if unsupported:
        raise ValueError(f'Model input contract cannot supply {sorted(unsupported)}; '
                         'research-only feature groups cannot be promoted to this inference interface')
    artifact_dir=Path(artifact_dir); artifact_dir.mkdir(parents=True,exist_ok=True)
    model_path=artifact_dir/'model.joblib' 
    joblib.dump(model,model_path,compress=3)
    cats={}
    pre=model['preprocess']
    encoder=pre.named_transformers_['categorical']['encode']
    for col,values in zip(pre.transformers_[1][2],encoder.categories_):
        if col in MODEL_INPUTS: cats[col]=[str(v) for v in values]
    manifest={'artifact_format':'census-income-pipeline-v1','package_version':__version__,
        'model_version':model_metadata.get('run_id','local-validated-refit'),
        'created_utc':datetime.now(timezone.utc).isoformat(),'python':platform.python_version(),
        'dependencies':dependency_versions(),'sha256':hashlib.sha256(model_path.read_bytes()).hexdigest(),
        'bytes':model_path.stat().st_size,'model_inputs':MODEL_INPUTS,'categorical_levels':cats,
        'decision_rule':'RandomForest argmax; class 0 on exact probability ties',
        'source_year':1994,'calibrated':False,'metadata':model_metadata}
    (artifact_dir/'model_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['reproduce','search'],default='reproduce')
    parser.add_argument('--download',action='store_true')
    args=parser.parse_args()
    from .data import load_adult,split_data
    from .evaluate import metric_summary
    df,audit=load_adult(download=args.download)
    dev,test,ydev,ytest=split_data(df)
    if args.mode=='search':
        comparison,folds=compare_features(dev,ydev)
        group=str(comparison.iloc[0].model)
        search=search_model(dev,ydev,group)
        model=search.best_estimator_; params=search.best_params_
        REPORTS_DIR.mkdir(exist_ok=True)
        comparison.to_csv(REPORTS_DIR/'feature_comparison.csv',index=False)
        folds.to_csv(REPORTS_DIR/'feature_comparison_folds.csv',index=False)
        pd.DataFrame(search.cv_results_).to_csv(REPORTS_DIR/'random_search.csv',index=False)
    else:
        reference=json.loads((ARTIFACTS_DIR/'historical/results.json').read_text())
        group=reference['selected_features']; params=reference['best_parameters']
        model=make_pipeline(group,**{k.replace('forest__',''):v for k,v in params.items()}).fit(dev,ydev)
    metrics=metric_summary(ytest,model.predict(test),model.predict_proba(test)[:,1])
    manifest=save_model(model,{'run_id':datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'),
        'training_mode':args.mode,'selected_feature_group':group,'parameters':params,'test_metrics':metrics,
        'dataset':audit})
    (ARTIFACTS_DIR/'training_run.json').write_text(json.dumps({'mode':args.mode,'test_metrics':metrics,
        'model_sha256':manifest['sha256'],'note':'For the complete research reports, execute the notebook.'},indent=2))
    print(json.dumps(metrics,indent=2))


if __name__=='__main__': main()
