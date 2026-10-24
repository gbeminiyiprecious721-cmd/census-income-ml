"""Small research reporting helpers; fresh measurements never come from literals."""
from datetime import datetime,timezone
from pathlib import Path
import hashlib,json,platform,sys
import numpy as np
import pandas as pd
from .config import ARTIFACTS_DIR,REPORTS_DIR,SEED,COLUMNS
from .train import dependency_versions,save_model

LIMITATIONS = [
    'Historical 1994 US Census data; no contemporary or temporal external evaluation.',
    'Five folds reused for feature-group selection and a 12-configuration F1 search; no nested CV.',
    'Two complete predictor profiles cross the preserved split with conflicting labels; person identifiers are unavailable.',
    'Scores are uncalibrated; threshold comparisons are diagnostics, not test-selected policies.',
    'Demographic error disparities and small subgroup denominators; incomplete intersectional evaluation.',
    'Correlated and derived inputs complicate importance interpretation; importance is not causal.',
    'The historical test set was previously inspected; this is a reproduction, not fresh independent confirmation.',
    'No production, fairness or consequential-use readiness is established.'
]


def write_json(path,content):
    Path(path).write_text(json.dumps(content,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)),encoding='utf-8')


def export_eda(frame,audit,report_dir=REPORTS_DIR):
    out=Path(report_dir); out.mkdir(exist_ok=True,parents=True)
    write_json(out/'data_quality.json',audit)
    summary={'rows':len(frame),'numeric_summary':frame.select_dtypes(include='number').describe().to_dict(),
        'income_counts':frame.income.value_counts().to_dict(),
        'sex_counts':frame.sex.value_counts().to_dict(),'race_counts':frame.race.value_counts().to_dict(),
        'note':'Aggregated historical sample statistics; unweighted by fnlwgt, not current population estimates.'}
    write_json(out/'eda_summary.json',summary)
    return summary


def finalize_research(model,feature_group,search,comparison,fold_scores,test_metrics,audit,independence,report_dir=REPORTS_DIR,reproduction_policy="strict"):
    """Reconcile a refit; strict refuses drift, record explicitly retains differences.

    Record mode is for separately labelled runs in an isolated project copy; it
    never means that the archived measurements were reproduced exactly.
    """
    if reproduction_policy not in {"strict","record"}:
        raise ValueError("Unknown reproduction policy")
    out=Path(report_dir); out.mkdir(exist_ok=True,parents=True)
    reference=json.loads((ARTIFACTS_DIR/'historical/results.json').read_text())
    prior_comparison=pd.read_csv(ARTIFACTS_DIR/'historical/feature_comparison.csv').set_index('model')
    new_comparison=comparison.set_index('model')
    changes={m:float(test_metrics[m]-reference['test_metrics'][m]) for m in test_metrics}
    cv_changes={m:float((new_comparison.loc[prior_comparison.index,m]-prior_comparison[m]).abs().max())
                for m in ['accuracy','precision','recall','f1','roc_auc']}
    reproduced=(all(abs(v)<1e-6 for v in changes.values()) and
        abs(search.best_score_-reference['best_cv_f1'])<1e-6 and
        all(v<1e-6 for v in cv_changes.values()) and search.best_params_==reference['best_parameters']
        and feature_group==reference['selected_features'])
    if not reproduced:
        write_json(out/'reproduction_difference.json',{'test_differences':changes,'feature_cv_differences':cv_changes,'parameters':search.best_params_})
        if reproduction_policy=='strict':
            raise AssertionError('Current refit differs from historical measurements beyond tolerance; inspect reproduction_difference.json before promotion')
    run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    selected={m:float(search.cv_results_['mean_test_'+m][search.best_index_]) for m in test_metrics}
    dataset={**audit,'independence':independence}
    # Sizes are reconciled to actual fold sample count by the caller's split audit.
    dataset['development_rows']=int(independence['development_rows'])
    dataset['test_rows']=int(independence['test_rows'])
    metadata={'run_id':run_id,'training_mode':'full development-only research search','selected_feature_group':feature_group,
              'parameters':search.best_params_,'test_metrics':test_metrics,'source_provenance':audit['provenance'],
              'dataset_clean_rows':audit['clean_rows'],'seed':SEED,'limitations':LIMITATIONS}
    manifest=save_model(model,metadata)
    summary={'run_id':run_id,'status':'computed; notebook validation pending','model_version':run_id,
        'dataset':dataset,'selected_feature_group':feature_group,'selected_parameters':search.best_params_,
        'cv_metrics':selected,'test_metrics':test_metrics,
        'historical_reference':{'label':'Previously reviewed measurements; not new independent data',
            'best_cv_f1':reference['best_cv_f1'],'test_metrics':reference['test_metrics']},
        'reproduction':{'matches_reported_precision':reproduced,'policy':reproduction_policy,'test_metric_differences':changes,
            'feature_cv_max_abs_differences':cv_changes,'best_cv_f1_difference':float(search.best_score_-reference['best_cv_f1']),
            'note':'Shared record-local engineering moved inside the saved pipeline; same split, feature ordering, seeds and search space.'},
        'dependencies':dependency_versions(),'python':platform.python_version(),'python_executable':sys.executable,
        'seed':SEED,'search_iterations':int(search.n_iter),'model_sha256':manifest['sha256'],
        'limitations':LIMITATIONS}
    comparison.to_csv(out/'feature_comparison.csv',index=False)
    fold_scores.to_csv(out/'feature_comparison_folds.csv',index=False)
    pd.DataFrame(search.cv_results_).to_csv(out/'random_search.csv',index=False)
    write_json(out/'summary.json',summary)
    return summary,manifest
