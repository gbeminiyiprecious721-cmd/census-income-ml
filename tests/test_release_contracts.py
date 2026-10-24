"""Release boundaries that must hold before writing a trusted model artifact."""
import pytest
from census_income.preprocessing import make_pipeline
from census_income.train import save_model


def test_model_with_protected_input_contract_cannot_overwrite_release(small_training,tmp_path):
    X,y=small_training
    model=make_pipeline('Original six',n_estimators=2,max_depth=2,n_jobs=1).fit(X,y)
    destination=tmp_path/'release'
    destination.mkdir()
    artifact=destination/'model.joblib';artifact.write_bytes(b'previous trusted release')
    with pytest.raises(ValueError,match='input contract'):
        save_model(model,{'run_id':'unsupported'},destination)
    assert artifact.read_bytes()==b'previous trusted release'
    assert not (destination/'model_manifest.json').exists()


def test_single_class_model_is_rejected_before_artifact_creation(small_training,tmp_path):
    X,y=small_training
    model=make_pipeline(n_estimators=2,max_depth=2,n_jobs=1).fit(X,y*0)
    destination=tmp_path/'invalid'
    with pytest.raises(ValueError,match='classes'):
        save_model(model,{},destination)
    assert not destination.exists()


def test_explicit_record_policy_retains_mismatch_evidence(fitted_pipeline,tmp_path,monkeypatch):
    import json
    from types import SimpleNamespace
    import pandas as pd
    import census_income.research as research
    historical=tmp_path/'historical';historical.mkdir()
    metrics={'accuracy':.8,'precision':.6,'recall':.7,'f1':.64,'roc_auc':.9}
    reference={'selected_features':'Expanded + interaction','best_parameters':{},
               'best_cv_f1':.64,'test_metrics':{**metrics,'accuracy':.79}}
    (historical/'results.json').write_text(json.dumps(reference))
    comparison=pd.DataFrame([{'model':'Expanded + interaction',**metrics}])
    comparison.to_csv(historical/'feature_comparison.csv',index=False)
    monkeypatch.setattr(research,'ARTIFACTS_DIR',tmp_path)
    monkeypatch.setattr(research,'save_model',lambda *args:{'sha256':'test-artifact'})
    search=SimpleNamespace(best_score_=.64,best_params_={},best_index_=0,n_iter=1,
                           cv_results_={**{'mean_test_'+k:[v] for k,v in metrics.items()}})
    out=tmp_path/'reports'
    summary,_=research.finalize_research(fitted_pipeline,'Expanded + interaction',search,
        comparison,pd.DataFrame(),metrics,{'clean_rows':10,'provenance':{}},
        {'development_rows':8,'test_rows':2},report_dir=out,reproduction_policy='record')
    assert not summary['reproduction']['matches_reported_precision']
    assert summary['reproduction']['test_metric_differences']['accuracy']==pytest.approx(.01)
    assert summary['reproduction']['policy']=='record'
    assert (out/'reproduction_difference.json').exists()
    with pytest.raises(AssertionError,match='historical'):
        research.finalize_research(fitted_pipeline,'Expanded + interaction',search,
            comparison,pd.DataFrame(),metrics,{'clean_rows':10,'provenance':{}},
            {'development_rows':8,'test_rows':2},report_dir=tmp_path/'strict')


def test_supported_pipeline_export_roundtrips_predictions(fitted_pipeline,small_training,tmp_path):
    import joblib
    import numpy as np
    from census_income.config import MODEL_INPUTS
    X,_=small_training
    manifest=save_model(fitted_pipeline,{'run_id':'test'},tmp_path)
    restored=joblib.load(tmp_path/'model.joblib')  # only the artifact created in this test
    assert manifest['model_inputs']==MODEL_INPUTS
    np.testing.assert_allclose(restored.predict_proba(X[MODEL_INPUTS]),
                               fitted_pipeline.predict_proba(X),atol=1e-12)
