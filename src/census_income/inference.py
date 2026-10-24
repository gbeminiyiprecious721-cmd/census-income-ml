"""Validated inference only. Never imports training or accepts model uploads."""
import hashlib
import importlib.metadata as metadata
import json
import numbers
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from . import __version__
from .config import MODEL_PATH, MANIFEST_PATH, MODEL_INPUTS, NUMERIC_RANGES

_CACHE = {}


def load_validated_model():
    """Load only the maintainer-generated fixed artifact after integrity checks.

    The manifest is local provenance, not a cryptographic signature. A model and
    manifest obtained from an untrusted party must never be placed here.
    """
    if not MODEL_PATH.is_file() or not MANIFEST_PATH.is_file():
        raise FileNotFoundError('Validated model artifact/manifest missing. Run the research notebook or the documented training command.')
    manifest=json.loads(MANIFEST_PATH.read_text(encoding='utf-8'))
    if manifest.get('artifact_format')!='census-income-pipeline-v1':
        raise ValueError('Unsupported model artifact format')
    if manifest.get('package_version')!=__version__:
        raise ValueError('Model package version is incompatible; reproduce the fitted artifact')
    for dependency in ['scikit-learn','numpy','pandas','scipy','joblib']:
        if manifest.get('dependencies',{}).get(dependency)!=metadata.version(dependency):
            raise ValueError(f'Model dependency version mismatch: {dependency}')
    digest=hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()
    if digest!=manifest.get('sha256'):
        raise ValueError('Model checksum/integrity mismatch; regenerate the trusted artifact')
    if manifest.get('model_inputs')!=MODEL_INPUTS:
        raise ValueError('Model input schema mismatch')
    key=(str(MODEL_PATH.resolve()),digest)
    if key not in _CACHE:
        model=joblib.load(MODEL_PATH)
        if not isinstance(model,Pipeline) or list(model.named_steps)!=['engineer','preprocess','forest']:
            raise ValueError('Unexpected model pipeline structure')
        if not np.array_equal(model.classes_,[0,1]):
            raise ValueError('Unexpected model target classes')
        _CACHE.clear(); _CACHE[key]=model
    return _CACHE[key]


def _validated_records(records):
    if isinstance(records,list): records=pd.DataFrame(records)
    if not isinstance(records,pd.DataFrame): raise TypeError('Provide a DataFrame or list of record dictionaries')
    if records.empty: raise ValueError('Provide at least one observation')
    if records.columns.duplicated().any(): raise ValueError('Duplicate input columns are not supported')
    missing=set(MODEL_INPUTS)-set(records)
    if missing: raise ValueError(f'Missing required input columns: {sorted(missing)}')
    result=records[MODEL_INPUTS].copy()
    warnings=[[] for _ in range(len(result))]
    extras=set(records)-set(MODEL_INPUTS)
    if extras:
        for row in warnings: row.append('Extra columns ignored: '+', '.join(sorted(extras)))
    for column in MODEL_INPUTS:
        for position,value in enumerate(result[column].tolist()):
            if value is None or value is pd.NA or (isinstance(value,numbers.Real) and np.isnan(value)):
                result.iloc[position,result.columns.get_loc(column)]=np.nan
                warnings[position].append(f'{column}: missing value imputed by fitted preprocessing')
                continue
            if column in NUMERIC_RANGES:
                if isinstance(value,(bool,np.bool_)) or not isinstance(value,numbers.Real) or not np.isfinite(value):
                    raise ValueError(f'{column} must be a finite numeric value or missing')
                lower,upper=NUMERIC_RANGES[column]
                if not lower<=value<=upper or not float(value).is_integer():
                    raise ValueError(f'{column} must be an integer in the historical range [{lower}, {upper}]')
            else:
                if not isinstance(value,str): raise ValueError(f'{column} must be a categorical string or missing')
                cleaned=value.strip()
                if cleaned in ('','?'):
                    result.iloc[position,result.columns.get_loc(column)]=np.nan
                    warnings[position].append(f'{column}: missing value imputed by fitted preprocessing')
                else: result.iloc[position,result.columns.get_loc(column)]=cleaned
    return result,warnings


def predict_records(records):
    frame,warnings=_validated_records(records)
    model=load_validated_model()
    pre=model['preprocess']; cat_cols=list(pre.transformers_[1][2])
    encoder=pre.named_transformers_['categorical']['encode']
    known={c:set(v) for c,v in zip(cat_cols,encoder.categories_) if c in frame}
    for c,values in known.items():
        for i,value in enumerate(frame[c].tolist()):
            if pd.notna(value) and value not in values:
                warnings[i].append(f'{c}: unseen category encoded without a fitted indicator')
    score=model.predict_proba(frame)[:,1]
    prediction=model.predict(frame)
    return pd.DataFrame({'predicted_class':prediction,'income_category':np.where(prediction==1,'>50K','<=50K'),
        'model_score':score,'input_warnings':['; '.join(row) for row in warnings]},index=frame.index)
