"""UCI acquisition, schema checks, deterministic cleaning and benchmark splitting."""
from pathlib import Path
import hashlib
import urllib.request
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from .config import COLUMNS, NUMERIC_COLUMNS, RAW_DIR, DATA_URL, SOURCE_HASHES, SEED


def validate_schema(frame):
    missing = set(COLUMNS) - set(frame.columns)
    extra = set(frame.columns) - set(COLUMNS)
    if missing or extra:
        raise ValueError(f'Dataset schema mismatch; missing={sorted(missing)}, extra={sorted(extra)}')
    for col in NUMERIC_COLUMNS:
        if not pd.api.types.is_numeric_dtype(frame[col]):
            raise ValueError(f'{col} must be numeric')
        if frame[col].isna().any() or not np.isfinite(frame[col].to_numpy()).all():
            raise ValueError(f'{col} contains missing or non-finite source values')
    if frame['income'].isna().any() or not set(frame.income.unique()) <= {'<=50K','>50K'}:
        raise ValueError('Unknown or missing income labels')
    if (frame['education-num'] < 1).any() or (frame['education-num'] > 16).any():
        raise ValueError('education-num must be between 1 and 16')
    return frame


def clean_data(frame):
    if set(frame.columns) != set(COLUMNS):
        raise ValueError('Dataset schema does not match the UCI Adult columns')
    clean = frame.copy()
    for col in clean.select_dtypes(include=['object','string']).columns:
        clean[col] = clean[col].str.strip().replace('?', np.nan)
    clean['income'] = clean['income'].str.rstrip('.')
    validate_schema(clean)
    return clean.drop_duplicates().reset_index(drop=True)


def encode_target(values):
    labels = pd.Series(values).astype('string').str.strip().str.rstrip('.')
    if labels.isna().any() or not set(labels.unique()) <= {'<=50K','>50K'}:
        raise ValueError('Unknown or missing income labels')
    result = labels.map({'<=50K':0,'>50K':1}).astype(int)
    result.name = getattr(values, 'name', 'income')
    return result


def load_adult(data_dir=RAW_DIR, download=False):
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True,exist_ok=True)
    frames, provenance = [], {}
    for filename,skip in [('adult.data',0),('adult.test',1)]:
        path = data_dir/filename
        if not path.exists():
            if not download:
                raise FileNotFoundError(f'Missing {path}. Run training with --download to fetch the official UCI files.')
            urllib.request.urlretrieve(DATA_URL+filename,path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != SOURCE_HASHES[filename]:
            raise ValueError(f'Source checksum mismatch for {filename}; inspect the data before continuing')
        provenance[filename] = {'url':DATA_URL+filename,'sha256':digest,'bytes':path.stat().st_size}
        frames.append(pd.read_csv(path,names=COLUMNS,header=None,skiprows=skip))
    raw = pd.concat(frames,ignore_index=True)
    if raw.shape != (48842,15):
        raise ValueError(f'Unexpected UCI source shape: {raw.shape}')
    normalised = raw.copy()
    for col in normalised.select_dtypes(include=['object','string']).columns:
        normalised[col] = normalised[col].str.strip().replace('?',np.nan)
    normalised['income'] = normalised.income.str.rstrip('.')
    cleaned = clean_data(raw)
    audit = {'raw_rows':len(raw),'raw_columns':len(raw.columns),'clean_rows':len(cleaned),
             'duplicates_removed':len(raw)-len(cleaned),'source_year':1994,
             'missing_before_deduplication':{k:int(v) for k,v in normalised.isna().sum().items()},
             'dtypes':normalised.dtypes.astype(str).to_dict(),
             'categories':{c:sorted(normalised[c].dropna().unique().tolist()) for c in normalised.select_dtypes(include=['object','string']).columns},
             'target_distribution':{k:int(v) for k,v in cleaned.income.value_counts().items()},
             'provenance':provenance,
             'transformations':['strip categorical whitespace','replace ? with missing','remove income terminal period','remove exact full-record duplicates']}
    return cleaned,audit


def split_data(frame):
    validate_schema(frame)
    X = frame.drop(columns='income')
    y = encode_target(frame.income)
    return train_test_split(X,y,test_size=.2,stratify=y,random_state=SEED)


def split_independence(frame, X_dev, X_test):
    if set(X_dev.index) & set(X_test.index):
        raise ValueError('Development and test row indices overlap')
    full = pd.util.hash_pandas_object(frame[COLUMNS],index=False)
    if set(full.loc[X_dev.index]) & set(full.loc[X_test.index]):
        raise ValueError('Exact complete records cross the split')
    predictors = pd.util.hash_pandas_object(frame[COLUMNS[:-1]],index=False)
    overlap = set(predictors.loc[X_dev.index]) & set(predictors.loc[X_test.index])
    conflicting = sum(frame.loc[predictors.eq(key),'income'].nunique() > 1 for key in overlap)
    return {'index_overlap':0,'full_record_overlap':0,'predictor_profile_overlap':len(overlap),
            'conflicting_cross_split_profiles':int(conflicting),
            'interpretation':'Profile overlap is an independence/data-quality concern; person identity is unavailable.'}
