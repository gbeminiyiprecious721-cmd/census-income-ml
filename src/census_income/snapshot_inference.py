"""Inference from the exact frozen model paired with displayed research reports."""
from functools import lru_cache
import hashlib
import importlib.metadata as metadata
import io
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from . import __version__
from .config import MODEL_INPUTS
from .inference import _validated_records
from .run_store import BundleError, DEPENDENCIES, Snapshot


@lru_cache(maxsize=2)
def _model(payload: bytes, manifest_bytes: bytes) -> Pipeline:
    manifest = json.loads(manifest_bytes)
    if (manifest.get('artifact_format') != 'census-income-pipeline-v1'
            or manifest.get('package_version') != __version__
            or manifest.get('model_inputs') != MODEL_INPUTS
            or manifest.get('sha256') != hashlib.sha256(payload).hexdigest()):
        raise BundleError('Snapshot model format, schema or integrity mismatch')
    for dependency in DEPENDENCIES:
        if manifest.get('dependencies', {}).get(dependency) != metadata.version(dependency):
            raise BundleError(f'Model dependency version mismatch: {dependency}')
    model = joblib.load(io.BytesIO(payload))
    if (not isinstance(model, Pipeline)
            or list(model.named_steps) != ['engineer', 'preprocess', 'forest']
            or not np.array_equal(model.classes_, [0, 1])):
        raise BundleError('Unexpected snapshot model pipeline or target classes')
    return model


def predict_snapshot_records(snapshot: Snapshot, records) -> pd.DataFrame:
    frame, warnings = _validated_records(records)
    model = _model(snapshot.bytes('artifacts/model.joblib'), snapshot.bytes('artifacts/model_manifest.json'))
    pre = model['preprocess']
    cat_cols = list(pre.transformers_[1][2])
    encoder = pre.named_transformers_['categorical']['encode']
    known = {c: set(v) for c, v in zip(cat_cols, encoder.categories_) if c in frame}
    for column, values in known.items():
        for i, value in enumerate(frame[column].tolist()):
            if pd.notna(value) and value not in values:
                warnings[i].append(f'{column}: unseen category encoded without a fitted indicator')
    score = model.predict_proba(frame)[:, 1]
    prediction = model.predict(frame)
    return pd.DataFrame({'predicted_class': prediction,
                         'income_category': np.where(prediction == 1, '>50K', '<=50K'),
                         'model_score': score,
                         'input_warnings': ['; '.join(row) for row in warnings]}, index=frame.index)
