"""Fold-local learning; the saved pipeline also owns deterministic engineering."""
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from .config import FEATURE_SETS, NUMERIC_FEATURES, SEED, JOBS
from .features import FeatureEngineer


def make_pipeline(feature_group='Expanded + interaction', **forest_params):
    if feature_group not in FEATURE_SETS:
        raise ValueError(f'Unknown feature group: {feature_group}')
    cols = FEATURE_SETS[feature_group]
    numerical = [c for c in cols if c in NUMERIC_FEATURES]
    categorical = [c for c in cols if c not in numerical]
    preprocessing = ColumnTransformer([
        ('numeric',SimpleImputer(strategy='median'),numerical),
        ('categorical',Pipeline([('impute',SimpleImputer(strategy='most_frequent')),
            ('encode',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]),categorical)])
    params = {'random_state':SEED,'n_jobs':JOBS,**forest_params}
    return Pipeline([('engineer',FeatureEngineer()),('preprocess',preprocessing),
                     ('forest',RandomForestClassifier(**params))])
