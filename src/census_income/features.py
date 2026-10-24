"""Deterministic transformations shared by research and saved inference."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


def engineer_features(frame):
    if not isinstance(frame,pd.DataFrame):
        raise TypeError('Features must be a pandas DataFrame')
    required = {'age','hours-per-week','education-num'}
    if required-set(frame):
        raise ValueError(f'Missing engineering inputs: {sorted(required-set(frame))}')
    result = frame.copy()
    for col in result.select_dtypes(include=['object','string']).columns:
        result[col] = result[col].str.strip().replace('?',np.nan)
    result['education_bin'] = pd.cut(result['education-num'],[0,9,13,16],
        labels=['HS or less','College to Bachelors','Postgraduate']).astype(object)
    result['hours_age_interaction'] = result['age']*result['hours-per-week']
    return result


class FeatureEngineer(TransformerMixin, BaseEstimator):
    def fit(self,X,y=None):
        engineer_features(X)
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = np.asarray(X.columns,dtype=object)
        return self

    def transform(self,X):
        return engineer_features(X)

    def get_feature_names_out(self,input_features=None):
        names = list(self.feature_names_in_ if input_features is None else input_features)
        return np.asarray(names+['education_bin','hours_age_interaction'],dtype=object)
