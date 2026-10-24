"""Calibration report arithmetic on held-out predictions from actual UCI rows."""
import numpy as np
import pytest
from sklearn.metrics import brier_score_loss,log_loss
from census_income.evaluate import calibration_diagnostics


def test_calibration_bin_counts_and_losses_match_actual_predictions(fitted_pipeline,small_training):
    X,y=small_training
    score=fitted_pipeline.predict_proba(X)[:,1]
    bins,metrics=calibration_diagnostics(y,score)
    assert bins['count'].sum()==len(y)
    assert metrics['brier_score']==pytest.approx(brier_score_loss(y,score))
    assert metrics['log_loss']==pytest.approx(log_loss(y,score,labels=[0,1]))
    assert np.all((bins.observed_fraction>=0)&(bins.observed_fraction<=1))
