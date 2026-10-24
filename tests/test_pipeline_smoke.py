"""A real-data integration contract for the refactored pipeline."""
import numpy as np

from census_income.data import load_adult, split_data
from census_income.preprocessing import make_pipeline
from census_income.evaluate import metric_summary


def test_real_records_pass_through_engineering_preprocessing_and_classifier():
    frame, audit = load_adult()
    dev, test, y_dev, y_test = split_data(frame)
    model = make_pipeline('Expanded + interaction', n_estimators=8, max_depth=4)
    model.fit(dev.iloc[:700], y_dev.iloc[:700])
    scores = model.predict_proba(test.iloc[:30])[:, 1]
    pred = model.predict(test.iloc[:30])
    metrics = metric_summary(y_test.iloc[:30], pred, scores)
    assert np.isclose(metrics['accuracy'], np.mean(pred == y_test.iloc[:30].to_numpy()))
    assert len(scores) == 30 and np.isfinite(scores).all()
    assert audit['raw_rows'] == 48842
    assert set(dev.index).isdisjoint(test.index)
