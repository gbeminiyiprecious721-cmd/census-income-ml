"""Independent small fixtures catch filter and denominator mistakes."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'app'))
from dashboard_data import cohort, feature_distribution, interval_view, selected_income


@pytest.fixture
def sample():
    return pd.DataFrame({'age':[20,30,40,50], 'sex':['Female','Male','Female','Male'],
        'education-num':[8,10,13,16], 'occupation':['A','A','B','B'],
        'income':['<=50K','>50K','<=50K','<=50K']})


def test_filters_intersect_and_do_not_mutate_source(sample):
    result = cohort(sample, (25,45), 'Female', 'College to Bachelors', 'All')
    assert result.index.tolist() == [2]
    assert len(sample) == 4 and 'education_group' not in sample
    assert cohort(sample,(17,90),'All','All','>50K').index.tolist() == [1]


def test_empty_cohort_has_safe_aggregate_schema(sample):
    result = cohort(sample, (80,90), 'All', 'All', 'All')
    assert result.empty
    assert feature_distribution(result,'occupation').empty


def test_distribution_uses_within_group_denominators(sample):
    data = feature_distribution(sample,'occupation')
    actual = {(r.Category,r.Income):(r.Records,r.Denominator,r.Share) for r in data.itertuples()}
    assert actual[('A','>50K')] == (1,2,.5)
    assert actual[('B','>50K')] == (0,2,0)
    assert actual[('B','<=50K')] == (2,2,1)
    assert data.Records.sum() == 4


def test_false_negative_interval_complements_and_reverses_endpoints():
    data=pd.DataFrame({'group':['A'],'positives':[20],'negatives':[80],
       'fnr':[.25],'fpr':[.1],'recall_tpr_ci_low':[.5],'recall_tpr_ci_high':[.9],
       'fpr_ci_low':[.05],'fpr_ci_high':[.2]})
    row=interval_view(data,'fnr').iloc[0]
    assert row.Low == pytest.approx(.1) and row.High == .5
    assert row.Denominator == 20 and row.Rate == .25
    assert interval_view(data,'fpr').iloc[0].Denominator == 80


def test_chart_selection_accepts_only_known_income_categories():
    assert selected_income({'selection':{'points':[{'customdata':['>50K']} ]}}) == '>50K'
    assert selected_income({'selection':{'points':[]}}) == 'All'
    assert selected_income({'selection':{'points':[{'customdata':['unknown']} ]}}) == 'All'
