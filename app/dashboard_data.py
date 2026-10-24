"""Pure view calculations; never fit a model or modify scientific reports."""
from __future__ import annotations

import pandas as pd

EDUCATION_GROUPS = ['HS or less', 'College to Bachelors', 'Postgraduate']
INCOMES = ['<=50K', '>50K']


def cohort(frame: pd.DataFrame, age_range: tuple, sex: str, education: str, income: str) -> pd.DataFrame:
    mask = frame.age.between(*age_range)
    if sex != 'All':
        mask &= frame.sex.eq(sex)
    if education != 'All':
        groups = pd.cut(frame['education-num'], [0,9,13,16], labels=EDUCATION_GROUPS)
        mask &= groups.eq(education)
    if income != 'All':
        mask &= frame.income.eq(income)
    return frame.loc[mask].copy()


def feature_distribution(frame: pd.DataFrame, feature: str) -> pd.DataFrame:
    columns = ['Category','Income','Records','Denominator','Share']
    if frame.empty:
        return pd.DataFrame(columns=columns)
    if feature == 'age':
        groups = pd.cut(frame.age, [17,25,35,45,55,65,75,91], right=False,
                        labels=['17–24','25–34','35–44','45–54','55–64','65–74','75–90'])
    elif feature == 'hours-per-week':
        groups = pd.cut(frame[feature], [0,20,35,41,60,100], right=False,
                        labels=['Under 20','20–34','35–40','41–59','60–99'])
    else:
        groups = frame[feature].astype('string').fillna('Missing')
    data = pd.DataFrame({'Category': groups.astype('string'), 'Income': frame.income})
    counts = data.groupby(['Category','Income'], observed=True).size()
    categories = sorted(data.Category.unique())
    index = pd.MultiIndex.from_product([categories, INCOMES], names=['Category','Income'])
    result = counts.reindex(index,fill_value=0).rename('Records').reset_index()
    result['Denominator'] = result.groupby('Category').Records.transform('sum')
    result['Share'] = result.Records/result.Denominator
    return result[columns]


def interval_view(frame: pd.DataFrame, rate: str) -> pd.DataFrame:
    result = frame.copy()
    result['Rate'] = result[rate]
    if rate == 'fnr':
        result['Low'] = 1-result.recall_tpr_ci_high
        result['High'] = 1-result.recall_tpr_ci_low
        result['Denominator'] = result.positives
    else:
        result['Low'], result['High'] = result.fpr_ci_low, result.fpr_ci_high
        result['Denominator'] = result.negatives
    result['Upper'] = (result.High-result.Rate).clip(lower=0)
    result['Lower'] = (result.Rate-result.Low).clip(lower=0)
    return result


def selected_income(event: dict) -> str:
    points = event.get('selection', {}).get('points', [])
    if len(points) == 1:
        values = points[0].get('customdata', [])
        if values and values[0] in INCOMES:
            return values[0]
    return 'All'
