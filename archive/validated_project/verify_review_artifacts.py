"""Independent arithmetic and source-profile checks; no model fitting or selection."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

root = Path(__file__).resolve().parent
out = root/'artifacts'
r = json.loads((out/'scientific_review.json').read_text())
cols = ['age','workclass','fnlwgt','education','education-num','marital-status','occupation',
        'relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country','income']
d = pd.concat([pd.read_csv(root/'data/adult.data',names=cols,header=None),
    pd.read_csv(root/'data/adult.test',names=cols,header=None,skiprows=1)],ignore_index=True)
for c in d.select_dtypes(include=['object','string']): d[c]=d[c].str.strip().replace('?',np.nan)
d['income']=d.income.str.rstrip('.')
d=d.drop_duplicates().reset_index(drop=True)
dev,test = train_test_split(d.index,test_size=.2,stratify=d.income.eq('>50K'),random_state=42)
fingerprint = pd.util.hash_pandas_object(d[cols[:-1]],index=False)
overlap = set(fingerprint.loc[dev]) & set(fingerprint.loc[test])
conflicting = sum(d.loc[fingerprint.eq(profile),'income'].nunique()==2 for profile in overlap)
assert len(overlap)==r['shared_full_predictor_profiles']
assert conflicting == len(overlap)
t=pd.read_csv(out/'review_threshold_diagnostics.csv')
g=pd.read_csv(out/'review_subgroup_metrics.csv')
at_half=t.loc[t.threshold.eq(.5)].iloc[0]
for table in [t,g]:
    assert np.allclose(table.accuracy,(table.tn+table.tp)/(table.tn+table.fp+table.fn+table.tp))
    assert np.allclose(table.precision,table.tp/(table.tp+table.fp))
    assert np.allclose(table.f1,2*table.tp/(2*table.tp+table.fp+table.fn))
assert np.allclose(t.recall,t.tp/(t.tp+t.fn))
assert np.allclose(g.recall_tpr,g.tp/g.positives)
assert np.allclose(g.fpr,g.fp/g.negatives)
for attribute in ['sex','race']:
    s=g[g.attribute.eq(attribute)]
    assert s.n.sum()==r['heldout_rows']
    assert np.array_equal(s[['tn','fp','fn','tp']].sum().to_numpy(),at_half[['tn','fp','fn','tp']].to_numpy())
report={'complete_predictor_profiles_crossing_split':len(overlap),
    'crossing_profiles_with_conflicting_income_labels':int(conflicting),
    'threshold_and_subgroup_metric_arithmetic_verified':True,
    'subgroup_confusion_totals_reconciled':True}
(out/'review_independent_checks.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
