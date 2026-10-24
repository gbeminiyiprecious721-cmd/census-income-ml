from pathlib import Path
import shutil
import nbformat as nbf

path = Path('Adult_Income_Random_Forest_End_to_End.ipynb')
backup = path.with_suffix('.original.ipynb')
if not backup.exists():
    shutil.copy2(path, backup)
n = nbf.v4.new_notebook()
cells = []
def md(s): cells.append(nbf.v4.new_markdown_cell(s))
def code(s): cells.append(nbf.v4.new_code_cell(s))
md('''# Adult Income: Random Forest, end to end
## Goal and methodology
Predict income above $50,000 using the actual UCI Adult Census dataset. This portfolio tutorial covers the 17 requested exercises and a separate five-fold model-selection workflow. Results below are computed, never placeholders.

**Source:** Becker & Kohavi (1996), [UCI Adult](https://archive.ics.uci.edu/dataset/2/adult), DOI: 10.24432/C5XW20, CC BY 4.0. Records originate in the 1994 Census database.

### Key assumptions
We pool the two original source files, remove exact duplicate records before splitting, and use a reproducible stratified 80/20 development/test split. The original six features are those in the supplied notebook. Tutorial depth selection uses a validation subset of development data, never the final test set. Five-fold comparisons and random search use development data only. Accuracy is unweighted per record; `fnlwgt` is neither a predictor nor a sample weight. This is a historical benchmark, not a deployable income decision system.''')
md('''## Setup
Use the project `.venv312` interpreter (Python 3.12) and install `requirements.txt`. This cell records the executable and package versions actually used. The original notebook specified generic `python3`; a running IDE kernel cannot be inferred from that metadata. Downloads are cached under `data/`, with SHA-256 provenance recorded.''')
code('''from pathlib import Path
import sys, os, json, hashlib, platform, importlib.metadata as metadata
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy, ucimlrepo, ipykernel
from IPython.display import display, Markdown
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay)
SEED = 42
JOBS = min(4, os.cpu_count() or 1)
N_ITER = 12  # bounded search: 60 fits; increase for a more exhaustive study
np.random.seed(SEED)
OUT = Path('artifacts'); OUT.mkdir(exist_ok=True)
versions = {p: metadata.version(p) for p in ['pandas','numpy','matplotlib','seaborn','scikit-learn','ucimlrepo','scipy','ipykernel','nbclient','nbformat']}
print('Python:', sys.executable, platform.python_version(), 'forest threads:', JOBS)
display(pd.Series(versions, name='version'))
sns.set_theme(style='whitegrid', palette=['#245782','#C17C28'])
plt.rcParams.update({'figure.figsize': (9, 5), 'figure.dpi': 110})
def savefig(name):
    plt.tight_layout(); plt.savefig(OUT / (name + '.png'), bbox_inches='tight'); plt.show()
''')
md('## Data verification and cleaning — tasks 1–4\nWhitespace and terminal target punctuation are deterministic record-level cleaning. Missing `?` values become nulls. We retain missing predictors for fold-local imputation. Exact full-record duplicates are removed conservatively; legitimate identical individuals cannot be distinguished without identifiers.')
code('''import urllib.request
columns = ['age','workclass','fnlwgt','education','education-num','marital-status','occupation',
           'relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country','income']
DATA = Path('data'); DATA.mkdir(exist_ok=True)
base = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/'
frames, provenance = [], {}
for filename, skip in [('adult.data',0), ('adult.test',1)]:
    local = DATA / filename
    if not local.exists():
        urllib.request.urlretrieve(base + filename, local)
    provenance[filename] = {'url': base + filename, 'sha256': hashlib.sha256(local.read_bytes()).hexdigest()}
    frames.append(pd.read_csv(local, names=columns, header=None, skiprows=skip))
raw = pd.concat(frames, ignore_index=True)
assert raw.shape == (48842, 15), raw.shape
print('Raw source shape:', raw.shape)
df = raw.copy()
for col in df.select_dtypes(include=['object','string']).columns:
    df[col] = df[col].str.strip().replace('?', np.nan)
df['income'] = df['income'].str.rstrip('.')
assert set(df.income.unique()) == {'<=50K','>50K'}
assert df.income.notna().all()
duplicate_count = int(df.duplicated().sum())
print('Exact duplicate records removed:', duplicate_count)
display(pd.DataFrame({'dtype': df.dtypes.astype(str), 'missing': df.isna().sum(), 'unique': df.nunique()}))
display(pd.DataFrame({c: pd.Series(sorted(df[c].dropna().unique())) for c in df.select_dtypes(include=['object','string']).columns}))
df = df.drop_duplicates().reset_index(drop=True)
distribution = df.income.value_counts().rename_axis('income').to_frame('count')
distribution['percent'] = 100 * distribution['count'] / len(df)
display(distribution)
distribution['count'].plot.bar(rot=0, color='#245782')
plt.title('Income distribution after exact deduplication'); plt.ylabel('Records'); plt.xlabel('Annual income class')
savefig('income_distribution')
(OUT / 'provenance.json').write_text(json.dumps(provenance, indent=2))
''')
md('''## Leakage-safe split and features — task 5
All learned imputation and encoding live inside pipelines. Education bins use fixed cut points: 1–9 (high school or less), 10–13 (some college through bachelors), 14–16 (postgraduate). Expanded features retain the education group and add workclass, occupation, marital status, relationship and country. The age × weekly-hours interaction is a hypothesis, not measured annual work. Expanded features omit sex/race, but proxies remain; this is not a fairness guarantee. An expanded ablation without the interaction tests its value. Education number/text are omitted to avoid redundant encodings.''')
code('''df['education_bin'] = pd.cut(df['education-num'], [0,9,13,16], labels=['HS or less','College to Bachelors','Postgraduate']).astype(object)
assert df.education_bin.notna().all()
df['hours_age_interaction'] = df['age'] * df['hours-per-week']
y = df.income.eq('>50K').astype(int)
original = ['age','capital-gain','capital-loss','hours-per-week','sex','race']
education = original + ['education_bin']
expanded = ['age','capital-gain','capital-loss','hours-per-week','education_bin','workclass',
            'occupation','marital-status','relationship','native-country']
feature_sets = {'Original six': original, 'Plus education': education,
                'Expanded census': expanded, 'Expanded + interaction': expanded + ['hours_age_interaction']}
all_cols = sorted(set(sum(feature_sets.values(), [])))
X = df[all_cols]
X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=.2, stratify=y, random_state=SEED)
X_train, X_val, y_train, y_val = train_test_split(X_dev, y_dev, test_size=.25, stratify=y_dev, random_state=SEED)
assert set(X_dev.index).isdisjoint(X_test.index)
assert set(X_train.index).isdisjoint(X_val.index)
print('Training, validation, final test:', len(X_train), len(X_val), len(X_test))
print('Positive proportions:', y_train.mean(), y_val.mean(), y_test.mean())
def make_pipeline(cols, **params):
    categorical = [c for c in cols if not pd.api.types.is_numeric_dtype(X[c])]
    numerical = [c for c in cols if c not in categorical]
    pre = ColumnTransformer([
        ('numeric', SimpleImputer(strategy='median'), numerical),
        ('categorical', Pipeline([('impute',SimpleImputer(strategy='most_frequent')),
            ('encode',OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), categorical)])
    return Pipeline([('preprocess',pre), ('forest',RandomForestClassifier(random_state=SEED,n_jobs=JOBS,**params))])
def importance(model):
    return pd.DataFrame({'feature': model['preprocess'].get_feature_names_out(),
                         'importance': model['forest'].feature_importances_}).sort_values('importance',ascending=False)
''')
md('''## Tutorial experiments — tasks 6–17
Fit the default 100-tree baseline, tune depths 1–25 on validation accuracy, then repeat with education categories. Ties choose the shallowest depth. Training curves diagnose fit; high training accuracy alone is insufficient. This preserves the supplied exercises while replacing their repeated test-set selection with validation. Results are not claimed to reproduce a separately supplied Codecademy answer key.''')
code('''baseline = make_pipeline(original)
baseline.fit(X_train, y_train)
baseline_accuracy = accuracy_score(y_val, baseline.predict(X_val))
print('Default Random Forest validation accuracy:', baseline_accuracy)
print('Majority-class validation accuracy:', accuracy_score(y_val, np.repeat(y_train.mode()[0],len(y_val))))
def depth_experiment(cols, title):
    rows = []
    for depth in range(1,26):
        model = make_pipeline(cols,max_depth=depth)
        model.fit(X_train,y_train)
        rows.append({'depth':depth,'train_accuracy':accuracy_score(y_train,model.predict(X_train)),
                     'validation_accuracy':accuracy_score(y_val,model.predict(X_val))})
    table = pd.DataFrame(rows)
    best = table.loc[table.validation_accuracy.idxmax()]
    selected = make_pipeline(cols,max_depth=int(best.depth)).fit(X_train,y_train)
    ax = table.plot(x='depth',y=['train_accuracy','validation_accuracy'],style=['-','--'])
    ax.set(xlabel='Maximum tree depth',ylabel='Accuracy',title=title)
    savefig(title.replace(' ','_'))
    print(title, best.to_dict())
    return table, best, selected
original_curve, original_best, original_model = depth_experiment(original,'Original features depth tuning')
education_curve, education_best, education_model = depth_experiment(education,'Education features depth tuning')
imp_comparison = pd.concat([importance(original_model).assign(model='Original'),importance(education_model).assign(model='Education')])
fig, axes = plt.subplots(1,2,figsize=(15,5))
for ax, model, title in zip(axes,[original_model,education_model],['Original','With education']):
    importance(model).head(10).sort_values('importance').plot.barh(x='feature',y='importance',ax=ax,legend=False)
    ax.set(title=title,xlabel='Impurity importance',ylabel='Encoded feature')
savefig('tutorial_importance_comparison')
display(pd.DataFrame({'model':['Default original','Depth-tuned original','Depth-tuned education'],
    'validation_accuracy':[baseline_accuracy,original_best.validation_accuracy,education_best.validation_accuracy]}))
display(Markdown(f'Education changes best validation accuracy by **{100*(education_best.validation_accuracy-original_best.validation_accuracy):+.2f} percentage points**. This comparison includes depth selection and is exploratory.'))
''')
md('''## Five-fold feature comparison
The same shuffled stratified folds and fixed 100-tree, depth-16 forest compare all candidate groups. Report all five metrics and F1 dispersion; fold standard deviations are descriptive, not confidence intervals. Select the feature group by mean positive-class F1. These folds remain internal development evidence; selection can make their scores optimistic.''')
code('''cv = StratifiedKFold(n_splits=5,shuffle=True,random_state=SEED)
scoring = {'accuracy':'accuracy','precision':'precision','recall':'recall','f1':'f1','roc_auc':'roc_auc'}
rows = []
fold_rows = []
for name, cols in feature_sets.items():
    print('Cross-validating',name,flush=True)
    result = cross_validate(make_pipeline(cols,n_estimators=100,max_depth=16),X_dev,y_dev,cv=cv,
        scoring=scoring,n_jobs=1,return_train_score=True,error_score='raise')
    rows.append({'model':name,**{m:result['test_'+m].mean() for m in scoring},
                 'f1_std':result['test_f1'].std(ddof=1),'train_f1':result['train_f1'].mean()})
    fold_rows.extend([{'model':name,'fold':i+1,**{m:result['test_'+m][i] for m in scoring}} for i in range(5)])
comparison = pd.DataFrame(rows).sort_values('f1',ascending=False)
display(comparison.round(4))
winning_name = comparison.iloc[0]['model']; winning_cols = feature_sets[winning_name]
comparison.to_csv(OUT/'feature_comparison.csv',index=False)
pd.DataFrame(fold_rows).to_csv(OUT/'feature_comparison_folds.csv',index=False)
comparison.set_index('model')[['train_f1','f1']].plot.bar(rot=15)
plt.ylabel('Positive-class F1'); plt.xlabel('Feature group'); plt.title('Five-fold training versus validation F1')
savefig('cv_feature_comparison')
print('Selected feature group:',winning_name)
''')
md('''## Hyperparameter optimisation
RandomizedSearchCV evaluates 12 reproducible combinations × five folds, using positive-class F1 for refitting while recording accuracy, precision, recall and ROC-AUC. Forests use at most four threads; search runs sequentially to avoid nested CPU oversubscription. The 12-trial budget is an explicit computational compromise and is not a claim of global optimality. The final threshold stays at 0.5.''')
code('''search = RandomizedSearchCV(make_pipeline(winning_cols),param_distributions={
    'forest__n_estimators':[100,200,300], 'forest__max_depth':[8,12,16,20,None],
    'forest__max_features':['sqrt',.5,1.0], 'forest__min_samples_split':[2,5,10],
    'forest__min_samples_leaf':[1,2,4,8], 'forest__class_weight':[None,'balanced','balanced_subsample']},
    n_iter=N_ITER,scoring=scoring,refit='f1',cv=cv,random_state=SEED,n_jobs=1,
    verbose=1,return_train_score=True,error_score='raise')
search.fit(X_dev,y_dev)
best_model = search.best_estimator_
search_results = pd.DataFrame(search.cv_results_)
search_results.to_csv(OUT/'random_search.csv',index=False)
print('Best CV F1:',search.best_score_); print('Best parameters:',search.best_params_)
display(search_results.sort_values('rank_test_f1')[['params']+['mean_test_'+m for m in scoring]].head(5))
plot = search_results.sort_values('mean_test_f1').reset_index(drop=True)
plt.plot(plot.index+1,plot.mean_train_f1,'o-',label='Training F1')
plt.plot(plot.index+1,plot.mean_test_f1,'s--',label='Validation F1')
plt.xlabel('Candidate ranked by validation F1'); plt.ylabel('Positive-class F1'); plt.title('Search training versus validation performance'); plt.legend()
savefig('search_train_validation')
''')
md('## Final held-out evaluation\nThis is the only test-set model evaluation. No test score is used to choose features, parameters or thresholds. Training/validation scores above and final test scores below have different roles and denominators.')
code('''final_pred = best_model.predict(X_test)
final_probability = best_model.predict_proba(X_test)[:,1]
final_metrics = {'accuracy':accuracy_score(y_test,final_pred),
    'precision':precision_score(y_test,final_pred,zero_division=0),
    'recall':recall_score(y_test,final_pred,zero_division=0),
    'f1':f1_score(y_test,final_pred,zero_division=0),'roc_auc':roc_auc_score(y_test,final_probability)}
display(pd.Series(final_metrics,name='Held-out test'))
print(classification_report(y_test,final_pred,target_names=['Income <=$50K','Income >$50K'],zero_division=0))
fig, axes = plt.subplots(1,3,figsize=(18,5))
ConfusionMatrixDisplay.from_predictions(y_test,final_pred,display_labels=['<=50K','>50K'],ax=axes[0],colorbar=False,cmap='Blues')
axes[0].set_title('Held-out confusion matrix')
RocCurveDisplay.from_predictions(y_test,final_probability,ax=axes[1],plot_chance_level=True)
axes[1].set_title('Held-out ROC curve')
PrecisionRecallDisplay.from_predictions(y_test,final_probability,ax=axes[2])
axes[2].axhline(y_test.mean(),linestyle=':',color='grey',label='Positive prevalence'); axes[2].legend()
axes[2].set_title('Held-out precision–recall curve')
savefig('final_evaluation')
importance_df = importance(best_model)
display(importance_df.head(10))
importance_df.head(10).sort_values('importance').plot.barh(x='feature',y='importance',legend=False,figsize=(10,6))
plt.xlabel('Mean decrease in impurity'); plt.ylabel('Encoded feature'); plt.title('Final forest: top 10 feature importances')
savefig('final_importance')
importance_df.to_csv(OUT/'feature_importance.csv',index=False)
''')
md('''## Technical validation
Assertions verify split integrity, raw-record separation, metric arithmetic, target mapping and prediction on missing values and unseen categories. The unseen-observation smoke test exercises the fitted preprocessing without fitting anything on the test set.''')
code('''assert y.equals(df.income.map({'<=50K':0,'>50K':1}))
assert len(set(X_dev.index) & set(X_test.index)) == 0
record_hash = pd.util.hash_pandas_object(df[columns],index=False)
assert set(record_hash.loc[X_dev.index]).isdisjoint(record_hash.loc[X_test.index])
tn, fp, fn, tp = confusion_matrix(y_test,final_pred).ravel()
assert np.isclose(final_metrics['accuracy'],(tn+tp)/(tn+fp+fn+tp))
assert np.isclose(final_metrics['f1'],2*tp/(2*tp+fp+fn))
assert np.isclose(final_metrics['precision'],tp/(tp+fp))
assert np.isclose(final_metrics['recall'],tp/(tp+fn))
unseen = X_dev.iloc[:2].copy()
for c in winning_cols:
    if pd.api.types.is_numeric_dtype(unseen[c]): unseen.loc[unseen.index[0],c] = np.nan
    else:
        unseen[c] = unseen[c].astype(object)
        unseen.loc[unseen.index[0],c] = 'UNSEEN_CATEGORY'
        unseen.loc[unseen.index[1],c] = np.nan
assert np.isfinite(best_model.predict_proba(unseen)).all()
assert all(0 <= v <= 1 for v in final_metrics.values())
print('All split, target, metric and unseen-observation checks passed.')
results = {'baseline_validation_accuracy':baseline_accuracy,
    'best_original_validation_accuracy':float(original_best.validation_accuracy),
    'best_education_validation_accuracy':float(education_best.validation_accuracy),
    'original_best_depth':int(original_best.depth),'education_best_depth':int(education_best.depth),
    'best_cv_f1':search.best_score_,'selected_features':winning_name,'best_parameters':search.best_params_,
    'test_metrics':final_metrics,'top_five_features':importance_df.head(5).to_dict('records'),
    'raw_shape':list(raw.shape),'clean_shape':list(df.shape),'duplicates_removed':duplicate_count,
    'split_sizes':{'development':len(X_dev),'test':len(X_test)},'versions':versions,
    'python':sys.executable,'seed':SEED,'search_iterations':N_ITER}
(OUT/'results.json').write_text(json.dumps(results,indent=2))
''')
md('''## Findings, limitations and recommendations
Impurity feature importances describe this fitted model; they do not establish causation. Continuous and high-cardinality features can receive inflated importance, while correlated predictors split importance. Education and marital/relationship fields may encode structural inequality. Sex and race appear in tutorial models; omitting them in expanded models does not remove demographic proxies. A subgroup error audit, uncertainty estimates and a defensible use case are necessary before any consequential use.

The majority class makes accuracy alone misleading; positive-class F1, precision, recall, ROC and precision–recall provide complementary evidence. Shallow-depth curves reveal underfitting and training–validation gaps reveal overfitting. Fold-local preprocessing prevents learned-transform leakage. Deduplication prevents exact full records crossing partitions, but no person identifier exists to rule out all related observations. The pooled random split differs from the original UCI split and contemporary deployment distributions. No claim of fairness, causality, calibrated probabilities or temporal generalisation is made. Cross-validation selection is not nested and has no unbiased selection-performance guarantee; the untouched test is the final estimate. Follow-up work should compare the original source partition, repeated CV, calibration, subgroup uncertainty and simpler models, using new evaluation data if modelling changes.''')
code('''delta = float(education_best.validation_accuracy-original_best.validation_accuracy)
cv_index = comparison.set_index('model')
cv_delta = float(cv_index.loc['Plus education','f1']-cv_index.loc['Original six','f1'])
display(Markdown(f"""### Actual conclusions
- Default original validation accuracy: **{baseline_accuracy:.4%}**.
- Best original validation accuracy: **{original_best.validation_accuracy:.4%}** at depth {int(original_best.depth)}.
- Best education validation accuracy: **{education_best.validation_accuracy:.4%}** at depth {int(education_best.depth)}; change **{100*delta:+.3f} percentage points**.
- At consistent depth 16, education changes five-fold F1 by **{cv_delta:+.4f}**; see the feature comparison table for the expanded and interaction ablations. No statistical significance is claimed.
- Selected feature group: **{winning_name}**. Best searched five-fold F1: **{search.best_score_:.4f}**.
- Final test accuracy **{final_metrics['accuracy']:.4%}**, F1 **{final_metrics['f1']:.4f}**, ROC-AUC **{final_metrics['roc_auc']:.4f}**.
- Top encoded predictors: **{', '.join(importance_df.head(5).feature)}**.

All results are from the downloaded UCI records. The twelve-trial search is limited by its budget. Validation gains and impurity rankings should not be interpreted as causal effects or guaranteed deployment improvements."""))
''')
n.cells = cells
n.metadata = {'kernelspec':{'display_name':'Python (Adult Income .venv312)','language':'python','name':'adult-income'},'language_info':{'name':'python','version':'3.12'}}
nbf.validate(n)
nbf.write(n,path)
print('Created',len(cells),'cells; backup:',backup)
