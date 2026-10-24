"""Author the professional notebook; normal reproduction uses execute_research.py."""
from pathlib import Path
import nbformat as nb

root=Path(__file__).resolve().parents[1]
cells=[]
def md(text): cells.append(nb.v4.new_markdown_cell(text))
def code(text): cells.append(nb.v4.new_code_cell(text))
md('''# CensusIncome ML: Explainable Income Classification and Model Evaluation
## 1. Executive Summary
This study evaluates Random Forest classifiers of historical US income categories. It asks whether additional census attributes improve positive-class F1, how model configurations trade precision for recall, and how errors vary across demographic groups. The original reviewed evidence is preserved in `archive/validated_project/`; this notebook recomputes the research using shared package functions and produces a versioned inference pipeline.

**Evidence boundary:** the previously inspected benchmark test set is held out from fitting and selection, but this rerun is a reproduction—not new independent confirmation. No threshold, feature removal or demographic policy is selected from the new test diagnostics. Final computed findings appear in section 15 and `reports/summary.json`.''')
code('''from pathlib import Path
import sys,json,platform
ROOT=Path.cwd()
if ROOT.name=='notebooks': ROOT=ROOT.parent
assert (ROOT/'src/census_income').is_dir(), 'Run from the project root or notebooks directory'
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display,Markdown
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_validate
from sklearn.metrics import classification_report,ConfusionMatrixDisplay,RocCurveDisplay,PrecisionRecallDisplay,make_scorer,precision_score
from census_income.config import REPORTS_DIR,ARTIFACTS_DIR,SEED,FEATURE_SETS,SCORING,MODEL_INPUTS
from census_income.data import load_adult,split_data,split_independence
from census_income.features import engineer_features
from census_income.train import compare_features,search_model,stratified_cv
from census_income.evaluate import export_evaluation
from census_income.explainability import permutation_diagnostics
from census_income.research import export_eda,finalize_research,write_json,LIMITATIONS
REPORTS_DIR.mkdir(exist_ok=True)
FIGURES=REPORTS_DIR/'figures'; FIGURES.mkdir(exist_ok=True)
np.random.seed(SEED)
sns.set_theme(style='whitegrid',palette=['#245782','#C17C28'])
plt.rcParams.update({'figure.figsize':(9,5),'figure.dpi':110})
def show_figure(name):
    plt.tight_layout(); plt.savefig(FIGURES/(name+'.png'),bbox_inches='tight'); plt.show()
print('Interpreter:',sys.executable,platform.python_version(),'Seed:',SEED)
''')
md('''## 2. Analytical Problem and Research Questions
The target is income above $50,000 in the historical observation period, encoded as 1; income at or below $50,000 is 0. Primary model selection uses positive-class F1. Accuracy, precision, recall and ROC-AUC provide complementary evidence. The majority-class baseline helps interpret accuracy under imbalance.

The objectives are comparative research, reproducible evaluation and an educational inference demonstration. The project does not estimate current salaries or justify employment, lending, eligibility or other consequential decisions.''')
md('''## 3. Dataset Provenance and Limitations
[UCI Adult](https://archive.ics.uci.edu/dataset/2/adult), Becker & Kohavi (1996), DOI [10.24432/C5XW20](https://doi.org/10.24432/C5XW20), CC BY 4.0. Records derive from the 1994 Census database. The source `adult.data` and `adult.test` files are pooled for comparability with the reviewed study. SHA-256 checks reject unexpected source files. Metrics and summaries are unweighted per observation: `fnlwgt` is neither a predictor nor a sample weight. They are not population estimates.''')
code('''df,audit=load_adult(download=False)
display(pd.Series({k:audit[k] for k in ['raw_rows','raw_columns','clean_rows','duplicates_removed','source_year']}))
display(pd.DataFrame(audit['provenance']).T)
''')
md('''## 4. Data Quality Assessment
Categorical whitespace and target terminal periods are normalised, `?` becomes missing, and exact **full-record** duplicates are removed before splitting. Missing predictors remain for fold-local imputation. Category spellings follow the historical source. Full-record deduplication includes income and therefore does not remove identical predictors with conflicting targets. The split audit reports those profiles explicitly; there is no person identifier to establish individual-level independence.''')
code('''quality=pd.DataFrame({'dtype':audit['dtypes'],'missing_before_deduplication':audit['missing_before_deduplication']})
display(quality)
X_dev,X_test,y_dev,y_test=split_data(df)
independence=split_independence(df,X_dev,X_test)
independence.update(development_rows=len(X_dev),test_rows=len(X_test))
display(pd.Series(independence))
assert audit['raw_rows']==48842 and audit['clean_rows']==48790
assert y_dev.index.equals(X_dev.index) and y_test.index.equals(X_test.index)
eda=export_eda(df,audit)
''')
md('''## 5. Exploratory Data Analysis
Class counts describe imbalance in this historical sample. Age and weekly-hours distributions show feature spread; descriptive group relationships are observational rather than causal. The application offers aggregate exploration plus explicit browsing and export of public historical UCI rows, without person identifiers. No test-specific exploration is used to choose predictors or thresholds.''')
code('''counts=df.income.value_counts().rename_axis('Income class').to_frame('Records')
counts['Sample proportion']=counts.Records/len(df)
display(counts)
fig,axes=plt.subplots(1,3,figsize=(15,4))
counts.Records.plot.bar(ax=axes[0],rot=0,color='#245782'); axes[0].set(title='Historical income distribution',ylabel='Records')
axes[1].hist(df.age,bins=25,color='#245782'); axes[1].set(title='Age distribution',xlabel='Age (years)',ylabel='Records')
axes[2].hist(df['hours-per-week'],bins=25,color='#245782'); axes[2].set(title='Weekly work hours',xlabel='Hours per week',ylabel='Records')
show_figure('data_distributions')
display(df.groupby('income')[['age','hours-per-week','education-num']].median().rename_axis('Income class'))
''')
md('''## 6. Feature Engineering and Preprocessing
The saved pipeline owns both record-local engineering and learned preprocessing. Education codes 1–9 become “HS or less”, 10–13 “College to Bachelors”, and 14–16 “Postgraduate”. Age × weekly hours is an interaction hypothesis, not measured annual work. Median numeric imputation, most-frequent categorical imputation and one-hot encoding are fitted inside each CV training fold. Unseen categories are ignored by encoding and warned about by the validated inference interface.

Four feature sets reproduce the reviewed comparison. Expanded sets add occupation, marital status, relationship, workclass and country, and omit direct sex/race predictors. Because additions and removals occur together, this is not an isolated effect estimate. Omission of sensitive fields does not remove proxies or establish fairness.''')
code('''display(pd.DataFrame([{'Feature group':name,'Predictors':', '.join(cols)} for name,cols in FEATURE_SETS.items()]))
engineered=engineer_features(X_dev.iloc[:3])
assert np.array_equal(engineered.hours_age_interaction,X_dev.iloc[:3].age*X_dev.iloc[:3]['hours-per-week'])
print('Deterministic engineering is part of the pipeline; no fit uses test records.')
''')
md('''## 7. Baseline Model Development
The majority-class baseline and original-six-feature Random Forest contextualise model performance. All feature-group forests have 100 trees and maximum depth 16, with default `sqrt` feature subsampling, minimum split 2, minimum leaf 1 and no class weighting. They use the same shuffled five-fold stratification on development data. Fold standard deviations describe fold variation, not confidence intervals.''')
code('''dummy_scoring={**SCORING,'precision':make_scorer(precision_score,zero_division=0)}
dummy=cross_validate(DummyClassifier(strategy='most_frequent'),X_dev,y_dev,cv=stratified_cv(),scoring=dummy_scoring,n_jobs=1)
dummy_metrics={m:float(dummy['test_'+m].mean()) for m in SCORING}
comparison,fold_scores=compare_features(X_dev,y_dev)
comparison.to_csv(REPORTS_DIR/'feature_comparison.csv',index=False)
fold_scores.to_csv(REPORTS_DIR/'feature_comparison_folds.csv',index=False)
display(pd.Series(dummy_metrics,name='Majority-class baseline: five-fold means'))
display(comparison.round(4))
winning_group=str(comparison.iloc[0]['model'])
comparison.set_index('model')[['train_f1','f1']].plot.bar(rot=15)
plt.ylabel('Positive-class F1'); plt.xlabel('Feature group'); plt.title('Training and validation F1: fixed forests')
show_figure('feature_comparison')
''')
md('''## 8. Hyperparameter Optimisation
RandomizedSearchCV samples 12 seeded configurations across five folds. It tunes tree count, depth, feature subsampling, minimum split/leaf sizes and class weighting, refitting on mean positive-class F1. Search uses one candidate at a time and at most four forest threads. The finite budget is not exhaustive. Feature-group selection precedes tuning and reuses the same folds; consequently the best CV score is not an unbiased nested-CV estimate. The test set is not consulted for selection.''')
code('''search=search_model(X_dev,y_dev,winning_group,n_iter=12)
best_model=search.best_estimator_
search_results=pd.DataFrame(search.cv_results_)
search_results.to_csv(REPORTS_DIR/'random_search.csv',index=False)
print('Selected feature group:',winning_group)
print('Selected hyperparameters:',search.best_params_)
print('Selected CV F1:',search.best_score_)
''')
md('''## 9. Cross-Validation Results
The fixed comparison forest and F1-selected forest have different configurations and objectives. A lower accuracy after F1 selection may accompany increased recall; it is not automatically evidence of failed generalisation. Training–validation gaps describe fit to development folds. The historical tutorial's depth sweeps are retained in the archive, not repeated as redundant coursework experiments here.''')
code('''selected_cv={m:float(search_results.loc[search.best_index_,'mean_test_'+m]) for m in SCORING}
cv_rows=[{'model':'Majority-class baseline','evaluation':'five-fold CV',**dummy_metrics}]
cv_rows += [{'model':row.model,'evaluation':'fixed forest: five-fold CV',**{m:float(getattr(row,m)) for m in SCORING}} for row in comparison.itertuples()]
cv_rows += [{'model':'F1-selected forest','evaluation':'five-fold CV',**selected_cv}]
display(pd.DataFrame(cv_rows).round(4))
for metric in SCORING:
    assert np.allclose(search_results[[f'split{k}_test_{metric}' for k in range(5)]].mean(axis=1),search_results['mean_test_'+metric])
plot=search_results.sort_values('mean_test_f1').reset_index(drop=True)
plt.plot(plot.index+1,plot.mean_train_f1,'o-',label='Training F1')
plt.plot(plot.index+1,plot.mean_test_f1,'s--',label='Validation F1')
plt.xlabel('Candidate ranked by validation F1'); plt.ylabel('Positive-class F1'); plt.title('Search fit comparison'); plt.legend()
show_figure('search_fit_comparison')
''')
md('''## 10. Final Held-Out Evaluation
The selected configuration is refit on all development records, then evaluated at the classifier's unchanged argmax/0.50 rule. Classification metrics, confusion matrix, ROC and precision–recall curves use the same frozen predictions. The current test sample was previously inspected: these are reproducibility measurements, not a new external validation. Neither its scores nor later diagnostics restart tuning.''')
code('''final_pred=best_model.predict(X_test)
final_score=best_model.predict_proba(X_test)[:,1]
test_metrics,thresholds,groups,calibration,calibration_metrics=export_evaluation(y_test,final_pred,final_score,df.loc[X_test.index,['sex','race']],REPORTS_DIR)
assert np.array_equal(final_pred,(final_score>.5).astype(int))
display(pd.Series(test_metrics,name='Held-out reproduction'))
print(classification_report(y_test,final_pred,target_names=['Income <=50K','Income >50K'],zero_division=0))
pd.DataFrame(cv_rows+[{'model':'F1-selected forest','evaluation':'held-out test reproduction',**test_metrics}]).to_csv(REPORTS_DIR/'model_comparison.csv',index=False)
fig,axes=plt.subplots(1,3,figsize=(17,5))
ConfusionMatrixDisplay.from_predictions(y_test,final_pred,ax=axes[0],display_labels=['<=50K','>50K'],colorbar=False,cmap='Blues'); axes[0].set_title('Confusion matrix')
RocCurveDisplay.from_predictions(y_test,final_score,ax=axes[1],plot_chance_level=True); axes[1].set_title('ROC curve')
PrecisionRecallDisplay.from_predictions(y_test,final_score,ax=axes[2]); axes[2].axhline(y_test.mean(),color='grey',linestyle=':'); axes[2].set_title('Precision–recall curve')
show_figure('heldout_evaluation')
''')
md('''## 11. Model Explainability
Impurity credits are aggregated from encoded columns to model-feature columns. Ten seeded held-out permutations measure F1 and ROC-AUC decreases. Impurity and permutation use different scales; rankings are model-specific, not causal. Shuffle standard deviation is not a sampling confidence interval. Single-column perturbation of age/hours/interaction breaks their deterministic identity; complementary source-input permutation through the full pipeline recomputes the interaction. Neither removes all correlation concerns.

Method: [scikit-learn permutation importance](https://scikit-learn.org/stable/modules/permutation_importance.html).''')
code('''importance,coherent=permutation_diagnostics(best_model,X_test,y_test,winning_group,REPORTS_DIR,n_repeats=10)
display(importance.round(5))
display(coherent.groupby('source')[['f1_decrease','roc_auc_decrease']].agg(['mean','std']))
ordered=importance.sort_values('f1_decrease_mean')
fig,axes=plt.subplots(1,2,figsize=(14,6))
axes[0].barh(ordered.feature,ordered.impurity_importance,color='#245782'); axes[0].set(xlabel='Aggregated training impurity',title='Impurity importance')
axes[1].barh(ordered.feature,ordered.f1_decrease_mean,xerr=ordered.f1_shuffle_sd,color='#C17C28'); axes[1].set(xlabel='Held-out F1 decrease (shuffle SD)',title='Permutation importance')
show_figure('importance_comparison')
''')
md('''## 12. Threshold Sensitivity Analysis
Thresholds 0.20, 0.35, 0.50, 0.65 and 0.80 are predeclared descriptive probes. Every result is shown in threshold order; none is selected from test performance. The saved classifier retains its original rule. Scores are not calibrated current-income probabilities. A future threshold policy would require development-only/OOF evidence, explicit costs and independent evaluation.''')
code('''display(thresholds.round(4))
assert thresholds.recall.is_monotonic_decreasing and thresholds.selection_rate.is_monotonic_decreasing
plt.plot(thresholds.threshold,thresholds.precision,'o-',label='Precision')
plt.plot(thresholds.threshold,thresholds.recall,'s--',label='Recall')
plt.axvline(.5,color='grey',linestyle=':',label='Unchanged decision rule')
plt.xlabel('Diagnostic threshold'); plt.ylabel('Positive-class metric'); plt.ylim(0,1); plt.legend()
plt.title('Threshold trade-off: no threshold selected'); show_figure('threshold_diagnostics')
''')
md('''## 13. Demographic Performance Assessment
Sex and race are joined by held-out row index for audit only. Rates are presented with positive/negative denominators, confusion counts and marginal Wilson 95% intervals. These intervals assume independent records, are not simultaneous comparisons, and do not capture dataset or deployment bias. Small groups can have very wide uncertainty. The source's coarse historical categories do not represent all identities. Different base rates and compositions influence precision/selection rates; separate sex/race views do not cover intersectional harms. Measured error disparities do not prove discrimination or establish fairness.''')
code('''display(groups[['attribute','group','n','positives','negatives','accuracy','precision','recall_tpr','fpr','fnr','f1','roc_auc']].round(4))
display(groups[['attribute','group','recall_tpr_ci_low','recall_tpr_ci_high','fpr_ci_low','fpr_ci_high']].round(4))
fig,axes=plt.subplots(1,2,figsize=(14,6))
labels=groups.attribute+': '+groups.group
for ax,metric,title in zip(axes,['recall_tpr','fpr'],['True-positive rate','False-positive rate']):
    values=groups[metric].to_numpy()
    errors=np.vstack([np.maximum(0,values-groups[metric+'_ci_low']),np.maximum(0,groups[metric+'_ci_high']-values)])
    ax.errorbar(values,np.arange(len(groups)),xerr=errors,fmt='o',capsize=3)
    ax.set_yticks(np.arange(len(groups)),labels); ax.set(xlim=(0,1),xlabel='Rate (Wilson 95% interval)',title=title)
show_figure('subgroup_rates')
''')
md('''## 14. Methodological Limitations
The preserved split contains conflicting cross-split predictor profiles; a profile-grouped sensitivity analysis is future work, not silently substituted here. Reused model-selection folds, finite search, one pooled random split, unweighted survey observations, historical labels and uncalibrated scores limit interpretation. Feature group changes confound added variables and sensitive-feature removal. No causal, fairness or deployment-readiness conclusion follows.

Fixed-bin calibration adds measured evidence about score agreement on this inspected sample. It does not fit a calibrator. Brier score, log loss and expected calibration error depend on the sample and binning; they do not establish contemporary generalisation.''')
code('''display(pd.Series(calibration_metrics))
display(calibration.round(4))
plt.plot([0,1],[0,1],':',color='grey',label='Ideal agreement')
plt.plot(calibration.mean_score,calibration.observed_fraction,'o-',label='Observed test bins')
plt.xlabel('Mean uncalibrated model score'); plt.ylabel('Observed higher-income fraction')
plt.title('Held-out score calibration diagnostic'); plt.legend(); show_figure('calibration')
display(pd.Series(LIMITATIONS,name='Research limitations'))
''')
md('''## 15. Conclusions and Future Research
The following result is computed after all diagnostics, reconciled to the archived measurements, and saved with source hashes, dependency versions, selected parameters and a pipeline checksum. This preserves the historical benchmark rather than presenting old measurements as fresh results.

Confirmed findings are model comparisons, measured error rates, score calibration diagnostics and observed importance rankings. Causes of demographic disparities, stable interaction benefits, contemporary performance and the source of any small CV/test difference remain hypotheses. Future work needs development-only experiments, grouped/temporal evaluation, repeated or nested CV, calibration and intersectional audits, followed by new independent data if decisions change. Nothing is deployed or certified for consequential use.''')
code('''summary,manifest=finalize_research(best_model,winning_group,search,comparison,fold_scores,test_metrics,audit,independence)
from census_income.inference import load_validated_model,predict_records
restored=load_validated_model()
probe=X_test[MODEL_INPUTS].iloc[:5].copy()
np.testing.assert_allclose(restored.predict_proba(probe),best_model.predict_proba(X_test.iloc[:5]),atol=1e-12)
inference_output=predict_records(probe)
assert np.isfinite(inference_output.model_score).all()
fixed=comparison.set_index('model').loc[winning_group]
configuration_gap=100*(fixed.accuracy-selected_cv['accuracy'])
evaluation_gap=100*(selected_cv['accuracy']-test_metrics['accuracy'])
write_json(REPORTS_DIR/'scientific_assessment.json',{'configuration_accuracy_difference_pp':configuration_gap,
    'selected_cv_test_accuracy_difference_pp':evaluation_gap,'independence':independence,
    'calibration':calibration_metrics,'permutation_top_five':importance.head(5).feature.tolist(),
    'model_changed_from_historical_selection':False,'threshold_selected_from_test':False,
    'interpretation':'Arithmetic separation of model/configuration and evaluation differences; not causal attribution.'})
display(Markdown(f"""**Computed held-out reproduction:** accuracy **{test_metrics['accuracy']:.2%}**, F1 **{test_metrics['f1']:.4f}**, ROC-AUC **{test_metrics['roc_auc']:.4f}**.

Selected model CV accuracy is **{selected_cv['accuracy']:.2%}** and F1 **{selected_cv['f1']:.4f}**. Compared with the fixed forest's **{fixed.accuracy:.2%}** CV accuracy, the configuration/objective difference is **{configuration_gap:.3f} percentage points**; the selected configuration's CV-to-test difference is **{evaluation_gap:.3f} points**. They must not be conflated.

Fresh computations match archived metrics at reported precision. **{independence['conflicting_cross_split_profiles']} conflicting complete predictor profiles** cross the preserved split; person-level independence remains unestablished. Permutation top inputs: **{', '.join(importance.head(5).feature)}**.

Fitted pipeline version **{summary['model_version']}** has a recorded checksum and dependency contract. Saved-pipeline inference matches the research pipeline. These checks support reproducible educational inference, not production or fairness readiness."""))
''')
n=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'census-income','display_name':'Python (CensusIncome ML)','language':'python'},'language_info':{'name':'python','version':'3.12'}})
nb.validate(n)
(root/'notebooks').mkdir(exist_ok=True)
nb.write(n,root/'notebooks/01_income_classification_research.ipynb')
print('Created professional research notebook:',len(cells),'cells')
