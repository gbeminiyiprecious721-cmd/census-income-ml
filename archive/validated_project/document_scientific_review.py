"""Write a concise README audit from executed review artifacts only."""
from pathlib import Path
import json, hashlib
import nbformat
import pandas as pd

root = Path(__file__).resolve().parent
out = root/'artifacts'
r = json.loads((out/'scientific_review.json').read_text())
independent_checks = json.loads((out/'review_independent_checks.json').read_text())
v = json.loads((out/'execution_validation.json').read_text())
importance = pd.read_csv(out/'review_importance_comparison.csv')
thresholds = pd.read_csv(out/'review_threshold_diagnostics.csv')
groups = pd.read_csv(out/'review_subgroup_metrics.csv')
configurations = pd.read_csv(out/'review_model_configurations.csv')
fixed = configurations.iloc[0]
coherent = pd.read_csv(out/'review_coherent_source_permutation.csv')
prior_search = pd.read_csv(out/'pre_review/random_search.csv')
current_search = pd.read_csv(out/'random_search.csv')
score_cols = [c for c in prior_search.columns if c.startswith('mean_test_')]
assert prior_search.params.tolist() == current_search.params.tolist()
max_search_difference = float((prior_search[score_cols]-current_search[score_cols]).abs().to_numpy().max())
metric_reproduction_differences = (prior_search[score_cols]-current_search[score_cols]).abs().max().to_dict()
for name in ['accuracy','precision','recall','f1']:
    assert metric_reproduction_differences['mean_test_'+name] == 0
assert max_search_difference < 1e-6, max_search_difference
assert prior_search[score_cols].round(4).equals(current_search[score_cols].round(4))
n = nbformat.read(root/'Adult_Income_Random_Forest_End_to_End.ipynb',4)
code = [c for c in n.cells if c.cell_type=='code']
hashes = json.loads((out/'pre_review/training_code_hashes.json').read_text())
assert hashes == [hashlib.sha256(c.source.encode()).hexdigest() for c in code[:len(hashes)]]
assert [c.execution_count for c in code] == list(range(1,len(code)+1))
assert len(code)==v['code_cells'] and not [o for c in code for o in c.outputs if o.output_type=='error']
assert r['prior_metrics_reproduced'] and not r['model_changed_by_review']
assert independent_checks['crossing_profiles_with_conflicting_income_labels']==r['shared_full_predictor_profiles']
for cell in n.cells:
    if cell.cell_type=='markdown' and cell.source.startswith('### Scientific assessment'):
        cell.source=cell.source.split('\n\n**Post-run artifact checks:**')[0]
        cell.source+=f"\n\n**Post-run artifact checks:** Accuracy, precision, recall and F1 match exactly across all 12 search candidates; maximum mean ROC-AUC variation is {max_search_difference:.10f}, with all reported means unchanged at four decimals. Floating-point/tie-order effects are a plausible explanation, not an isolated cause. Independent reconstruction confirms {r['shared_full_predictor_profiles']} complete 14-predictor profiles cross the development/test split with conflicting income labels. Full-record deduplication includes the target, so these survive. This is a confirmed data-quality/independence concern, not proof of duplicated people or a test-to-fit path. A predictor-profile-grouped split sensitivity check with new evaluation data is appropriate; the working split/model were not altered."
nbformat.validate(n)
nbformat.write(n,root/'Adult_Income_Random_Forest_End_to_End.ipynb')
from nbconvert import HTMLExporter
html,_=HTMLExporter().from_notebook_node(n)
(root/'Adult_Income_Random_Forest_End_to_End.html').write_text(html,encoding='utf-8')
def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,row))+' |' for row in rows])
gap = r['accuracy_gap_percentage_points']
selected = r['selected_model_cv']; test = r['selected_model_test']
section = f"""## Scientific review — diagnostic audit

The {len(code)} code cells executed sequentially in a fresh kernel with zero errors; all nine original code cells are unchanged (SHA-256 check). Accuracy, precision, recall and F1 match exactly for all 12 search candidates. Maximum mean ROC-AUC variation is **{max_search_difference:.10f}**; all five mean metrics agree at four decimals. Such a small variation is compatible with floating-point/tie-order effects in parallel forests, but its cause was not isolated. Fold means and confusion-matrix arithmetic reconcile; test ROC-AUC independently matches the Mann–Whitney rank calculation. Backups: `Adult_Income_Random_Forest_End_to_End.pre_review.ipynb`, `README.pre_review.md` and `artifacts/pre_review/`.

### Why 86.05% CV accuracy differs from 83.22% test accuracy

{table(['Model / evaluation','Accuracy','Positive F1','Recall'],[
    ['Fixed forest / five-fold validation',f"{r['fixed_model_cv_accuracy']:.2%}",f"{fixed.f1:.4f}",f"{fixed.recall:.2%}"],
    ['F1-selected forest / same five folds',f"{selected['accuracy']:.2%}",f"{selected['f1']:.4f}",f"{selected['recall']:.2%}"],
    ['Selected refit / held-out test',f"{test['accuracy']:.2%}",f"{test['f1']:.4f}",f"{test['recall']:.2%}"]])}

The fixed feature-comparison forest has 100 trees, depth 16, `max_features='sqrt'`, split size 2, leaf size 1 and no class weights. The selected forest has 100 trees, depth 20, `max_features=1.0`, split size 10, leaf size 1 and `class_weight='balanced_subsample'`. Search refits on **F1**, not accuracy, and its observed improvement in recall comes with lower precision/accuracy. The selected CV estimate averages five fits on 80% of development data; the final model is refit on all development records and evaluated on a separate test sample. Test precision is {test['precision']:.4f}, recall {test['recall']:.4f}, ROC-AUC {test['roc_auc']:.4f}.

The total **{gap['total']:.3f}-percentage-point** difference separates arithmetically into **{gap['configuration_and_objective']:.3f} points** between CV configurations/objectives and **{gap['selected_model_cv_minus_test']:.3f} points** between the selected model's own CV and test accuracy. This is not a causal decomposition or evidence that the whole difference is overfitting. Sampling variation, training-size differences and selection optimism are plausible contributors to the remaining gap, not established causes.

### Leakage assessment

No test-to-fit path was found: learned preprocessing is inside CV pipelines; engineering is record-local; model selection uses development data; fitted imputer statistics match development records. Sex/race are joined by index only for audit. However, **{r['shared_full_predictor_profiles']} complete 14-predictor profiles cross the split with conflicting target labels**: deduplication includes income, so these survive. This is a confirmed data-quality/independence concern, not proof of duplicate people or target leakage. Do not claim complete record independence. A predictor-profile-grouped split sensitivity analysis and person-level provenance would be appropriate future checks; the current split/model are preserved.

### Held-out permutation importance

Ten seeded permutations per pipeline input use all {r['heldout_rows']:,} test observations. Encoded impurity contributions are aggregated to the same raw-column level. Their scales are different: impurity is normalised training split credit; permutation is a held-out score decrease. Spearman rank correlation for impurity versus F1 permutation is **{r['permutation_impurity_rank_spearman']:.3f}**.

{table(['Input feature','Impurity rank','F1 permutation rank','Mean F1 decrease','Shuffle SD'],[[x.feature,int(x.impurity_rank),int(x.f1_permutation_rank),f'{x.f1_decrease_mean:.5f}',f'{x.f1_shuffle_sd:.5f}'] for x in importance.head(5).itertuples()])}

Aggregated impurity top five: {', '.join(r['aggregated_impurity_top_five'])}. Full F1/ROC-AUC rankings and repeat-level scores are saved. Shuffle SD is not a sampling confidence interval. Single-column permutation breaks correlation and, for age/hours/interaction, the deterministic interaction identity. The supplementary source permutation recomputes the interaction; its mean F1 decreases are {', '.join(f'{source}: {values.f1_decrease.mean():.5f}' for source,values in coherent.groupby('source'))}. These measure joint-source sensitivity, not a directly interchangeable single-column ranking. Neither method is causal. See [scikit-learn's permutation-importance guidance](https://scikit-learn.org/stable/modules/permutation_importance.html).

### Fixed-threshold diagnostics

These thresholds were specified before inspecting their metrics; none is selected from test performance. Predictions use `probability > threshold`, preserving the classifier's class-0 tie rule. The model remains at **0.50**.

{table(['Threshold','Precision','Recall','F1','Selected positive'],[[f'{x.threshold:.2f}',f'{x.precision:.4f}',f'{x.recall:.4f}',f'{x.f1:.4f}',f'{x.selection_rate:.2%}'] for x in thresholds.itertuples()])}

Scores are not assumed calibrated after class weighting. Future threshold decisions require development-only/OOF evidence and explicit error costs, followed by independent evaluation; see [scikit-learn's threshold guidance](https://scikit-learn.org/stable/modules/classification_threshold.html).

### Sex and race disparities at the frozen threshold

Labels reproduce the dataset's coarse historical sex/race coding and do not represent all identities. Intervals below are marginal Wilson 95% intervals, conditional on independent records; they are not simultaneous multiple-comparison intervals. Positive and negative denominators matter, especially for small groups.

{table(['Attribute / group','n','Positives','Negatives','Precision','TPR [95% interval]','FPR [95% interval]'],[[f'{x.attribute}: {x.group}',x.n,x.positives,x.negatives,f'{x.precision:.3f}',f'{x.recall_tpr:.3f} [{x.recall_tpr_ci_low:.3f}, {x.recall_tpr_ci_high:.3f}]',f'{x.fpr:.3f} [{x.fpr_ci_low:.3f}, {x.fpr_ci_high:.3f}]'] for x in groups.itertuples()])}

Full confusion counts, prevalence, selection rates, accuracy, precision, recall, FPR/FNR, F1 and ROC-AUC are saved in `artifacts/review_subgroup_metrics.csv`. Group totals reconcile to overall test confusion counts. These are descriptive error disparities: they do not establish discrimination, explain causes, or demonstrate fairness. Different group compositions and base rates affect precision/selection rates; separate sex/race summaries omit intersectional harms. Removing sensitive predictors does not remove proxies. No group-specific thresholds or interventions were chosen.

### Remaining weaknesses and interpretation

**Confirmed design limitations:** one historical pooled random split; no person identifier or contemporary external evaluation; reused folds for feature selection and 12-trial search; no nested CV; feature groups selected before tuning; simultaneous additions/removal of predictors confound isolated feature-effect claims; uncalibrated scores; correlated/derived-feature interpretation limits; small subgroup denominators; no simultaneous inference or full intersectional audit.

**Unverified hypotheses:** sampling variation or selection optimism explains the small selected-model CV/test gap; the interaction's gain generalises; proxies or group composition explain observed disparities. Further development-only experiments and new independent evaluation are needed. The test set has now been inspected extensively and cannot remain an untouched basis for later changes informed by this review. No model, hyperparameter or threshold was changed, and no deployment or production/fairness readiness is claimed.

Review tables and figures are under `artifacts/review_*`; the machine-readable assessment is `artifacts/scientific_review.json`. Re-execute with `execute_notebook.py`; then run `document_results.py`, `verify_review_artifacts.py`, and `document_scientific_review.py` in that order to refresh summaries and independent artifact checks. The original `build_notebook.py` is a reconstruction scaffold; do not run it for ordinary review execution because it clears outputs and omits this appended audit (reapply `append_scientific_review.py` if intentionally rebuilding).
"""
readme = root/'README.md'
body = readme.read_text(encoding='utf-8').split('\n## Scientific review — diagnostic audit')[0]
readme.write_text(body.rstrip()+'\n\n'+section,encoding='utf-8')
validation = {'code_cells':len(code),'sequential_counts':[c.execution_count for c in code],
    'original_code_cells_unchanged':True,'error_outputs':0,
    'embedded_figures':sum('image/png' in o.get('data',{}) for c in code for o in c.outputs),
    'scientific_review_present':True,'prior_metric_checks_passed':True}
validation['max_all_search_mean_metric_reproduction_difference'] = max_search_difference
validation['all_search_mean_metric_differences'] = metric_reproduction_differences
(out/'scientific_review_validation.json').write_text(json.dumps(validation,indent=2))
print(json.dumps(validation,indent=2))
