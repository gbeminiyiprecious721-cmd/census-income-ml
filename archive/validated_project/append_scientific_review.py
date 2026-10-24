"""Append a diagnostic review; preserve every existing training/evaluation code cell."""
from pathlib import Path
import json, shutil, hashlib
import nbformat as nb

root = Path(__file__).resolve().parent
p = root/'Adult_Income_Random_Forest_End_to_End.ipynb'
backup = root/'Adult_Income_Random_Forest_End_to_End.pre_review.ipynb'
if not backup.exists(): shutil.copy2(p,backup)
baseline = root/'artifacts'/'pre_review'
baseline.mkdir(exist_ok=True)
for name in ['results.json','feature_comparison.csv','feature_comparison_folds.csv','random_search.csv','execution_validation.json']:
    if not (baseline/name).exists(): shutil.copy2(root/'artifacts'/name,baseline/name)
if not (root/'README.pre_review.md').exists(): shutil.copy2(root/'README.md',root/'README.pre_review.md')
n = nb.read(p,4)
marker = '## Scientific review — diagnostic audit'
for i,c in enumerate(n.cells):
    if c.cell_type=='markdown' and c.source.startswith(marker):
        n.cells=n.cells[:i]; break
original_codes = [hashlib.sha256(c.source.encode()).hexdigest() for c in n.cells if c.cell_type=='code']
(baseline/'training_code_hashes.json').write_text(json.dumps(original_codes,indent=2))
def md(s): n.cells.append(nb.v4.new_markdown_cell(s))
def code(s): n.cells.append(nb.v4.new_code_cell(s))
md('''## Scientific review — diagnostic audit
This review preserves the training implementation and freezes its selected feature group, hyperparameters and 0.50 decision rule. New held-out analyses are **post-selection diagnostics**, not new model/threshold selection. The test set is now an inspected audit set: any later modelling decisions informed by these diagnostics require new evaluation data.

### Verification and leakage assessment
The clean-kernel rerun below checks the previously reported metrics against preserved artifacts, reconciles fold means, independently recomputes ROC-AUC via the Mann–Whitney rank identity, and verifies fitted preprocessing statistics against development records. Source review confirms that imputation/encoding occur inside CV pipelines; fixed education bins and the interaction use each record only; feature-group and hyperparameter selection use development data only. Schema inspection, deterministic cleaning and deduplication precede splitting, without fitting target-dependent transformations. These checks support absence of test-to-fit leakage in this code path; they cannot establish person-level independence without identifiers.''')
code('''from scipy.stats import rankdata, norm, spearmanr
from sklearn.inspection import permutation_importance
import hashlib
audit_prior = json.loads((OUT/'pre_review'/'results.json').read_text())
for metric, value in audit_prior['test_metrics'].items():
    assert np.isclose(final_metrics[metric],value,atol=1e-10,rtol=0), (metric,value,final_metrics[metric])
assert search.best_params_ == audit_prior['best_parameters']
assert winning_name == audit_prior['selected_features']
assert np.isclose(search.best_score_,audit_prior['best_cv_f1'],atol=1e-10,rtol=0)
audit_old_comparison = pd.read_csv(OUT/'pre_review'/'feature_comparison.csv').set_index('model')
audit_new_comparison = comparison.set_index('model')
for metric in scoring:
    assert np.allclose(audit_new_comparison.loc[audit_old_comparison.index,metric],audit_old_comparison[metric],atol=1e-10,rtol=0)
audit_cv_results = pd.DataFrame(search.cv_results_)
for metric in scoring:
    fold_mean = audit_cv_results[[f'split{k}_test_{metric}' for k in range(5)]].mean(axis=1)
    assert np.allclose(fold_mean,audit_cv_results['mean_test_'+metric])
audit_selected = audit_cv_results.loc[search.best_index_]
assert search.best_index_ == int(audit_cv_results.mean_test_f1.idxmax())
audit_y = y_test.to_numpy()
audit_npos = int(audit_y.sum()); audit_nneg = len(audit_y)-audit_npos
audit_rank_auc = (rankdata(final_probability)[audit_y==1].sum()-audit_npos*(audit_npos+1)/2)/(audit_npos*audit_nneg)
assert np.isclose(audit_rank_auc,final_metrics['roc_auc'])
assert np.array_equal(final_pred,(final_probability > .5).astype(int))  # sklearn argmax sends a tie to class 0
audit_pre = best_model['preprocess']
audit_num = list(audit_pre.transformers_[0][2]); audit_cat = list(audit_pre.transformers_[1][2])
assert np.allclose(audit_pre.named_transformers_['numeric'].statistics_,X_dev[audit_num].median().to_numpy())
audit_modes = [X_dev[c].mode().iloc[0] for c in audit_cat]
assert list(audit_pre.named_transformers_['categorical']['impute'].statistics_) == audit_modes
for train_pos, val_pos in cv.split(X_dev,y_dev):
    assert set(X_dev.index[train_pos]).isdisjoint(X_dev.index[val_pos])
    assert set(X_dev.index[train_pos]).isdisjoint(X_test.index)
audit_raw_hash = pd.util.hash_pandas_object(df[columns[:-1]],index=False)
audit_profile_overlap = len(set(audit_raw_hash.loc[X_dev.index]) & set(audit_raw_hash.loc[X_test.index]))
audit_same_label_ties = int(np.count_nonzero(final_probability==.5))
audit_config_table = pd.DataFrame([
    {'model':'Fixed feature-comparison forest','evaluation':'Mean of five validation folds',
     'n_estimators':100,'max_depth':16,'max_features':'sqrt','min_samples_split':2,'min_samples_leaf':1,'class_weight':'None',
     **{m:float(audit_new_comparison.loc[winning_name,m]) for m in scoring}},
    {'model':'F1-selected search forest','evaluation':'Mean of same five validation folds',
     **{k.replace('forest__',''):v for k,v in search.best_params_.items()},
     **{m:float(audit_selected['mean_test_'+m]) for m in scoring}},
    {'model':'F1-selected search forest','evaluation':'Refit on development; held-out test',
     **{k.replace('forest__',''):v for k,v in search.best_params_.items()},**final_metrics}])
display(audit_config_table.round(4))
audit_config_table.to_csv(OUT/'review_model_configurations.csv',index=False)
print('Prior scores reproduced; fold means reconciled; rank-based AUC:',audit_rank_auc)
print('Shared complete 14-predictor profiles across development/test:',audit_profile_overlap)
print('Exactly 0.50 test probabilities:',audit_same_label_ties)
''')
md('''### Permutation versus impurity importance
We permute each original **pipeline input column** on all held-out observations, with ten fixed-seed repeats, scoring the decrease in positive-class F1 and ROC-AUC. One-hot impurity importances are summed back to the same input-column level; their normalised training impurity scale is not numerically comparable with a held-out score decrease. Repeat standard deviations reflect shuffle variability, **not confidence intervals or sampling uncertainty**. Correlated features can mask or share importance; shuffling may create implausible combinations. Rankings are model-specific and noncausal.

Standard single-column permutation breaks the identity `interaction = age × hours`. We additionally permute age or hours and **recompute the interaction**, producing coherent joint-source sensitivity estimates. These perturb different column groups, so they are complementary rather than a replacement ranking.

References: [scikit-learn permutation importance](https://scikit-learn.org/stable/modules/permutation_importance.html), [decision thresholds](https://scikit-learn.org/stable/modules/classification_threshold.html).''')
code('''audit_repeats = 10
audit_perm = permutation_importance(best_model,X_test[winning_cols],y_test,n_repeats=audit_repeats,
    random_state=SEED,n_jobs=1,scoring={'f1':'f1','roc_auc':'roc_auc'})
audit_mdi = {}; audit_pos = 0
for c in audit_num:
    audit_mdi[c] = float(best_model['forest'].feature_importances_[audit_pos]); audit_pos += 1
audit_encoder = audit_pre.named_transformers_['categorical']['encode']
for c,categories in zip(audit_cat,audit_encoder.categories_):
    width = len(categories)
    audit_mdi[c] = float(best_model['forest'].feature_importances_[audit_pos:audit_pos+width].sum()); audit_pos += width
assert audit_pos == len(best_model['forest'].feature_importances_)
assert np.isclose(sum(audit_mdi.values()),1)
audit_importance = pd.DataFrame({'feature':winning_cols,'impurity_importance':[audit_mdi[c] for c in winning_cols]})
for metric, obj in audit_perm.items():
    audit_importance[metric+'_decrease_mean'] = obj.importances_mean
    audit_importance[metric+'_shuffle_sd'] = obj.importances_std
    pd.DataFrame(obj.importances.T,columns=winning_cols).to_csv(OUT/f'review_permutation_{metric}_repeats.csv',index=False)
audit_importance['impurity_rank'] = audit_importance.impurity_importance.rank(ascending=False,method='min')
audit_importance['f1_permutation_rank'] = audit_importance.f1_decrease_mean.rank(ascending=False,method='min')
audit_importance = audit_importance.sort_values('f1_decrease_mean',ascending=False)
display(audit_importance.round(5))
audit_importance.to_csv(OUT/'review_importance_comparison.csv',index=False)
audit_spearman = float(spearmanr(audit_importance.impurity_importance,audit_importance.f1_decrease_mean).statistic)
fig, axes = plt.subplots(1,2,figsize=(14,6))
audit_plot = audit_importance.sort_values('f1_decrease_mean')
axes[0].barh(audit_plot.feature,audit_plot.impurity_importance,color='#245782')
axes[0].set(xlabel='Aggregated training impurity importance',title='Impurity: raw input columns')
axes[1].barh(audit_plot.feature,audit_plot.f1_decrease_mean,xerr=audit_plot.f1_shuffle_sd,color='#C17C28')
axes[1].axvline(0,color='grey',linewidth=1)
axes[1].set(xlabel='Held-out F1 decrease (error bars: shuffle SD)',title='Permutation: ten repeats')
savefig('review_importance_comparison')
audit_coherent_rows = []
if 'hours_age_interaction' in winning_cols:
    for source in ['age','hours-per-week']:
        rng = np.random.default_rng(SEED)
        for repeat in range(audit_repeats):
            perturbed = X_test.copy()
            perturbed[source] = X_test[source].to_numpy()[rng.permutation(len(X_test))]
            perturbed['hours_age_interaction'] = perturbed['age']*perturbed['hours-per-week']
            probability = best_model.predict_proba(perturbed)[:,1]
            audit_coherent_rows.append({'source':source,'repeat':repeat+1,
                'f1_decrease':final_metrics['f1']-f1_score(y_test,(probability>.5).astype(int)),
                'roc_auc_decrease':final_metrics['roc_auc']-roc_auc_score(y_test,probability)})
audit_coherent = pd.DataFrame(audit_coherent_rows)
if len(audit_coherent):
    display(audit_coherent.groupby('source')[['f1_decrease','roc_auc_decrease']].agg(['mean','std']))
    audit_coherent.to_csv(OUT/'review_coherent_source_permutation.csv',index=False)
print('Spearman correlation between impurity and F1 permutation rankings:',audit_spearman)
''')
md('''### Fixed-threshold diagnostic: no threshold selection
The thresholds **0.20, 0.35, 0.50, 0.65 and 0.80** were fixed before examining their results. Report all of them without ranking, optimisation or a recommendation. Predictions use `probability > threshold`, preserving the classifier's class-0 tie behaviour at 0.50. The fitted classifier and 0.50 rule remain frozen. Class weighting means these scores are not assumed calibrated population probabilities. Threshold changes for a real use case would need development-only/OOF analysis, explicit error costs and independent validation.''')
code('''audit_thresholds = [.20,.35,.50,.65,.80]
audit_threshold_rows = []
for threshold in audit_thresholds:
    pred = (final_probability>threshold).astype(int)
    tn_t,fp_t,fn_t,tp_t = confusion_matrix(y_test,pred,labels=[0,1]).ravel()
    audit_threshold_rows.append({'threshold':threshold,'accuracy':accuracy_score(y_test,pred),
        'precision':precision_score(y_test,pred,zero_division=0),'recall':recall_score(y_test,pred,zero_division=0),
        'f1':f1_score(y_test,pred,zero_division=0),'selection_rate':float(pred.mean()),
        'tn':int(tn_t),'fp':int(fp_t),'fn':int(fn_t),'tp':int(tp_t)})
audit_threshold_table = pd.DataFrame(audit_threshold_rows)
assert audit_threshold_table.recall.is_monotonic_decreasing
assert audit_threshold_table.selection_rate.is_monotonic_decreasing
assert np.array_equal((final_probability>.5).astype(int),final_pred)
assert np.isclose(audit_threshold_table.loc[audit_threshold_table.threshold==.5,'f1'].iloc[0],final_metrics['f1'])
display(audit_threshold_table.round(4))
audit_threshold_table.to_csv(OUT/'review_threshold_diagnostics.csv',index=False)
plt.plot(audit_threshold_table.threshold,audit_threshold_table.precision,'o-',label='Precision')
plt.plot(audit_threshold_table.threshold,audit_threshold_table.recall,'s--',label='Recall')
plt.axvline(.5,color='grey',linestyle=':',label='Frozen classifier threshold')
plt.xlabel('Diagnostic threshold'); plt.ylabel('Positive-class metric'); plt.ylim(0,1)
plt.title('Held-out precision–recall trade-off: no threshold selected'); plt.legend()
savefig('review_threshold_diagnostics')
''')
md('''### Sex and race subgroup audit at the frozen 0.50 rule
Sensitive attributes are joined by held-out row index **for evaluation only**; they are not added to the selected model. Each row reports support, positive/negative denominators, confusion counts, prevalence, selection rate, accuracy, precision, recall/TPR, FPR, FNR, F1 and ROC-AUC. Wilson 95% intervals for TPR, FPR and precision quantify binomial sampling uncertainty under an independent-record assumption; they are not simultaneous intervals and do not address dataset bias or related individuals. Undefined denominators produce missing metrics, not invented zeros. Small-group results need particular caution. Separate sex/race summaries do not cover intersections, age groups or all relevant harms; different base rates affect precision and selection rates. Descriptive disparities neither prove discrimination nor establish fairness.''')
code('''def audit_ratio(num,den): return float(num/den) if den else float('nan')
def audit_wilson(num,den):
    if not den: return (float('nan'),float('nan'))
    z = norm.ppf(.975); p = num/den; divisor = 1+z*z/den
    centre = (p+z*z/(2*den))/divisor
    half = z*np.sqrt(p*(1-p)/den+z*z/(4*den*den))/divisor
    return float(max(0,centre-half)),float(min(1,centre+half))
audit_sensitive = df.loc[X_test.index,['sex','race']].copy()
assert audit_sensitive.index.equals(y_test.index)
audit_group_rows = []
for attribute in ['sex','race']:
    for group in sorted(audit_sensitive[attribute].unique()):
        mask = audit_sensitive[attribute].eq(group).to_numpy()
        truth = audit_y[mask]; pred = final_pred[mask]; prob = final_probability[mask]
        tn_g,fp_g,fn_g,tp_g = map(int,confusion_matrix(truth,pred,labels=[0,1]).ravel())
        positives = tp_g+fn_g; negatives = tn_g+fp_g
        row = {'attribute':attribute,'group':group,'n':int(mask.sum()),'positives':positives,'negatives':negatives,
            'tn':tn_g,'fp':fp_g,'fn':fn_g,'tp':tp_g,'prevalence':audit_ratio(positives,int(mask.sum())),
            'selection_rate':float(pred.mean()),'accuracy':float((tn_g+tp_g)/mask.sum()),
            'precision':audit_ratio(tp_g,tp_g+fp_g),'recall_tpr':audit_ratio(tp_g,positives),
            'fpr':audit_ratio(fp_g,negatives),'fnr':audit_ratio(fn_g,positives),
            'f1':audit_ratio(2*tp_g,2*tp_g+fp_g+fn_g),
            'roc_auc':roc_auc_score(truth,prob) if positives and negatives else float('nan')}
        for metric,num,den in [('recall_tpr',tp_g,positives),('fpr',fp_g,negatives),('precision',tp_g,tp_g+fp_g)]:
            row[metric+'_ci_low'],row[metric+'_ci_high'] = audit_wilson(num,den)
        audit_group_rows.append(row)
audit_groups = pd.DataFrame(audit_group_rows)
display(audit_groups[['attribute','group','n','positives','negatives','prevalence','selection_rate','accuracy','precision','recall_tpr','fpr','fnr','f1','roc_auc']].round(4))
display(audit_groups[['attribute','group','recall_tpr_ci_low','recall_tpr_ci_high','fpr_ci_low','fpr_ci_high','precision_ci_low','precision_ci_high']].round(4))
audit_groups.to_csv(OUT/'review_subgroup_metrics.csv',index=False)
for attribute in ['sex','race']:
    subset = audit_groups[audit_groups.attribute==attribute]
    assert int(subset.n.sum()) == len(y_test)
    assert np.array_equal(subset[['tn','fp','fn','tp']].sum().to_numpy(),confusion_matrix(y_test,final_pred).ravel())
audit_gaps = audit_groups.groupby('attribute')[['recall_tpr','fpr','precision','selection_rate']].agg(lambda x:x.max()-x.min())
display(audit_gaps.rename(columns=lambda c:c+'_max_minus_min').round(4))
audit_gaps.to_csv(OUT/'review_subgroup_gaps.csv')
fig,axes = plt.subplots(1,2,figsize=(14,6))
labels = audit_groups.attribute+': '+audit_groups.group
for ax,metric,title in zip(axes,['recall_tpr','fpr'],['True-positive rate','False-positive rate']):
    values = audit_groups[metric].to_numpy()
    errors = np.vstack([np.maximum(0,values-audit_groups[metric+'_ci_low'].to_numpy()),np.maximum(0,audit_groups[metric+'_ci_high'].to_numpy()-values)])
    ax.errorbar(values,np.arange(len(labels)),xerr=errors,fmt='o',capsize=3,color='#245782')
    ax.set_yticks(np.arange(len(labels)),labels)
    ax.set(xlim=(0,1),xlabel='Rate (Wilson 95% interval)',title=title)
savefig('review_subgroup_rates')
''')
md('''### Scientific assessment
**Confirmed by the implementation and rerun:** 86.05% is the mean validation accuracy of the fixed 100-tree, depth-16, `sqrt`, unweighted forest used to compare feature groups. It is not the selected tuned model's CV accuracy, nor a score for its final refit. The F1-selected configuration uses depth 20, all features per split, split size 10 and `balanced_subsample`. The configuration table separates this model's own CV scores from the refit test scores. This separates a model/objective difference from a CV-versus-test difference; it is not valid to attribute the entire 2.83-point accuracy difference to generalisation loss.

**Methodological limitations:** feature-group selection and search reuse five development folds, so the best CV score is selection-biased rather than a nested-CV estimate. Only 12 configurations were searched; choosing the feature group before tuning may miss another group's better tuned model. One pooled random split differs from UCI's original source partition and does not establish temporal generalisation. Exact deduplication cannot establish person-level independence. The expanded comparison changes both added predictors and removal of sex/race, so it does not isolate any one feature's causal contribution. Fixed-threshold comparisons, multiple subgroup comparisons and importance inspection are descriptive; no test threshold, subgroup-specific policy or feature removal is selected. ROC-AUC does not establish calibration or decision usefulness. No deployment or consequential use is warranted by this review.

**Hypotheses requiring new validation:** sampling variation and reused-fold selection may contribute to the selected model's CV/test gap; the interaction may provide stable incremental signal; sensitive-feature proxies and different group compositions may contribute to error disparities. These explanations have not been causally identified or tested for generalisation. New development-only experiments, calibration and uncertainty analysis, intersectional audits, and fresh independent evaluation would be needed to investigate them.''')
code('''audit_fixed = audit_new_comparison.loc[winning_name]
audit_total_gap = float(audit_fixed.accuracy-final_metrics['accuracy'])
audit_configuration_gap = float(audit_fixed.accuracy-audit_selected.mean_test_accuracy)
audit_evaluation_gap = float(audit_selected.mean_test_accuracy-final_metrics['accuracy'])
assert np.isclose(audit_total_gap,audit_configuration_gap+audit_evaluation_gap)
audit_summary = {
    'prior_metrics_reproduced':True,'preprocessing_statistics_verified':True,
    'fixed_model_cv_accuracy':float(audit_fixed.accuracy),
    'selected_model_cv':{m:float(audit_selected['mean_test_'+m]) for m in scoring},
    'selected_model_test':final_metrics,'accuracy_gap_percentage_points':{
        'total':100*audit_total_gap,'configuration_and_objective':100*audit_configuration_gap,
        'selected_model_cv_minus_test':100*audit_evaluation_gap},
    'heldout_rows':len(y_test),'permutation_repeats':audit_repeats,
    'permutation_f1_top_five':audit_importance.head(5).feature.tolist(),
    'aggregated_impurity_top_five':audit_importance.sort_values('impurity_importance',ascending=False).head(5).feature.tolist(),
    'permutation_impurity_rank_spearman':audit_spearman,
    'shared_full_predictor_profiles':audit_profile_overlap,'thresholds':audit_thresholds,
    'frozen_decision_rule':'probability > 0.50; class 0 on exact ties',
    'subgroup_gaps':audit_gaps.reset_index().to_dict('records'),
    'model_changed_by_review':False,'diagnostics_are_not_selection':True}
(OUT/'scientific_review.json').write_text(json.dumps(audit_summary,indent=2))
audit_low = audit_threshold_table.iloc[0]; audit_high = audit_threshold_table.iloc[-1]
display(Markdown(f"""### Review findings from actual results
- Previously reported CV and test scores reproduced; preprocessing development statistics and fold means verified.
- Accuracy difference: **{100*audit_total_gap:.3f} points** = **{100*audit_configuration_gap:.3f} points** between model configurations/CV objectives + **{100*audit_evaluation_gap:.3f} points** between the selected model's own CV accuracy and test accuracy. This arithmetic is not a causal decomposition.
- Held-out F1 permutation top five: **{', '.join(audit_importance.head(5).feature)}**. Aggregated impurity top five: **{', '.join(audit_summary['aggregated_impurity_top_five'])}**. Rank correlation: **{audit_spearman:.3f}**. See the coherent age/hours sensitivity results before interpreting the interaction ranking.
- At threshold {audit_low.threshold:.2f}, precision/recall are **{audit_low.precision:.4f}/{audit_low.recall:.4f}**; at {audit_high.threshold:.2f}, **{audit_high.precision:.4f}/{audit_high.recall:.4f}**. All predeclared thresholds are reported; **0.50 remains frozen**.
- Sex-group TPR/FPR max–min gaps: **{audit_gaps.loc['sex','recall_tpr']:.4f}/{audit_gaps.loc['sex','fpr']:.4f}**; race-group gaps: **{audit_gaps.loc['race','recall_tpr']:.4f}/{audit_gaps.loc['race','fpr']:.4f}**. These descriptive gaps must be read with the subgroup supports and Wilson intervals, not as a fairness verdict.
- Shared complete predictor profiles across development/test: **{audit_profile_overlap}**. Absence of shared profiles would still not prove person-level independence.

The model, hyperparameters and decision threshold were not changed. These test diagnostics are now visible and must not guide further selection without independent evaluation data."""))
''')
nb.validate(n); nb.write(n,p)
print('Preserved original training code; appended review. Total code cells:',sum(c.cell_type=='code' for c in n.cells))
