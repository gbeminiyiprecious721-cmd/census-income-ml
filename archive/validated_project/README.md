# Adult Income Random Forest

An executed portfolio notebook predicting income above $50,000 using the real UCI Adult Census dataset. The original notebook is preserved in `Adult_Income_Random_Forest_End_to_End.original.ipynb`.

## Executed results
Default original validation accuracy: **81.78%**; depth-tuned original: **83.20%**; depth-tuned education: **84.19%**.
Selected feature group: **Expanded + interaction**; best searched five-fold F1: **0.7004**.
Final held-out test: accuracy **83.22%**, positive-class F1 **0.6990**, ROC-AUC **0.9142**.
All **14 code cells** executed sequentially in a fresh kernel with zero error outputs. Validation scores are model-selection evidence, not test estimates.

At fixed depth 16 under identical five-fold validation, F1 increased from 0.5031 (original) to 0.5787 (education), 0.6689 (expanded), and 0.6760 (expanded plus interaction). The interaction's observed incremental gain is small and has not been tested for statistical significance.

Tuning trades accuracy for positive-class recovery: under the same folds, the selected candidate has accuracy **83.40%**, recall **81.03%** and F1 **0.7004**, compared with **86.05%**, **60.78%** and **0.6760** for its fixed-parameter feature-comparison model. This is consistent with the stated F1 objective.

## Setup and execution

Python 3.12 was used. Select `.venv312/Scripts/python.exe` in VS Code's notebook kernel picker. The notebook metadata names the project-local `adult-income` kernel; the IDE's previously active runtime was not observable from the supplied file.

Create an environment and install the pinned packages:

```powershell
py -3.12 -m venv .venv312
.\.venv312\Scripts\python.exe -m pip install -r requirements.txt
.\.venv312\Scripts\python.exe -m ipykernel install --prefix .jupyter --name adult-income --display-name "Python (Adult Income .venv312)"
.\.venv312\Scripts\python.exe execute_notebook.py
```

The project-local kernelspec is included for this workspace; regenerate it if the project moves. `execute_notebook.py` sets local Jupyter paths, starts a fresh kernel, executes every cell, saves outputs, validates the notebook and exports HTML. Windows may require permission to set Jupyter connection-file ACLs. The first dependency install and any uncached source download need internet access. `requirements.txt` pins the executed environment. The failed initial `.venv` setup is unused; `.venv312` is the working environment.

`build_notebook.py` is the authoring scaffold; running it intentionally reconstructs the notebook and clears outputs. Normal users should run the notebook or execution script, not rebuild it. `document_results.py` adds the verified result summary and creates this README after execution.

## Dataset and provenance

Becker, B. & Kohavi, R. (1996). Adult. UCI Machine Learning Repository. DOI: [10.24432/C5XW20](https://doi.org/10.24432/C5XW20), CC BY 4.0. Source records derive from the 1994 Census database. The downloaded `adult.data` and `adult.test` contain 48,842 records and 14 predictors plus the target. Raw files are cached under `data/`; SHA-256 values and source URLs are in `artifacts/provenance.json`.

Whitespace and target periods are removed, `?` becomes missing, and 52 exact full-record duplicates are removed before splitting. Missing predictors are imputed inside training folds. The notebook reports schema, category inventories, missingness, target counts and class proportions. Dataset category spellings are retained rather than silently renaming historical labels.

## Methodology

The pooled source data use a seeded stratified 80/20 development/test split (39,032/9,758). The tutorial further splits development data 75/25, giving approximately 60/20/20 overall. No final test outcomes select models. All forests and splitters use seed 42. One-hot encoding ignores unseen categories; imputation is fitted only within training partitions. `fnlwgt` is excluded and metrics are unweighted per observation.

The 17 tutorial requirements are covered in order: income distribution; whitespace cleaning; categorical/numeric preparation; binary target; split; default forest; depths 1–25; best original depth; train/validation curves; feature importance; education bins; updated feature list; repeated depth search; performance comparison; updated curves; importance comparison; computed conclusions. The original supplied six-feature definition is preserved. Unlike the original exploratory approach, depth tuning uses validation data.

Four feature sets are compared under identical five-fold stratified CV with a fixed forest: original six, plus education, expanded census, expanded plus age–hours interaction. This ablation measures whether the interaction helps. Expanded predictors include occupation, marital status, relationship, workclass, native country and education groups. Sex/race are omitted in expanded variants, so their comparison combines predictor additions and removals and cannot isolate individual effects.

RandomizedSearchCV samples 12 configurations (60 fold fits), tunes all six requested forest hyperparameters and refits on mean positive-class F1. Accuracy, precision, recall and ROC-AUC are also recorded. Search runs sequentially, each forest using at most four threads. The threshold remains 0.5. Feature-set selection and tuning reuse development folds; scores are not nested-CV estimates.

## Actual findings

Education changes best tutorial validation accuracy by +0.984 percentage points. See `artifacts/feature_comparison.csv` and `feature_comparison_folds.csv` for consistent-CV evidence, including expanded predictors and interaction ablation. These are observed differences, without a significance claim.

Best original depth: 13; best education depth: 17.

Best searched hyperparameters:

```json
{
  "forest__n_estimators": 100,
  "forest__min_samples_split": 10,
  "forest__min_samples_leaf": 1,
  "forest__max_features": 1.0,
  "forest__max_depth": 20,
  "forest__class_weight": "balanced_subsample"
}
```

Final test precision: 0.6126; recall: 0.8138. Top five encoded impurity features:

- categorical__marital-status_Married-civ-spouse: 0.3981
- numeric__capital-gain: 0.1259
- numeric__hours_age_interaction: 0.1195
- categorical__education_bin_HS or less: 0.0766
- numeric__age: 0.0581

## Validation and artifacts

The clean-kernel run executed 14 code cells with zero errors in 1683.2 seconds. Assertions check disjoint indices and full-record hashes, target encoding, independent confusion-matrix metric arithmetic, valid probabilities and prediction with missing/unseen values. The saved notebook retains figures, tables, classification report and conclusions; the HTML provides a readable preview.

- `artifacts/results.json`: actual metrics, parameters, environment and split sizes.
- `artifacts/execution_validation.json`: clean-kernel cell counts and timing.
- `artifacts/random_search.csv`: every candidate, fold scores and timings.
- `artifacts/feature_comparison*.csv`: feature-group comparison.
- `artifacts/feature_importance.csv`: encoded impurity rankings.
- `artifacts/*.png`: income distribution, depth curves, importance comparisons, CV/search curves and final confusion/ROC/precision–recall panels.
- `execution.log`: execution trace; platform warnings are distinct from cell errors.

## Limitations and recommendations

Historical data and a pooled random split do not establish contemporary generalisation. The original UCI source split is not used as the final split. Exact deduplication is conservative; there is no person identifier to rule out all related individuals. Accuracy reflects class imbalance; prioritise positive-class F1, precision and recall alongside ROC/PR curves. The finite search does not prove global optimality, and reused selection folds can inflate CV scores. No uncertainty interval or statistical-significance claim is made.

Impurity importance favours some continuous/high-cardinality variables and splits credit across correlated predictors. It is not causal. Sex/race and their proxies encode historical inequalities; omitting direct sensitive features does not guarantee fairness. This project is educational, with no claim of deployment readiness, fairness or calibrated probabilities. Further work should examine original-source partition performance, subgroup errors with uncertainty, repeated/nested CV, calibration and simpler baselines using an appropriate new evaluation set if decisions change.

## Scientific review — diagnostic audit

The 14 code cells executed sequentially in a fresh kernel with zero errors; all nine original code cells are unchanged (SHA-256 check). Accuracy, precision, recall and F1 match exactly for all 12 search candidates. Maximum mean ROC-AUC variation is **0.0000001442**; all five mean metrics agree at four decimals. Such a small variation is compatible with floating-point/tie-order effects in parallel forests, but its cause was not isolated. Fold means and confusion-matrix arithmetic reconcile; test ROC-AUC independently matches the Mann–Whitney rank calculation. Backups: `Adult_Income_Random_Forest_End_to_End.pre_review.ipynb`, `README.pre_review.md` and `artifacts/pre_review/`.

### Why 86.05% CV accuracy differs from 83.22% test accuracy

| Model / evaluation | Accuracy | Positive F1 | Recall |
| --- | --- | --- | --- |
| Fixed forest / five-fold validation | 86.05% | 0.6760 | 60.78% |
| F1-selected forest / same five folds | 83.40% | 0.7004 | 81.03% |
| Selected refit / held-out test | 83.22% | 0.6990 | 81.38% |

The fixed feature-comparison forest has 100 trees, depth 16, `max_features='sqrt'`, split size 2, leaf size 1 and no class weights. The selected forest has 100 trees, depth 20, `max_features=1.0`, split size 10, leaf size 1 and `class_weight='balanced_subsample'`. Search refits on **F1**, not accuracy, and its observed improvement in recall comes with lower precision/accuracy. The selected CV estimate averages five fits on 80% of development data; the final model is refit on all development records and evaluated on a separate test sample. Test precision is 0.6126, recall 0.8138, ROC-AUC 0.9142.

The total **2.828-percentage-point** difference separates arithmetically into **2.649 points** between CV configurations/objectives and **0.179 points** between the selected model's own CV and test accuracy. This is not a causal decomposition or evidence that the whole difference is overfitting. Sampling variation, training-size differences and selection optimism are plausible contributors to the remaining gap, not established causes.

### Leakage assessment

No test-to-fit path was found: learned preprocessing is inside CV pipelines; engineering is record-local; model selection uses development data; fitted imputer statistics match development records. Sex/race are joined by index only for audit. However, **2 complete 14-predictor profiles cross the split with conflicting target labels**: deduplication includes income, so these survive. This is a confirmed data-quality/independence concern, not proof of duplicate people or target leakage. Do not claim complete record independence. A predictor-profile-grouped split sensitivity analysis and person-level provenance would be appropriate future checks; the current split/model are preserved.

### Held-out permutation importance

Ten seeded permutations per pipeline input use all 9,758 test observations. Encoded impurity contributions are aggregated to the same raw-column level. Their scales are different: impurity is normalised training split credit; permutation is a held-out score decrease. Spearman rank correlation for impurity versus F1 permutation is **0.973**.

| Input feature | Impurity rank | F1 permutation rank | Mean F1 decrease | Shuffle SD |
| --- | --- | --- | --- | --- |
| marital-status | 1 | 1 | 0.18285 | 0.00816 |
| capital-gain | 2 | 2 | 0.05845 | 0.00271 |
| hours_age_interaction | 3 | 3 | 0.04399 | 0.00472 |
| occupation | 5 | 4 | 0.03642 | 0.00370 |
| education_bin | 4 | 5 | 0.03409 | 0.00246 |

Aggregated impurity top five: marital-status, capital-gain, hours_age_interaction, education_bin, occupation. Full F1/ROC-AUC rankings and repeat-level scores are saved. Shuffle SD is not a sampling confidence interval. Single-column permutation breaks correlation and, for age/hours/interaction, the deterministic interaction identity. The supplementary source permutation recomputes the interaction; its mean F1 decreases are age: 0.05244, hours-per-week: 0.02540. These measure joint-source sensitivity, not a directly interchangeable single-column ranking. Neither method is causal. See [scikit-learn's permutation-importance guidance](https://scikit-learn.org/stable/modules/permutation_importance.html).

### Fixed-threshold diagnostics

These thresholds were specified before inspecting their metrics; none is selected from test performance. Predictions use `probability > threshold`, preserving the classifier's class-0 tie rule. The model remains at **0.50**.

| Threshold | Precision | Recall | F1 | Selected positive |
| --- | --- | --- | --- | --- |
| 0.20 | 0.4754 | 0.9336 | 0.6300 | 47.02% |
| 0.35 | 0.5487 | 0.8797 | 0.6759 | 38.38% |
| 0.50 | 0.6126 | 0.8138 | 0.6990 | 31.80% |
| 0.65 | 0.7021 | 0.7042 | 0.7031 | 24.01% |
| 0.80 | 0.8225 | 0.5497 | 0.6590 | 16.00% |

Scores are not assumed calibrated after class weighting. Future threshold decisions require development-only/OOF evidence and explicit error costs, followed by independent evaluation; see [scikit-learn's threshold guidance](https://scikit-learn.org/stable/modules/classification_threshold.html).

### Sex and race disparities at the frozen threshold

Labels reproduce the dataset's coarse historical sex/race coding and do not represent all identities. Intervals below are marginal Wilson 95% intervals, conditional on independent records; they are not simultaneous multiple-comparison intervals. Positive and negative denominators matter, especially for small groups.

| Attribute / group | n | Positives | Negatives | Precision | TPR [95% interval] | FPR [95% interval] |
| --- | --- | --- | --- | --- | --- | --- |
| sex: Female | 3326 | 383 | 2943 | 0.668 | 0.671 [0.622, 0.716] | 0.043 [0.037, 0.051] |
| sex: Male | 6432 | 1953 | 4479 | 0.605 | 0.842 [0.825, 0.857] | 0.240 [0.228, 0.253] |
| race: Amer-Indian-Eskimo | 94 | 13 | 81 | 0.379 | 0.846 [0.578, 0.957] | 0.222 [0.145, 0.324] |
| race: Asian-Pac-Islander | 311 | 71 | 240 | 0.543 | 0.803 [0.696, 0.879] | 0.200 [0.154, 0.255] |
| race: Black | 916 | 113 | 803 | 0.555 | 0.761 [0.675, 0.830] | 0.086 [0.068, 0.107] |
| race: Other | 84 | 10 | 74 | 0.545 | 0.600 [0.313, 0.832] | 0.068 [0.029, 0.149] |
| race: White | 8353 | 2129 | 6224 | 0.621 | 0.818 [0.801, 0.834] | 0.171 [0.161, 0.180] |

Full confusion counts, prevalence, selection rates, accuracy, precision, recall, FPR/FNR, F1 and ROC-AUC are saved in `artifacts/review_subgroup_metrics.csv`. Group totals reconcile to overall test confusion counts. These are descriptive error disparities: they do not establish discrimination, explain causes, or demonstrate fairness. Different group compositions and base rates affect precision/selection rates; separate sex/race summaries omit intersectional harms. Removing sensitive predictors does not remove proxies. No group-specific thresholds or interventions were chosen.

### Remaining weaknesses and interpretation

**Confirmed design limitations:** one historical pooled random split; no person identifier or contemporary external evaluation; reused folds for feature selection and 12-trial search; no nested CV; feature groups selected before tuning; simultaneous additions/removal of predictors confound isolated feature-effect claims; uncalibrated scores; correlated/derived-feature interpretation limits; small subgroup denominators; no simultaneous inference or full intersectional audit.

**Unverified hypotheses:** sampling variation or selection optimism explains the small selected-model CV/test gap; the interaction's gain generalises; proxies or group composition explain observed disparities. Further development-only experiments and new independent evaluation are needed. The test set has now been inspected extensively and cannot remain an untouched basis for later changes informed by this review. No model, hyperparameter or threshold was changed, and no deployment or production/fairness readiness is claimed.

Review tables and figures are under `artifacts/review_*`; the machine-readable assessment is `artifacts/scientific_review.json`. Re-execute with `execute_notebook.py`; then run `document_results.py`, `verify_review_artifacts.py`, and `document_scientific_review.py` in that order to refresh summaries and independent artifact checks. The original `build_notebook.py` is a reconstruction scaffold; do not run it for ordinary review execution because it clears outputs and omits this appended audit (reapply `append_scientific_review.py` if intentionally rebuilding).
