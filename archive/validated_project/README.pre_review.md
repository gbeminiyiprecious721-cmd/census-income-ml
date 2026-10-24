# Adult Income Random Forest

An executed portfolio notebook predicting income above $50,000 using the real UCI Adult Census dataset. The original notebook is preserved in `Adult_Income_Random_Forest_End_to_End.original.ipynb`.

## Executed results
Default original validation accuracy: **81.78%**; depth-tuned original: **83.20%**; depth-tuned education: **84.19%**.
Selected feature group: **Expanded + interaction**; best searched five-fold F1: **0.7004**.
Final held-out test: accuracy **83.22%**, positive-class F1 **0.6990**, ROC-AUC **0.9142**.
All **9 code cells** executed sequentially in a fresh kernel with zero error outputs. Validation scores are model-selection evidence, not test estimates.

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

The clean-kernel run executed 9 code cells with zero errors in 1654.5 seconds. Assertions check disjoint indices and full-record hashes, target encoding, independent confusion-matrix metric arithmetic, valid probabilities and prediction with missing/unseen values. The saved notebook retains figures, tables, classification report and conclusions; the HTML provides a readable preview.

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
