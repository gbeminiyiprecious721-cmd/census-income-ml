from pathlib import Path
import json
import nbformat
import pandas as pd

root = Path(__file__).resolve().parent
r = json.loads((root/'artifacts/results.json').read_text())
v = json.loads((root/'artifacts/execution_validation.json').read_text())
m = r['test_metrics']
comparison = pd.read_csv(root/'artifacts/feature_comparison.csv').set_index('model')
search_table = pd.read_csv(root/'artifacts/random_search.csv')
selected = search_table.loc[search_table.mean_test_f1.idxmax()]
engineering_evidence = (f"At fixed depth 16 under identical five-fold validation, F1 increased from "
    f"{comparison.loc['Original six','f1']:.4f} (original) to {comparison.loc['Plus education','f1']:.4f} "
    f"(education), {comparison.loc['Expanded census','f1']:.4f} (expanded), and "
    f"{comparison.loc['Expanded + interaction','f1']:.4f} (expanded plus interaction). "
    "The interaction's observed incremental gain is small and has not been tested for statistical significance.")
summary = f"""## Executed results
Default original validation accuracy: **{r['baseline_validation_accuracy']:.2%}**; depth-tuned original: **{r['best_original_validation_accuracy']:.2%}**; depth-tuned education: **{r['best_education_validation_accuracy']:.2%}**.
Selected feature group: **{r['selected_features']}**; best searched five-fold F1: **{r['best_cv_f1']:.4f}**.
Final held-out test: accuracy **{m['accuracy']:.2%}**, positive-class F1 **{m['f1']:.4f}**, ROC-AUC **{m['roc_auc']:.4f}**.
All **{v['code_cells']} code cells** executed sequentially in a fresh kernel with zero error outputs. Validation scores are model-selection evidence, not test estimates.

{engineering_evidence}

Tuning trades accuracy for positive-class recovery: under the same folds, the selected candidate has accuracy **{selected.mean_test_accuracy:.2%}**, recall **{selected.mean_test_recall:.2%}** and F1 **{selected.mean_test_f1:.4f}**, compared with **{comparison.loc[r['selected_features'],'accuracy']:.2%}**, **{comparison.loc[r['selected_features'],'recall']:.2%}** and **{comparison.loc[r['selected_features'],'f1']:.4f}** for its fixed-parameter feature-comparison model. This is consistent with the stated F1 objective.
"""
p = root/'Adult_Income_Random_Forest_End_to_End.ipynb'
n = nbformat.read(p,4)
n.cells[0].source = n.cells[0].source.split('\n\n## Executed results')[0] + '\n\n' + summary
nbformat.validate(n); nbformat.write(n,p)
from nbconvert import HTMLExporter
html,_ = HTMLExporter().from_notebook_node(n)
(root/'Adult_Income_Random_Forest_End_to_End.html').write_text(html,encoding='utf-8')
readme = f"""# Adult Income Random Forest

An executed portfolio notebook predicting income above $50,000 using the real UCI Adult Census dataset. The original notebook is preserved in `Adult_Income_Random_Forest_End_to_End.original.ipynb`.

{summary}
## Setup and execution

Python 3.12 was used. Select `.venv312/Scripts/python.exe` in VS Code's notebook kernel picker. The notebook metadata names the project-local `adult-income` kernel; the IDE's previously active runtime was not observable from the supplied file.

Create an environment and install the pinned packages:

```powershell
py -3.12 -m venv .venv312
.\\.venv312\\Scripts\\python.exe -m pip install -r requirements.txt
.\\.venv312\\Scripts\\python.exe -m ipykernel install --prefix .jupyter --name adult-income --display-name "Python (Adult Income .venv312)"
.\\.venv312\\Scripts\\python.exe execute_notebook.py
```

The project-local kernelspec is included for this workspace; regenerate it if the project moves. `execute_notebook.py` sets local Jupyter paths, starts a fresh kernel, executes every cell, saves outputs, validates the notebook and exports HTML. Windows may require permission to set Jupyter connection-file ACLs. The first dependency install and any uncached source download need internet access. `requirements.txt` pins the executed environment. The failed initial `.venv` setup is unused; `.venv312` is the working environment.

`build_notebook.py` is the authoring scaffold; running it intentionally reconstructs the notebook and clears outputs. Normal users should run the notebook or execution script, not rebuild it. `document_results.py` adds the verified result summary and creates this README after execution.

## Dataset and provenance

Becker, B. & Kohavi, R. (1996). Adult. UCI Machine Learning Repository. DOI: [10.24432/C5XW20](https://doi.org/10.24432/C5XW20), CC BY 4.0. Source records derive from the 1994 Census database. The downloaded `adult.data` and `adult.test` contain 48,842 records and 14 predictors plus the target. Raw files are cached under `data/`; SHA-256 values and source URLs are in `artifacts/provenance.json`.

Whitespace and target periods are removed, `?` becomes missing, and {r['duplicates_removed']} exact full-record duplicates are removed before splitting. Missing predictors are imputed inside training folds. The notebook reports schema, category inventories, missingness, target counts and class proportions. Dataset category spellings are retained rather than silently renaming historical labels.

## Methodology

The pooled source data use a seeded stratified 80/20 development/test split ({r['split_sizes']['development']:,}/{r['split_sizes']['test']:,}). The tutorial further splits development data 75/25, giving approximately 60/20/20 overall. No final test outcomes select models. All forests and splitters use seed {r['seed']}. One-hot encoding ignores unseen categories; imputation is fitted only within training partitions. `fnlwgt` is excluded and metrics are unweighted per observation.

The 17 tutorial requirements are covered in order: income distribution; whitespace cleaning; categorical/numeric preparation; binary target; split; default forest; depths 1–25; best original depth; train/validation curves; feature importance; education bins; updated feature list; repeated depth search; performance comparison; updated curves; importance comparison; computed conclusions. The original supplied six-feature definition is preserved. Unlike the original exploratory approach, depth tuning uses validation data.

Four feature sets are compared under identical five-fold stratified CV with a fixed forest: original six, plus education, expanded census, expanded plus age–hours interaction. This ablation measures whether the interaction helps. Expanded predictors include occupation, marital status, relationship, workclass, native country and education groups. Sex/race are omitted in expanded variants, so their comparison combines predictor additions and removals and cannot isolate individual effects.

RandomizedSearchCV samples {r['search_iterations']} configurations (60 fold fits), tunes all six requested forest hyperparameters and refits on mean positive-class F1. Accuracy, precision, recall and ROC-AUC are also recorded. Search runs sequentially, each forest using at most four threads. The threshold remains 0.5. Feature-set selection and tuning reuse development folds; scores are not nested-CV estimates.

## Actual findings

Education changes best tutorial validation accuracy by {100*(r['best_education_validation_accuracy']-r['best_original_validation_accuracy']):+.3f} percentage points. See `artifacts/feature_comparison.csv` and `feature_comparison_folds.csv` for consistent-CV evidence, including expanded predictors and interaction ablation. These are observed differences, without a significance claim.

Best original depth: {r['original_best_depth']}; best education depth: {r['education_best_depth']}.

Best searched hyperparameters:

```json
{json.dumps(r['best_parameters'],indent=2)}
```

Final test precision: {m['precision']:.4f}; recall: {m['recall']:.4f}. Top five encoded impurity features:

"""
readme += '\n'.join(f"- {x['feature']}: {x['importance']:.4f}" for x in r['top_five_features'])
readme += f"""

## Validation and artifacts

The clean-kernel run executed {v['code_cells']} code cells with zero errors in {v['seconds']:.1f} seconds. Assertions check disjoint indices and full-record hashes, target encoding, independent confusion-matrix metric arithmetic, valid probabilities and prediction with missing/unseen values. The saved notebook retains figures, tables, classification report and conclusions; the HTML provides a readable preview.

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
"""
(root/'README.md').write_text(readme,encoding='utf-8')
print(summary)
