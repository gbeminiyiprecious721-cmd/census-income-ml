# Model card

## Purpose and scope

Random Forest binary classification of the historical Adult income label: `1` means `>50K`; `0` means `<=50K`. Intended for research, engineering demonstration and educational exploration. The model is unsuitable for current income estimation or consequential decisions about people.

## Model and inputs

The fitted scikit-learn pipeline includes deterministic education bins and age × weekly-hours interaction, imputation, categorical encoding and a Random Forest. Its raw input contract comprises age, capital gain, capital loss, weekly hours, numeric education, workclass, occupation, marital status, relationship and native country. Race and sex are retained for evaluation rather than supplied as prediction inputs; excluding them does not eliminate demographic proxies or discrimination.

The historical selected configuration used 100 trees, maximum depth 20, all features available per split, minimum split size 10, minimum leaf size 1 and `balanced_subsample` class weights. Fresh selected parameters, dependency versions and model identity must be read from the current report and model manifest.

## Training and evaluation

The historical workflow pooled UCI files, removed 52 exact full-record duplicates, and split 48,790 records into 39,032 development and 9,758 test rows with seeded stratification. Preprocessing statistics are fitted within development folds. Four feature groups and a 12-candidate F1 search are compared with five-fold CV.

Historical selected-model results: CV F1 0.7004; test accuracy 0.8322, precision 0.6126, recall 0.8138, F1 0.6990 and ROC-AUC 0.9142. Sources: `artifacts/historical/results.json` and `scientific_review.json`. The clean-kernel run `20261022T103759Z` reproduced these metrics; actual fresh values and parameter reconciliation are recorded in `reports/summary.json`. All 14 code cells completed without errors. This is reproduction on the previously inspected test split, not fresh independent confirmation.

Predictions preserve the original argmax convention: score strictly greater than 0.50 yields class 1; ties yield class 0. Fixed alternative thresholds are descriptive diagnostics, not test-selected deployment policies. Model scores are uncalibrated, even when a calibration diagnostic is available.

## Interpretation and responsible evaluation

Permutation importance measures score changes when inputs are perturbed; coherent source-column perturbations preserve derived-feature dependencies. Impurity importance can favour certain feature types. Correlated predictors can redistribute rankings. Neither method establishes causality or justifies individual decisions.

Subgroup reports include sample counts, confusion counts, positive and negative denominators, rates and Wilson uncertainty intervals. Race and sex disparities are descriptive, with limited subgroup sizes and incomplete intersectional coverage. These results do not certify fairness, compliance or suitability for deployment.

## Limitations

The data reflect a selected historical 1994 US population. The task omits contemporary economic context, uses a fixed nominal income boundary, and does not establish representativeness. The search is limited and model-selection folds are reused. Two shared predictor profiles across the development/test split have conflicting labels. Test diagnostics have already been inspected, so another run is not a fresh independent estimate. External validation, calibration, more robust selection and a justified contemporary use case would be required before considering operational use.

## Artifact handling

`artifacts/model.joblib` and `artifacts/model_manifest.json` are a coupled release unit. Record the model version, creation/run identifier, source-data evidence, checksum and dependency versions. Only trusted maintainer-created artifacts may be deserialised. Fixed-path and checksum/version checks catch accidental corruption and incompatible environments; they do not establish authenticity if both files are maliciously replaced. The app must show an actionable error when artifacts are absent or invalid and must not retrain automatically.
