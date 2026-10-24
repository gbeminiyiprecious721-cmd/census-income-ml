# Research report: historical income classification

## Scope and finding

The research question is how feature design and F1-based Random Forest selection
change positive-class precision and recall on UCI Adult. The deliverable connects a
reusable pipeline, executed notebook, inspectable diagnostics, integrity-checked
model artifact and interactive dashboard. It supports methodological discussion,
not decisions about a person's present income.

The saved model has accuracy 83.2240%, precision 61.2633%, recall 81.3784%,
F1 .6990256 and ROC-AUC .9141796 on 9,758 historical test observations.
Its confusion counts are TN 6,220, FP 1,202, FN 435, TP 1,901. A constant-negative
baseline achieves 76.06% accuracy but finds no positives. The forest's additional
accuracy therefore has context; its false alarms and missed positives remain material.

## Data and population

The source is Becker & Kohavi's UCI Adult dataset (1996), extracted from the 1994
Census database. The 48,842 source rows contain 14 predictors and an income label.
Two original files are checksummed, pooled, whitespace/target suffixes normalised,
question marks converted to missing, and 52 exact full-record duplicates removed.
The 48,790 cleaned rows are split with seed 42 and target stratification into 39,032
development and 9,758 test rows. This is not the original UCI partition.

The audit finds no row-index or full-record overlap. Two complete 14-predictor
profiles cross the split with conflicting labels. This does not establish person
identity. Reduced selected-input profiles can match much more often after dropping
fields; that is a different question from complete-record duplication. Metrics are
unweighted observations, not population estimates; `fnlwgt` is not used as a weight.

## Pipeline and selection

Record-local education bins and age×weekly hours are inside a scikit-learn pipeline.
Median numeric imputation and most-frequent categorical imputation plus one-hot
encoding are fitted within each training fold. The encoder handles unseen levels;
the inference interface surfaces warnings. Education bins are predeclared, not
learned using held-out outcomes. No test row fits an imputer, encoder or forest.

Four fixed feature groups use 100-tree, depth 16 forests and the same five stratified
folds. Feature groups change several inputs together, so the comparison cannot
isolate a causal interaction benefit. The best group by positive-class F1 is
Expanded + interaction. A 12-candidate seeded RandomizedSearchCV refits on mean F1.
It selects 100 trees, depth 20, max_features 1.0, min_samples_split 10,
min_samples_leaf 1 and balanced_subsample weighting. The same five folds are reused
for group selection and tuning: selected CV is not an unbiased nested-CV estimate.

Compared with the fixed forest's 86.05% CV accuracy and .6760 F1, the selected
configuration has 83.40% CV accuracy and .7004 F1. Higher recall accompanies lower
precision/accuracy. Its test accuracy differs from its own CV by 0.1794 percentage
points. The larger 2.6491-point difference between the two CV configurations must
not be described as a CV-to-test generalisation gap or a proven causal attribution.

## Thresholds and calibration

The saved rule is score strictly greater than 0.50; exact ties map to class 0.
At 0.50 precision is 61.26% and recall 81.38%. The predeclared 0.80 diagnostic gives
82.25% precision and 54.97% recall, with 1,052 missed positives versus 435 at 0.50.
At 0.65 F1 happens to be higher on this inspected test set; it is not chosen as a
policy. A deployable threshold would require development-only selection, explicit
error costs/capacity and further validation.

Saved test Brier score is .111321, log loss .358382 and ten-bin ECE .083099.
These are descriptive diagnostics; ECE depends on binning and is not universal.
No calibrator was trained. A forest's positive score is not automatically a reliable
individual probability of contemporary income. ROC-AUC describes ranking across
thresholds and does not establish calibration or a suitable operating point.

## Explainability

Held-out permutation importance measures score decrease under input perturbation;
shuffle SD describes variation across perturbations, not a sampling confidence
interval. Saved means/SDs reconcile with all repeat tables. Marital status, capital
gain and the engineered age×hours term are leading F1 permutation inputs. Impurity
credit is separately labelled because it has a different mechanism and potential
feature-type biases. Correlation and derived dependencies complicate rankings.

A coherent source perturbation shuffles age or hours and recomputes their interaction.
This complements the transformed-column diagnostic; the two are not interchangeable.
Neither is a causal effect or a validated local explanation. No SHAP values or
individual contribution narrative is fabricated.

## Bias, uncertainty and responsible evaluation

At the fixed 0.50 rule, Female-coded records have FPR 4.35% (2,943 actual negatives)
and FNR 32.90% (383 actual positives). Male-coded records have FPR 23.98% (4,479
negatives) and FNR 15.82% (1,953 positives). These are substantial observed error
patterns. Sex/race exclusion from final inputs does not eliminate household,
occupation or other proxies.

Race group sizes range from 84 to 8,353. Some positive denominators are only 10 or 13,
so isolated percentage rankings can mislead. The saved subgroup reports include
Wilson 95% intervals for relevant rates and visible denominators. The current audit
adds sex×race diagnostics, also with denominators and marginal intervals. They are
exploratory; no multiple-comparison correction or causal discrimination conclusion
is claimed. There is no universal fairness score or fairness certification.

Mitigation is future research: define the use case and harms, consult affected
stakeholders, select a defensible fairness criterion and evaluate interventions
using development data and independent validation. Retrofitting a policy to the
already inspected test group rates would not repair the evidence.

## Reproducibility audit —2026-10-24

Two distinct questions were tested. Loading the unchanged trusted artifact exactly
reproduces its saved metrics, calibration and subgroup values. Repeating training on
the audited Mac selects the same group/parameters but produces slightly different
numbers from the archived Windows run. Eleven test labels change, net three more
correct: accuracy 83.2548%, F1 .6994113, ROC-AUC .9141468. Same-Mac one-thread and
four-thread refits agree in class predictions; maximum score difference is 3.33e-16.
The exact low-level source of cross-environment differences was not isolated.

Strict reconciliation rejected the mismatch. An explicit isolated record-differences
run then completed 14 code cells sequentially, zero errors, eight figures, in 197.53s.
It includes 20 fixed-group forest fits,60 search-fold fits and a final refit, plus the
baseline and diagnostics. Its outputs are separate; the dashboard model, reports
and threshold are unchanged. This establishes executable workflow verification,
not bitwise cross-platform training equivalence or new independent predictive validity.

[Execution and reconciliation evidence](portfolio_audit/reproduction/verification.json)
· [fresh executed notebook](portfolio_audit/reproduction/notebook.ipynb)
· [frozen-artifact verification](portfolio_audit/frozen/verification.json).

## Primary methodological references

- [UCI Adult provenance and license](https://archive.ics.uci.edu/dataset/2/adult).
- [scikit-learn: leakage and consistent preprocessing](https://scikit-learn.org/stable/common_pitfalls.html).

The numerical findings above come from this project's inspectable reports and fresh
audit evidence, not from those general methodological references.
