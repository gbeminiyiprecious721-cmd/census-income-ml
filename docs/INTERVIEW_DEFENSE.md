# Defending CensusIncome ML in an interview

## Your opening explanation

“This project studies how feature design and F1-based Random Forest selection change
precision and recall on historical Census data. I built a shared preprocessing and
inference pipeline, an executed research notebook and an integrity-checked six-page
dashboard. The saved model detects 81.38% of positive records at 61.26% precision.
I can trace those figures to confusion counts and explain the dataset, validation,
calibration and demographic limitations. It is a research demonstration, not a
present-day income decision system.”

Use “I” for work you personally understand and can demonstrate. Describe coding-agent
assistance honestly if asked; do not imply independent manual authorship of generated
code. Review the actual modules and rerun the checks yourself before an interview.

## Questions you should be able to answer

**Why not just report accuracy?** The negative class is about 76% of the sample.
A constant-negative classifier scores 76.06% on the test set with zero recall for
>$50K. The selected model finds 1,901 of 2,336 positives, while generating 1,202 false
positives. Precision, recall, F1 and ROC-AUC expose different aspects of that trade-off.

**Why F1, and is it a business-optimal objective?** F1 makes positive-class precision
and recall central in this research question. It is not a derived monetary utility
or proof of the right objective for a real service. A real use case needs agreed
harms, costs, capacity and validation before selecting an objective or threshold.

**How did you prevent preprocessing leakage?** The pipeline performs deterministic
record-local feature engineering, then fits imputation and one-hot encoding only
on each CV training fold. Test records never fit those components or the forest.
See `preprocessing.make_pipeline` and the fold-imputer regression test.

**Did you use the official UCI test partition?** No. To preserve the reviewed study,
the two files are pooled,52 exact full-record duplicates removed, then split with
seed 42 and stratification. That is disclosed and prevents direct official-split
benchmark comparisons. Two complete original-predictor profiles still cross the
split with conflicting labels. There is no person ID to prove independence.

**Is your selected CV score unbiased?** No. Feature-group selection and the
12-candidate search reuse five folds. That creates selection optimism. Nested CV
or a separate development validation design would improve that estimate. The test
set was not used for selection but has since been inspected; reruns are reproduction.

**Why does accuracy fall after tuning?** The selected class-weighted, deeper forest
has greater recall and F1 but lower precision/accuracy than the fixed forest. The
fixed-vs-selected CV difference and selected-CV-vs-test difference are different
comparisons. The evidence shows an observed trade-off, not a causal isolation of
which parameter caused it or proof that the selected model is universally better.

**Why not choose 0.65 after seeing the threshold chart?** It has higher observed F1
on this test set, but selecting it there would reuse evaluation data for policy
selection. The saved 0.50 strict rule stays fixed. Production threshold selection
would occur on development data with explicit objectives and later validation.

**Are your outputs probabilities?** They are uncalibrated positive-class scores.
ROC-AUC .9142 describes ranking, not calibration. Brier .1113 and ten-bin ECE .0831
are diagnostics; no calibrator was fitted. I would not describe a 0.7 score as a
70% reliable probability of a person's current income.

**What does feature importance mean?** Permutation importance measures loss of
model score under perturbation. Impurity credit uses split-based training evidence
and has different biases. Correlation and derived inputs complicate both. I added
coherent source perturbations for age/hours so their interaction is recomputed.
No ranking establishes causality, and global importance is not a local explanation.

**Is the model fair because sex and race are excluded?** No. Other fields can be
proxies. Female-coded records have FNR 32.90%, versus 15.82% for Male-coded records,
with different actual-positive denominators. FPRs also differ. These observations
require uncertainty, context and stakeholder-defined criteria. I added descriptive
sex×race diagnostics; small denominators prevent fairness certification.

**Why didn't you “fix bias” by changing predictions?** Mitigation requires a defined
use case, harms and fairness objective, a development-only experiment and independent
evaluation. Choosing adjustments from the inspected test rates would compromise the
study. This project makes the disparities visible and does not claim consequential
suitability or discrimination causality.

**What does reproducibility mean here?** Loading the unchanged saved artifact
reproduces its published metrics exactly. A fresh Mac refit chooses the same group
and parameters but changes 11 test labels, net three additional correct. Same-Mac
repeat fits have identical classes. I retain the discrepancy, keep strict comparison
as default, and run an explicit isolated record-differences verification. I do not
claim cross-platform bitwise training equivalence; the precise low-level cause was
not isolated. Neither refit replaces the frozen dashboard model automatically.

**How does the app avoid retraining and mixed evidence?** The notebook/package owns
training. The app loads a validated immutable bundle containing matching reports,
model, manifest and source. Publication is local and atomic; rejected updates retain
the last valid snapshot. Cached scores depend on snapshot identity. Shared controls
recompute scenarios without changing original evidence. Page navigation preserves
context; Explorer has its separate source-cohort state.

**What engineering bug did your audit catch?** Training allowed research feature
groups needing sex/race, while the final inference contract intentionally excludes
those inputs. The exporter now rejects incompatible or single-class artifacts
before touching a trusted release. Tests verify rejection and supported-model
round-trip parity. A real-browser reset→refresh test also caught stale threshold
state that AppTest alone missed.

**How are model files secured?** Only maintainer-created joblib files are accepted,
with fixed paths, hashes, schema and dependency checks. Joblib can execute code;
checksums do not authenticate a model if an attacker replaces both it and its
manifest. There is no untrusted-model upload feature or production security claim.

**What would you improve next?** A profile-grouped sensitivity experiment, nested
or otherwise independent development validation, a wider justified model comparison,
calibration fitted on development data, and stakeholder-informed fairness evaluation.
These would be new experiments with separately labelled results. Deployment needs
its own hosting, security and resource verification.

## A five-minute demonstration

1. Open Overview, quote the confusion counts and compare with the majority baseline.
2. Change threshold and population; explain why CV/importance stay explicitly global.
3. Show ROC/PR/calibration and error records on Performance; distinguish ranking from
   operating point and calibration.
4. Show subgroup denominators and intervals; explain why there is no fairness score.
5. Submit a hypothetical record, show its saved inputs and JSON export, then explain
   the score's historical/uncalibrated limits.
6. Open the notebook and current audit evidence; explain frozen-model parity versus
   the fresh refit difference without concealing either.

## Before an actual interview

Run the documented commands yourself. Be able to explain every model input, the two
engineered fields, a confusion-matrix example, how cross-validation fits preprocessing,
and one failed audit check. If you cannot yet explain a function or result, rehearse
it rather than memorising the headline metrics. The project is defensible evidence;
it cannot substitute for your own understanding.
