# Full scientific and portfolio audit —24 October 2026

## Verdict

**Ready for hiring-manager review as a historical ML research and engineering
portfolio project, subject to the explicitly documented boundaries below.** The
prepared source checkout runs independently, its scientific workflow has actually
been executed, and its conclusions distinguish saved evidence from fresh refits.
No claim of 100% certainty, production suitability, fairness certification, remote
CI success or contemporary income validity is made. No GitHub push was performed.

The most important audit finding was not a prettier metric: a fresh Mac refit does
not exactly reproduce archived Windows measurements. The discrepancy is retained,
strict promotion remains protected, and the dashboard's trusted original artifact
is unchanged. Concealing that result would make this project less defensible.

## Scores

The portfolio rubric scores visible evidence, not a prediction of hiring success.

| Dimension | Score | Evidence |
|---|---:|---|
| First impression | 5/5 | README opens with question/result and approved dashboard image |
| Problem framing | 3/5 | Explicit research decision about error trade-offs; no fabricated business ROI |
| Technical substance | 5/5 | Fold-local pipeline, search, calibration diagnostics, importance, subgroup uncertainty and independently checked arithmetic |
| Code quality and structure | 4/5 | Shared src package, CLI, contracts and snapshot tests; some inherited compact coding style remains |
| Reproducibility | 4/5 | Pinned fresh environment and standalone checkout verified; exact cross-environment refit equivalence is unclaimed |
| Evidence of outcome | 4/5 | Executed notebooks, figures, dashboard/screenshots and contextual metrics; no public live deployment |
| Communication | 4/5 | Research report, cards, explicit limitations and interview defense; no real-world intervention recommendation |

**Overall:29/35 —Flagship range under this rubric.** This is a portfolio/research
readiness verdict, not an assertion that every hiring manager will shortlist it.

## Do these first (high impact, low effort)

Completed during this audit:

1. **Replace stale onboarding and evidence** —README now starts the saved app without
   retraining, uses the current screenshot and distinguishes original 50-test evidence
   from current checks. This makes the first review accurate (~1 hour of typical work).
2. **Make the release self-contained** —project Git exclusions, Python version,
   MIT text/data NOTICE, selected release contents and scope-aware CI added. The
   environment and local secrets are omitted (~1 hour).
3. **Correct scientific prose** —public-record access is acknowledged; observed
   configuration trade-offs are not presented as isolated causal effects. Notebook
   code/output evidence remains historical and intact (~30 minutes).

## Do these next (high impact, higher effort)

Completed: actual full isolated notebook execution; row-level refit comparison;
export-contract fixes with failing/passing regressions; fresh-environment installation;
standalone tests and frozen-model audit. These are the engineering evidence behind
this verdict, not suggestions left for the owner to implement.

Future research—only with an appropriate new question—could add profile-grouped
sensitivity, independent/nested model selection, development-fitted calibration and
stakeholder-informed fairness mitigation. These are separately labelled studies,
not prerequisites to honestly presenting the completed historical project.

## Only if you have spare time (low impact)

Broader style normalization and an explanatory demo video could improve accessibility.
A public demo would be useful, but hosting has its own resource/security verification
and was not requested or performed. A configured workflow is not a remote green run.

## Don't bother

Do not add algorithm families merely to lengthen the tool list, bolt on an unrelated
chatbot, manufacture commit history, tune to the inspected test set, replace honest
bias findings with a fairness score, or suppress refit differences to claim certainty.

## Scope, controls and scientific integrity

The audit covered data acquisition/provenance, normalization/deduplication, target
encoding, split integrity, deterministic engineering, fold-only imputation/encoding,
all existing feature groups, the actual Random Forest search, selection objective,
held-out inference, confusion arithmetic, ROC-AUC, thresholds, calibration, global
importance, subgroup uncertainty, notebook execution, artifact integrity, dashboard
state, release packaging, dependencies and narrative consistency.

No new algorithm family was invented. The workflow contains the majority baseline,
four fixed-group forest comparisons and 12-candidate selection. The successful full
notebook performs 20 fixed-forest fold fits,60 search-fold fits and one final forest
refit, plus baseline evaluation and diagnostics. All scientific verification fits
are isolated; no refitted model is promoted to the approved dashboard.

## Actual verification results

| Check | Actual outcome | Evidence |
|---|---|---|
| Baseline full historical workspace suite |113 passed /1 known missing-root-file failure | [baseline log](baseline_pytest.log) |
| Export/reconciliation regressions |4 passed; new behavior first observed failing | [green](contracts_green.log), [red contract](contract_red.log), [red policy](reconciliation_red.log) |
| Current project suite |117 passed /1 external workspace check deselected,139.24s | [project log](release_pytest.log) |
| Standalone prepared checkout, fresh environment |117 passed /1 external check deselected,141.81s | [checkout log](staged_pytest.log) |
| Fresh pinned environment | Installed lock; dependency consistency passed | [install](locked_install.log), [check](dependency_check.log) |
| Frozen saved model | All 5 metrics,9,758 predictions, calibration/subgroups/thresholds reconcile | [machine-readable audit](frozen/verification.json), [standalone audit](staged_frozen.log) |
| Global permutation tables | Repeat means and population SD match saved summaries | [frozen audit](frozen/verification.json) |
| Full strict research rerun | Correctly rejected cross-environment numerical mismatch | [strict difference](strict_reproduction_difference.json), [log](reproduction.log) |
| Explicit isolated record-differences run |14 code cells, sequential counts,0 errors,8 figures,197.53s | [execution](reproduction/reports/execution_validation.json), [verification](reproduction/verification.json), [executed notebook](reproduction/notebook.ipynb) |
| Refit repeatability check | Same Mac 1/4-thread fits have identical classes; small floating summation differences | [row-level comparison](refit_comparison.json) |
| Dependency advisories | No known advisories among 90 published packages; local project is not a PyPI package | [advisory JSON](dependency_advisories.json) |
| Staged text/size scan | No matched credential patterns or files >50 MiB | [scan](release_scan.json) |

The standalone browser flow passed six page captures at 1440×900 and all
control/export/inference checks, with zero page errors; see
[the browser evidence](staged_browser/browser_qa.json). The previously approved three-viewport captures/automatic refresh/PNG
checks are preserved in `../premium_dashboard/`; this audit does not change the UI.

Advisory checks reflect the public database at audit time; the local package was
reviewed as source, not certified by a package vulnerability database. The credential
scan covers high-confidence text patterns and no Git history exists to scan. Neither
is an exhaustive security guarantee. No live external data is required by the app.

## Scientific findings you can defend

- Public UCI source hashes,48,842 raw rows,52 removed exact records and 48,790 cleaned
  observations reconcile. Split sizes 39,032/9,758 are reproducible; no row/full-record
  overlap occurs. Two conflicting complete 14-field profiles cross the split.
- Frozen pipeline metrics exactly match the historical report. Strict score >0.50
  matches the saved classifier predictions, including its tie convention.
- Calibration diagnostics and every numeric subgroup result reconcile; rates have
  correct actual-positive/negative denominators. New sex×race diagnostics cover 10
  groups (minimum sample size 32), with small-denominator cautions.
- Bias is not “fixed” by removing sex/race or editing predictions after test review.
  Observed sex differences are material; the report shows them and describes what
  a valid mitigation study would require.
- The feature-group/CV/search structure is sound as the declared historical
  experiment. Reused selection folds, limited search and the already inspected
  test set prevent unbiased selected-CV or fresh-independent-validation claims.

## Cross-environment finding and repair

The Mac rerun selects the same Expanded + interaction group and selected parameters.
Its accuracy is 83.2548%, F1 .6994113 and AUC .9141468, versus frozen 83.2240%,
.6990256 and .9141796. Eleven test classifications differ, net three more correct.
Repeated same-Mac fits have no class changes; the exact low-level reason for the
cross-environment difference is unisolated. This is not a newly improved model to
promote or evidence of contemporary predictive validity.

`finalize_research` still defaults to strict reconciliation. The explicit `record`
policy records genuine differences and a false match flag; it cannot silently claim
exact equality. `verify_research.py` runs this in a fresh copied project/kernel and
retains outputs separately. Source hashes and execution policy are recorded. The
old artifact/report release is never overwritten by the verification runner.

## Verified fixes and release changes

- `train.save_model` rejects a single-class estimator or research feature group
  whose non-dropped inputs cannot be supplied by the final inference schema,
  **before touching existing release files**. Positive supported-model round-trip
  parity is also tested. Current saved model predictions are unaffected.
- Reconciliation now supports explicit difference recording for isolated scientific
  verification; strict default behavior remains. No tolerance was relaxed to label
  a mismatch as equality.
- The stale literal 50-test README writer is retired. Its unchanged original lives
  in `archive/maintenance/finalize_delivery_20261022.py` for provenance.
- README, data-access prose, research report, model/data cards, release instructions
  and interview guide distinguish scopes and current vs historical evidence.
- Portable default pytest excludes only the explicitly marked enclosing-workspace
  audit. Its original manifest is untouched. The absent original root `.gitignore`
  remains a documented historical preservation gap—not a fabricated pass.
- Project-only ignore/license/attribution/Python configuration and read-only GitHub
  Actions checks added. CI follows the pinned lock. Remote execution is unverified.
- Prepared source release omits environments, caches, local secrets and duplicated
  runtime snapshots. Public data, trusted small model and notebook outputs remain.

## Preservation

All 170 original data/artifact/report/validated-archive files checked against the
audit baseline remain byte-identical. Original notebook code hash is unchanged;
only its public-data access prose was corrected. Intentional source/document edits
and the exact preservation scope are listed in [preservation.json](preservation.json).

## Deliverables and use

- [Research report](../RESEARCH_REPORT.md): methodology, findings, bias and limits.
- [Interview defense](../INTERVIEW_DEFENSE.md): opening narrative and challenging Q&A.
- [Release instructions](../GITHUB_RELEASE.md): prepared folder and publication scope.
- Prepared local source: `dist/CensusIncomeML/` (about 35 MB before local runtime caches).

Use the prepared folder's contents as a new repository root. No Git initialization,
commit, push or deployment has been performed. Owner review and an actual future
GitHub CI run remain publication actions, not locally verified achievements.

Finite tests cannot prove absence of every defect or guarantee future predictions.
The readiness claim is supported by the specific executed checks and transparent
research boundaries above; it is not 100% statistical or operational certainty.
