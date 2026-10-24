# GitHub release preparation

The intended repository root is **CensusIncome-ML**, not the enclosing NBA workspace.
No Git initialization, commit, push or public deployment was performed by this audit.

## Prepared source folder

```sh
python scripts/prepare_release.py
```

This creates `dist/CensusIncomeML/`, a self-contained source checkout with public
source data, small trusted model, reports, executed notebook, tests and selected
portfolio documents. It omits virtual environments, caches, egg-info, local secrets,
agent handoff/backups and generated research-store copies. No files exceed 50 MiB.
The `.release-stage.json` records included paths and SHA-256 values. Rebuild using
`--replace` only for a previously generated stage; the working source is never removed.

Use that folder's contents for the new repository. Preserve the small model and
its coupled manifest, historical reports and notebook outputs. Regenerate local
snapshots using `scripts/publish_run.py`; never publish an environment directory.
The source model is trusted maintainer-created data, not an untrusted upload.

## Release gates

- Install pinned dependencies using Python 3.12 in a fresh environment.
- Publish the existing local snapshot; verify the dashboard starts without training.
- Run default pytest and `scripts/audit_project.py` from the prepared folder.
- Read the current [audit report](portfolio_audit/REPORT.md), especially cross-platform
  refit differences and the external historical preservation gap.
- Check the README image, notebook rendering, research report, attribution and license.
- Review the staged file inventory for any personal material before publication.

Current local evidence is in `docs/portfolio_audit/`. GitHub Actions is configured
with read-only repository permissions and project tests. A remote green run is not
claimed before publication. Local macOS verification is not Linux/cloud validation.

## Test scopes

Default pytest checks the standalone project. The separately marked `workspace`
test verifies original files in the enclosing historical workspace, which cannot
be provided by a standalone clone. That original check still reports the already
missing original `.gitignore`; the manifest has not been altered. Run it explicitly
with `python -m pytest -m workspace` only in that historical workspace. This is a
provenance gap, not permission to fabricate evidence or weaken project tests.

## Repository presentation

Suggested name: `census-income-ml`. Suggested description: “Reproducible historical
income-classification research, validated inference and an interactive evaluation
dashboard.” Suggested topics: `machine-learning`, `scikit-learn`, `random-forest`,
`model-evaluation`, `streamlit`, `responsible-ai`, `reproducible-research`.

Create genuine commits from this point forward; do not manufacture a development
history. A public demo would improve accessibility, but requires a separately
verified hosting decision. There is no deployed URL to advertise yet. The screenshot
and rendered notebook provide a local-workflow fallback.
