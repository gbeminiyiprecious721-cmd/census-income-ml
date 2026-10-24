# Deployment preparation

This document prepares a research demonstration for possible deployment. It does not record publication or establish production readiness. No deployment is authorised or performed by these instructions.

## Runtime and entry point

Use Python 3.12, the project dependency specifications and `.python-version`. From the project root, the entry point is `app/main.py` and local startup is:

```bash
python -m pip install -e '.[dev]'
python -m streamlit run app/main.py
```

For Streamlit Community Cloud, configure the main file as `app/main.py` if `CensusIncome-ML` is the repository root. If the repository root is its parent, select `CensusIncome-ML/app/main.py` and ensure dependency discovery and package import setup use the corresponding project directory. Do not upload the surrounding unrelated workspace merely to obtain this path.

## Release contents and checks

A runnable release needs the application and package code, compatible runtime dependencies, generated reports, the trusted fitted pipeline and its matching manifest, and any source aggregates required by the explorer. Confirm that `requirements.txt` is discoverable at the deployment root or beside the selected app file, according to the hosting layout. Verify source-layout package imports in a clean environment rather than relying on a developer's editable install.

Before publication, run the automated suite, execute the research notebook through the isolated verification runner, verify fresh/historical reconciliation, check artifact checksums and versions, and inspect all pages and a narrow layout. Confirm that missing/invalid artifacts yield useful errors and no request initiates training. Confirm that charts and download files reflect the same run. Record actual evidence in verification reports; this checklist is not evidence of a pass.

No API keys or other secrets are required for the local historical-data demonstration. Avoid adding credentials, personal records, virtual environments or untrusted pickle/joblib uploads to a release.

## Operational boundaries

Retain visible historical-data, uncalibrated-score and consequential-use limitations, including small subgroup uncertainty. Keep model and manifest versions coupled. Provision memory for loading the fitted forest; do not run notebook search during app startup. A checksum does not authenticate an attacker-controlled artifact/manifest pair. A public deployment would additionally need a deliberate artifact distribution policy, dependency maintenance and hosting verification.

External validation, calibration, contemporary data suitability, security review and a defensible use case remain unresolved. Hosting a portfolio demonstration would not resolve these research limitations or establish production fairness.

## Historical local verification —2026-10-22

Python 3.12.14: editable package installation and dependency consistency check passed. The notebook completed all 14 code cells, and all 50 tests passed, including artifact integrity checks, prediction parity, missing-report states and all six AppTest views. Streamlit returned HTTP 200 `ok` on its local health endpoint. The fitted compressed artifact is 6,113,711 bytes. Runtime requirements are pinned separately from the full research environment lock. Browser screenshots are in `reports/screenshots/`. These are local Windows observations; clean Linux/cloud installation, host memory limits and public hosting remain unverified. No publication has occurred.

Observed local Streamlit resident working set after browser navigation and prediction: 32.9 MiB across the interpreter and launcher. This Windows working-set snapshot is not a peak allocation measurement or a hosting memory guarantee; see reports/runtime_memory.json.

## Current portfolio audit —2026-10-24

The current local audit, clean source-checkout evidence and cross-platform refit differences are documented in [portfolio audit](portfolio_audit/REPORT.md). Start the saved dashboard with `scripts/publish_run.py`; no training is required. The historical50-test Windows observations above are preserved provenance, not the current test count. GitHub Actions is configured but not remotely executed. Hosting remains unverified.
