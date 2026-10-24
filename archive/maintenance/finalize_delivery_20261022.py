"""Refresh delivery documentation and independently check preserved evidence."""
from pathlib import Path
import hashlib
import json
import os
import urllib.request
import xml.etree.ElementTree as ET
import nbformat

ROOT = Path(__file__).resolve().parents[1]
def read(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

manifest = read("reports/preservation_manifest.json")
for name, expected in manifest.items():
    assert sha(ROOT.parent / name) == expected["sha256"], name
    assert sha(ROOT / "archive/validated_project" / name) == expected["sha256"], name
nb = nbformat.read(ROOT / "notebooks/01_income_classification_research.ipynb", as_version=4)
nbformat.validate(nb)
code = [c for c in nb.cells if c.cell_type == "code"]
assert [c.execution_count for c in code] == list(range(1, 15))
assert not any(o.output_type == "error" for c in code for o in c.outputs)
model = read("artifacts/model_manifest.json")
summary = read("reports/summary.json")
assert sha(ROOT / "artifacts/model.joblib") == summary["model_sha256"]
suite = ET.parse(ROOT / "reports/pytest.xml").getroot()
cases = list(suite.iter("testcase"))
assert len(cases) == 50 and not list(suite.iter("failure")) and not list(suite.iter("error"))
with urllib.request.urlopen("http://127.0.0.1:8501/_stcore/health", timeout=10) as response:
    assert response.status == 200 and response.read().strip() == b"ok"
verification = {
    "run_id": summary["run_id"], "original_and_archive_files_sha256_verified": len(manifest),
    "notebook_valid": True, "executed_code_cells": len(code), "notebook_errors": 0,
    "tests_passed": len(cases), "tests_failed": 0, "model_sha256_verified": True,
    "local_http_health": "200 ok", "model_bytes": (ROOT / "artifacts/model.joblib").stat().st_size,
    "deployment": "Not performed; cloud and Linux hosting remain unverified."
}
(ROOT / "reports/final_verification.json").write_text(json.dumps(verification, indent=2) + "\n")

path = ROOT / "README.md"
text = path.read_text(encoding="utf-8")
start = text.index("\n", text.index("## Evidence and results")) + 1
end = text.index("## Architecture")
text = text[:start] + """
A clean-kernel execution on 22 October 2026 reproduced the reviewed results (run `20261022T103759Z`). The original notebook and outputs remain preserved in `archive/validated_project/`.

| Evaluation | Accuracy | F1 | ROC-AUC |
|---|---:|---:|---:|
| Fixed forest, five-fold development CV | 86.05% | 0.6760 | 0.9136 |
| F1-selected forest, five-fold development CV | 83.40% | 0.7004 | 0.9124 |
| Selected forest, held-out test reproduction | 83.22% | 0.6990 | 0.9142 |

Test precision is 0.6126 and recall is 0.8138. The 86.05% and 83.22% figures concern different forest configurations and evaluation methods. The F1 search selects a deeper, class-weighted forest with greater recall and lower precision; its comparable CV accuracy is 83.40%, only 0.18 percentage points above its test accuracy. This confirms an objective/configuration trade-off. Sampling variation and selection optimism may also contribute, but their individual effects have not been isolated.

All 14 code cells executed in sequence with zero errors and eight embedded figures. All **50 automated tests passed**. See [fresh results](reports/summary.json), [execution evidence](reports/execution_validation.json), [test log](reports/pytest.log), [final verification](reports/final_verification.json), and the [rendered notebook](reports/research_notebook.html). The previously inspected test set makes this a reproduction, not new independent confirmation.

The notebook includes held-out permutation versus impurity importance, coherent source-input perturbations, fixed threshold diagnostics, calibration diagnostics, and sex/race error rates with denominators and uncertainty. Thresholds were not selected using test performance. Preprocessing is fitted inside development folds; deterministic feature engineering uses each record alone. Reused selection folds, conflicting profiles across the preserved split and demographic disparities remain documented limitations.

## Application preview

![Research overview](reports/screenshots/01_overview.png)

The six views cover research overview, aggregate data exploration, model performance, explainability, responsible AI assessment and a hypothetical prediction sandbox. The app uses the saved fitted pipeline and never trains on page load.

""" + text[end:]
if "## Delivery inventory" not in text:
    text += "\n## Delivery inventory\n\nSee the [complete project file inventory](reports/file_inventory.txt), excluding virtual environments and disposable caches. Local verification does not establish cloud compatibility or production readiness. No deployment or publication was performed.\n"
path.write_text(text, encoding="utf-8")
path = ROOT / "docs/workspace_audit.md"
text = path.read_text().replace("The enclosing `NBA` directory contains unrelated work.", "The enclosing `NBA` directory contains the existing Adult Income project and environment files; no separate NBA application was identified during the audit.")
path.write_text(text)
path = ROOT / "docs/model_card.md"
text = path.read_text().replace("Fresh results belong to `reports/summary.json`; historical evidence must not be represented as a new execution.", "The clean-kernel run `20261022T103759Z` reproduced these metrics; actual fresh values and parameter reconciliation are recorded in `reports/summary.json`. All 14 code cells completed without errors. This is reproduction on the previously inspected test split, not fresh independent confirmation.")
path.write_text(text)
path = ROOT / "docs/deployment_preparation.md"
text = path.read_text()
if "## Local verification completed" not in text:
    text += "\n## Local verification completed\n\nPython 3.12.14: editable package installation and dependency consistency check passed. The notebook completed all 14 code cells, and all 50 tests passed, including artifact integrity checks, prediction parity, missing-report states and all six AppTest views. Streamlit returned HTTP 200 `ok` on its local health endpoint. The fitted compressed artifact is 6,113,711 bytes. Runtime requirements are pinned separately from the full research environment lock. Browser screenshots are in `reports/screenshots/`. These are local Windows observations; clean Linux/cloud installation, host memory limits and public hosting remain unverified. No publication has occurred.\n"
path.write_text(text)
print(json.dumps(verification, indent=2))

# This inventory intentionally excludes installed dependencies and disposable caches.
excluded = {".venv", ".uv-cache", ".jupyter", ".ipython", ".matplotlib", "__pycache__", ".pytest_cache", ".git"}
files = []
for directory, subdirs, names in os.walk(ROOT):
    subdirs[:] = [d for d in subdirs if d not in excluded and not d.endswith(".egg-info")]
    files.extend((Path(directory) / name).relative_to(ROOT).as_posix() for name in names)
for name in ("reports/file_inventory.txt", "reports/file_inventory.json"):
    if name not in files:
        files.append(name)
files.sort()
(ROOT / "reports/file_inventory.txt").write_text("\n".join(files) + "\n")
(ROOT / "reports/file_inventory.json").write_text(json.dumps({"excluded_directories": sorted(excluded), "files": files}, indent=2) + "\n")
